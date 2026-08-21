"""
finetune_model.py
====================
CLIP (backbone) + atención temporal aprendida + features de movimiento +
cabeza clasificadora.

Dos modos de adaptación del backbone:

  1. Descongelamiento selectivo (n_unfrozen_layers): las últimas N capas
     se entrenan COMPLETAS. Encontramos que con 12/12 (todas) hay olvido
     catastrófico (peor accuracy que con 6-8) -- el modelo destruye
     características generales útiles al reentrenar capas tempranas con
     pocos datos.

  2. LoRA (use_lora=True): en vez de reentrenar capas completas, agrega
     matrices de bajo rango pequeñas a TODAS las capas de atención/MLP,
     sin tocar los pesos originales de CLIP. Permite adaptar las 12 capas
     sin el riesgo de olvido catastrófico (los pesos base nunca cambian),
     y con muchos menos parámetros entrenables -- lo que además reduce el
     riesgo de sobreajustarse a las pocas identidades de personas
     disponibles en el set de entrenamiento. Compatible con backbones más
     grandes (ViT-L/14) sin disparar el costo de VRAM del fine-tuning
     completo.
"""

import torch
import torch.nn as nn
from transformers import CLIPVisionModel

try:
    from peft import LoraConfig, get_peft_model
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False


class TemporalAttentionPooling(nn.Module):
    def __init__(self, embed_dim: int = 768, n_heads: int = 8):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=embed_dim * 2,
            dropout=0.1, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, frame_embeds: torch.Tensor) -> torch.Tensor:
        batch_size = frame_embeds.shape[0]
        cls = self.cls_token.expand(batch_size, -1, -1)
        seq = torch.cat([cls, frame_embeds], dim=1)
        out = self.transformer(seq)
        return out[:, 0, :]


class ActionClassifierFinetune(nn.Module):
    def __init__(self, n_classes: int, clip_model_id: str = "openai/clip-vit-base-patch32",
                 n_unfrozen_layers: int = 6, n_motion_features: int = 4,
                 dropout: float = 0.3, use_lora: bool = False,
                 lora_r: int = 8, lora_alpha: int = 16, lora_dropout: float = 0.05):
        super().__init__()
        backbone = CLIPVisionModel.from_pretrained(clip_model_id)
        embed_dim = backbone.config.hidden_size
        self.use_lora = use_lora

        if use_lora:
            if not _PEFT_AVAILABLE:
                raise ImportError("Falta 'peft': pip install peft --break-system-packages")
            for param in backbone.parameters():
                param.requires_grad = False
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj",
                                "fc1", "fc2"],
                bias="none",
            )
            self.vision_backbone = get_peft_model(backbone, lora_config)
        else:
            self.vision_backbone = backbone
            if hasattr(self.vision_backbone, "vision_model"):
                encoder = self.vision_backbone.vision_model.encoder
                post_layernorm = self.vision_backbone.vision_model.post_layernorm
            else:
                encoder = self.vision_backbone.encoder
                post_layernorm = self.vision_backbone.post_layernorm

            for param in self.vision_backbone.parameters():
                param.requires_grad = False
            total_layers = len(encoder.layers)
            n_unfrozen_layers = min(n_unfrozen_layers, total_layers)
            for layer in encoder.layers[total_layers - n_unfrozen_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in post_layernorm.parameters():
                param.requires_grad = True

        self.temporal_pooling = TemporalAttentionPooling(embed_dim=embed_dim)

        combined_dim = embed_dim + n_motion_features
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, pixel_values: torch.Tensor, motion_features: torch.Tensor) -> torch.Tensor:
        batch_size, n_frames = pixel_values.shape[:2]
        flat_pixels = pixel_values.reshape(batch_size * n_frames, *pixel_values.shape[2:])

        vision_out = self.vision_backbone(pixel_values=flat_pixels)
        frame_embeds = vision_out.pooler_output
        frame_embeds = frame_embeds.view(batch_size, n_frames, -1)

        pooled = self.temporal_pooling(frame_embeds)
        combined = torch.cat([pooled, motion_features], dim=1)
        return self.classifier(combined)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())