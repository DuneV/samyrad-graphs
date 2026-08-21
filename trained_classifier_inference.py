"""
trained_classifier_inference.py
==================================
Carga CUALQUIERA de tus checkpoints ya entrenados (los .joblib de
embeddings CLIP + HistGradientBoosting, o los .pth de fine-tuning real)
y expone la MISMA interfaz: classify_window(crops, motion_features).

Esto es lo que faltaba conectar: benchmark_comparison.py (y en general
tu pipeline en vivo) seguían usando CLIP zero-shot porque el pipeline
en vivo clasifica frame a frame, mientras tus checkpoints esperan una
ventana de N frames + 4-6 features de movimiento -- interfaces
distintas. Este módulo resuelve esa diferencia.

Uso:
    clf = TrainedActionClassifier.load("action_classifier_finetuned_final.pth")
    # o:
    clf = TrainedActionClassifier.load("action_classifier_v3.joblib")

    action, confidence = clf.classify_window(crops, motion_features)
"""

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class TrainedActionClassifier:
    """
    Interfaz común sin importar el tipo de checkpoint real detrás.
    Detecta automáticamente si es un modelo sklearn (.joblib, entrenado
    sobre embeddings CLIP congelados + features de movimiento) o un
    modelo de fine-tuning real (.pth, con backbone CLIP incluido).
    """

    def __init__(self, backend: str, model, device: str = "cuda"):
        self.backend = backend   # "sklearn" o "pytorch"
        self.model = model
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: str, device: str = "cuda") -> "TrainedActionClassifier":
        path = Path(checkpoint_path)
        if path.suffix == ".joblib":
            return cls._load_sklearn(path)
        elif path.suffix in (".pth", ".pt"):
            return cls._load_pytorch(path, device)
        else:
            raise ValueError(f"Extensión no reconocida: {path.suffix} "
                            f"(esperaba .joblib o .pth/.pt)")

    @classmethod
    def _load_sklearn(cls, path: Path) -> "TrainedActionClassifier":
        import joblib
        bundle = joblib.load(path)
        obj = cls(backend="sklearn", model=bundle)
        obj._clip_model = None
        obj._clip_processor = None
        return obj

    @classmethod
    def _load_pytorch(cls, path: Path, device: str) -> "TrainedActionClassifier":
        import torch
        from finetune_model import ActionClassifierFinetune

        checkpoint = torch.load(path, map_location=device, weights_only=False)
        classes = checkpoint["label_encoder_classes"]
        state_dict = checkpoint["model_state_dict"]

        # Detecta la arquitectura real leyendo las formas de los tensores
        # guardados, en vez de asumir el default (ViT-B/32 sin LoRA) --
        # antes esto fallaba con checkpoints de LoRA/ViT-L/14 porque
        # construía un modelo con una arquitectura distinta a la guardada.
        use_lora = any("lora_A" in k for k in state_dict.keys())
        lora_r = 8
        if use_lora:
            for k, v in state_dict.items():
                if "lora_A" in k:
                    lora_r = v.shape[0]
                    break

        embed_dim = state_dict["classifier.0.weight"].shape[1] - 4  # -4 features de movimiento
        if embed_dim == 1024:
            clip_model_id = "openai/clip-vit-large-patch14"
        elif embed_dim == 768:
            clip_model_id = "openai/clip-vit-base-patch32"
        else:
            print(f"  ⚠ embed_dim={embed_dim} no reconocido, usando ViT-B/32 "
                  f"como mejor intento -- si falla, usa load_pytorch_explicit()")
            clip_model_id = "openai/clip-vit-base-patch32"

        print(f"  Arquitectura detectada: clip_model={clip_model_id}, "
              f"use_lora={use_lora}" + (f", lora_r={lora_r}" if use_lora else ""))

        model = ActionClassifierFinetune(
            n_classes=len(classes), clip_model_id=clip_model_id,
            use_lora=use_lora, lora_r=lora_r,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        obj = cls(backend="pytorch", model=model, device=device)
        obj._classes = classes
        obj._clip_processor = None
        return obj

    @classmethod
    def load_pytorch_explicit(cls, checkpoint_path: str, clip_model: str,
                              n_unfrozen_layers: int = 6, use_lora: bool = False,
                              lora_r: int = 8, device: str = "cuda"
                              ) -> "TrainedActionClassifier":
        """Usa esto si tu .pth se entrenó con una arquitectura no-default
        (LoRA, otro backbone, otro número de capas descongeladas) -- pasa
        exactamente los mismos flags que usaste en train_finetune.py."""
        import torch
        from finetune_model import ActionClassifierFinetune

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        classes = checkpoint["label_encoder_classes"]

        model = ActionClassifierFinetune(
            n_classes=len(classes), clip_model_id=clip_model,
            n_unfrozen_layers=n_unfrozen_layers, use_lora=use_lora, lora_r=lora_r,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        obj = cls(backend="pytorch", model=model, device=device)
        obj._classes = classes
        obj._clip_processor = None
        return obj

    # ── Clasificación ────────────────────────────────────────────────

    def classify_window(self, crops: List[np.ndarray],
                        motion_features: np.ndarray) -> Tuple[str, float]:
        """
        crops: lista de recortes BGR (numpy arrays), la ventana de frames
               de la interacción mano-objeto.
        motion_features: array de 4 o 6 floats (net_disp, curvature,
               area_change, angular_range[, aperture_mean, aperture_change])
               -- debe coincidir con lo que usó el checkpoint al entrenar.
        Devuelve (accion_predicha, confianza).
        """
        if self.backend == "sklearn":
            return self._classify_sklearn(crops, motion_features)
        else:
            return self._classify_pytorch(crops, motion_features)

    def _classify_sklearn(self, crops, motion_features):
        import cv2
        import torch
        from PIL import Image
        from transformers import CLIPModel, CLIPProcessor

        if self._clip_model is None:
            # Mismo backbone que se usó para extraer los embeddings de
            # entrenamiento -- ViT-L/14 (extract_training_embeddings_v2..v5)
            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14").to(self.device)
            self._clip_model.eval()
            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14")

        embeddings = []
        for crop in crops[:4]:   # las extracciones usaban hasta 4 frames
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            inputs = self._clip_processor(images=pil_img, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self._clip_model.get_image_features(**inputs)
            tensor = out if isinstance(out, torch.Tensor) else (
                getattr(out, "image_embeds", None) or getattr(out, "pooler_output"))
            embeddings.append(tensor.squeeze().cpu().numpy())

        clip_embedding = np.mean(embeddings, axis=0)
        combined = np.concatenate([clip_embedding, motion_features]).reshape(1, -1)

        bundle = self.model
        scaled = bundle["scaler"].transform(combined)
        pred_idx = bundle["classifier"].predict(scaled)[0]
        proba = bundle["classifier"].predict_proba(scaled).max() \
            if hasattr(bundle["classifier"], "predict_proba") else 1.0
        action = bundle["label_encoder"].inverse_transform([pred_idx])[0]
        return action, float(proba)

    def _classify_pytorch(self, crops, motion_features):
        import cv2
        import torch
        from PIL import Image
        from transformers import CLIPImageProcessor

        if self._clip_processor is None:
            self._clip_processor = CLIPImageProcessor.from_pretrained(
                "openai/clip-vit-base-patch32")

        images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops[:4]]
        while len(images) < 4:
            images.append(images[-1])   # rellena si hay menos de 4

        pixel_values = self._clip_processor(images=images, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.unsqueeze(0).to(self.device)   # (1, n_frames, 3, H, W)
        motion_t = torch.tensor(motion_features[:4], dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(pixel_values, motion_t)
            probs = torch.softmax(logits, dim=1)
            pred_idx = probs.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()

        action = self._classes[pred_idx]
        return action, confidence


class WindowedActionAccumulator:
    """
    Acumula una ventana de frames por cada mano en contacto con un
    objeto, y clasifica con TrainedActionClassifier cuando la ventana se
    completa (o cuando el contacto termina antes de llenarla). Reemplaza
    la clasificación frame-a-frame en vivo por clasificación por ventana,
    igual que se entrenó el modelo.
    """

    def __init__(self, classifier: TrainedActionClassifier, window_size: int = 8):
        self.classifier = classifier
        self.window_size = window_size
        self._buffers = {}   # hand_label -> dict con crops/positions/areas/angles

    def update(self, hand_label: str, crop: np.ndarray,
              wx: float, wy: float, obj_bbox, mid_mcp=None
              ) -> Optional[Tuple[str, float]]:
        import math

        if hand_label not in self._buffers:
            self._buffers[hand_label] = {
                "crops": [], "positions": [], "areas": [], "angles": [],
            }
        buf = self._buffers[hand_label]

        buf["crops"].append(crop)
        buf["positions"].append((wx, wy))
        xmin, ymin, xmax, ymax = obj_bbox
        buf["areas"].append(max(0, xmax - xmin) * max(0, ymax - ymin))
        if mid_mcp is not None:
            ox, oy = mid_mcp
            buf["angles"].append(math.atan2(oy - wy, ox - wx))
        else:
            buf["angles"].append(0.0)

        if len(buf["crops"]) >= self.window_size:
            result = self._classify_and_reset(hand_label)
            return result
        return None

    def flush(self, hand_label: str) -> Optional[Tuple[str, float]]:
        """Llamar cuando la mano deja de tocar el objeto, para
        clasificar lo que se acumuló aunque no se haya llenado la
        ventana completa."""
        buf = self._buffers.get(hand_label)
        if buf and len(buf["crops"]) >= 3:   # mínimo razonable de señal
            return self._classify_and_reset(hand_label)
        self._buffers.pop(hand_label, None)
        return None

    def _classify_and_reset(self, hand_label: str) -> Tuple[str, float]:
        buf = self._buffers.pop(hand_label)
        motion_features = self._compute_motion_features(
            buf["positions"], buf["areas"], buf["angles"])
        return self.classifier.classify_window(buf["crops"], motion_features)

    @staticmethod
    def _compute_motion_features(positions, areas, angles) -> np.ndarray:
        import math
        pts = np.array(positions, dtype=float)
        net_disp = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) >= 2 else 0.0
        path_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) >= 2 else 0.0
        curvature = 0.0 if net_disp == 0 else (path_length / net_disp) - 1.0
        area_change = 0.0
        if len(areas) >= 2 and areas[0] > 0:
            area_change = (areas[-1] - areas[0]) / areas[0]
        ang_range = 0.0
        if len(angles) >= 2:
            total = 0.0
            for a, b in zip(angles[:-1], angles[1:]):
                d = b - a
                d = (d + math.pi) % (2 * math.pi) - math.pi
                total += abs(d)
            ang_range = math.degrees(total)
        return np.array([net_disp, curvature, area_change, ang_range], dtype=np.float32)