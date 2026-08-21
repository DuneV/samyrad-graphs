"""
finetune_dataset.py
======================
Dataset de PyTorch que lee las ventanas cacheadas por cache_crops.py.

NUEVO respecto a la versión anterior: data augmentation en modo train
(color jitter + recorte aleatorio leve). El loss de entrenamiento bajaba
a 0.09-0.13 mientras val_acc se estancaba — señal clásica de overfitting
(el modelo memoriza los crops exactos en vez de aprender el patrón
general). Ver siempre los mismos 4 frames idénticos en cada época hace
que memorizar sea más fácil que generalizar; con augmentation cada
época ve una versión ligeramente distinta de cada ventana.
"""

import csv
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from transformers import CLIPImageProcessor


class ActionWindowDataset(Dataset):
    def __init__(self, cache_dir: str, label_encoder,
                 clip_model_id: str = "openai/clip-vit-base-patch32",
                 allowed_window_ids: Optional[set] = None,
                 remap: Optional[Dict[str, str]] = None,
                 augment: bool = False):
        """
        augment: si True, aplica color jitter + recorte aleatorio leve a
            cada frame (independiente por frame, para no perder la
            variación temporal real de la ventana). Usar solo en el
            dataset de TRAIN, nunca en test/validación.
        """
        self.cache_dir = Path(cache_dir)
        self.label_encoder = label_encoder
        self.processor = CLIPImageProcessor.from_pretrained(clip_model_id)
        self.remap = remap or {}
        self.augment = augment

        if augment:
            self.aug_transform = transforms.Compose([
                transforms.RandomApply([transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)], p=0.8),
                transforms.RandomResizedCrop(
                    size=224, scale=(0.85, 1.0), ratio=(0.95, 1.05)),
            ])
        else:
            self.aug_transform = None

        self.rows: List[dict] = []
        with open(self.cache_dir / "metadata.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if allowed_window_ids is not None and int(row["window_id"]) not in allowed_window_ids:
                    continue
                label = self.remap.get(row["label"], row["label"])
                if label not in label_encoder.classes_:
                    continue
                row["label"] = label
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        frame_files = row["frame_files"].split("|")

        images = [Image.open(self.cache_dir / fn).convert("RGB") for fn in frame_files]
        if self.augment and self.aug_transform is not None:
            images = [self.aug_transform(img) for img in images]

        processed = self.processor(images=images, return_tensors="pt")
        pixel_values = processed["pixel_values"]

        motion_features = torch.tensor([
            float(row["net_disp"]), float(row["curvature"]),
            float(row["area_change"]), float(row["angular_range"]),
        ], dtype=torch.float32)

        label_idx = self.label_encoder.transform([row["label"]])[0]

        return {
            "pixel_values": pixel_values,
            "motion_features": motion_features,
            "label": torch.tensor(label_idx, dtype=torch.long),
        }

    def get_all_labels(self) -> List[str]:
        return [r["label"] for r in self.rows]

    def get_all_window_ids(self) -> List[int]:
        return [int(r["window_id"]) for r in self.rows]