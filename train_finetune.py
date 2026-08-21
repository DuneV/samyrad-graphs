#!/usr/bin/env python3
"""
train_finetune.py
====================
Fine-tuning real. Nuevo respecto a la versión anterior (que se estancó
en 62-65% con overfitting visible: loss de train bajando a 0.09-0.13
mientras val_acc no mejoraba):

  1. Data augmentation en train (color jitter + recorte aleatorio leve),
     vía ActionWindowDataset(augment=True) — combate la memorización.
  2. --weight-decay (default 0.01): regularización L2 en AdamW.
  3. --label-smoothing (default 0.1): evita que el modelo se sobre-
     confíe en clases ambiguas (pour, displace), suaviza el objetivo.
  4. Decaimiento coseno del LR después del warmup, en vez de LR
     constante — converge a un mínimo más estable en vez de seguir
     "empujando" con LR alto una vez que ya aprendió lo esencial.

Uso:
    python3 train_finetune.py \
        --cache-dirs crops_cache/P01 crops_cache/P02 ... \
        --output action_classifier_finetuned.pth \
        --epochs 25 --batch-size 8 --unfrozen-layers 8
"""

import argparse
import csv
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from finetune_model import ActionClassifierFinetune
from finetune_dataset import ActionWindowDataset

DEFAULT_REMAP = {
    "pull": "displace", "push": "displace",
    "remove_from": "displace", "inspect": "displace",
    "close": "actuate", "open": "actuate",
}


def read_all_labels(cache_dirs, remap):
    all_labels = []
    all_window_ids = []
    all_video_ids = []
    for i, cd in enumerate(cache_dirs):
        with open(Path(cd) / "metadata.csv") as f:
            for row in csv.DictReader(f):
                label = remap.get(row["label"], row["label"])
                all_labels.append(label)
                all_window_ids.append((i, int(row["window_id"])))
                all_video_ids.append(row["video_id"])
    return all_labels, all_window_ids, all_video_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dirs", nargs="+", required=True)
    parser.add_argument("--output", default="action_classifier_finetuned.pth")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--unfrozen-layers", type=int, default=6)
    parser.add_argument("--use-lora", action="store_true",
                        help="Usa LoRA (adapta las 12 capas con matrices "
                             "pequeñas, sin tocar los pesos originales de "
                             "CLIP) en vez de descongelar capas completas. "
                             "Más robusto contra olvido catastrófico y "
                             "sobreajuste a las pocas identidades de "
                             "personas del set de entrenamiento. Combínalo "
                             "con --clip-model openai/clip-vit-large-patch14 "
                             "para usar el backbone más grande sin disparar "
                             "el costo de VRAM.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-samples-per-class", type=int, default=20)
    parser.add_argument("--no-remap", action="store_true")
    parser.add_argument("--no-augment", action="store_true",
                        help="Desactiva data augmentation (para comparar)")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--balance-mode", choices=["none", "full"], default="none")
    parser.add_argument("--group-by", choices=["video", "participant"], default="participant",
                        help="video: solo garantiza que un video no quede "
                             "partido entre train/test (participantes "
                             "pueden repetirse en ambos — el modelo ya vio "
                             "esa cocina, esas manos, en otro video). "
                             "participant (default, más riguroso): "
                             "garantiza que TODOS los videos de un "
                             "participante queden del mismo lado — prueba "
                             "real de generalización a gente nunca vista, "
                             "comparable con train_action_classifier_grouped.py.")
    args = parser.parse_args()

    remap = {} if args.no_remap else DEFAULT_REMAP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | fp16: {args.fp16} | augment: {not args.no_augment}")

    all_labels, all_window_ids, all_video_ids = read_all_labels(args.cache_dirs, remap)
    label_counts = Counter(all_labels)
    print("Distribución completa:", dict(label_counts))

    valid_classes = sorted(c for c, n in label_counts.items()
                           if n >= args.min_samples_per_class)
    print(f"Clases usadas ({len(valid_classes)}): {valid_classes}")

    le = LabelEncoder()
    le.fit(valid_classes)

    filtered = [(i, lbl, vid) for i, (lbl, vid) in enumerate(zip(all_labels, all_video_ids))
               if lbl in valid_classes]
    filtered_idx = [f[0] for f in filtered]
    filtered_labels = [f[1] for f in filtered]
    filtered_video_ids = [f[2] for f in filtered]

    if args.group_by == "participant":
        # Ej. "P01_11" -> "P01". Esta es la prueba rigurosa: el modelo
        # nunca vio a este participante en ningún video, ni siquiera uno
        # distinto de la misma persona/cocina/iluminación.
        participant_re = re.compile(r"^(P\d{2})")
        filtered_groups = [
            (participant_re.match(vid).group(1) if participant_re.match(vid) else vid)
            for vid in filtered_video_ids
        ]
    else:
        filtered_groups = filtered_video_ids

    # IMPORTANTE: split agrupado (por video o por participante, ver
    # --group-by). Cada segmento de acción genera hasta
    # --windows-per-segment ventanas muy parecidas entre sí (mismo
    # objeto, mano, frames cercanos). Un split por ventana suelta podía
    # dejar 3 ventanas de un segmento en train y la 4ta (casi idéntica)
    # en test — el modelo "reconocía" una variante de algo que ya vio,
    # no generalizaba a una acción nueva.
    n_splits = max(2, round(1 / args.test_size))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    train_pos, test_pos = next(sgkf.split(
        np.zeros(len(filtered_idx)), filtered_labels, filtered_groups))

    train_groups_check = set(filtered_groups[p] for p in train_pos)
    test_groups_check = set(filtered_groups[p] for p in test_pos)
    overlap_check = train_groups_check & test_groups_check
    assert not overlap_check, f"Fuga detectada: grupos en ambos splits: {overlap_check}"
    print(f"Split agrupado por {args.group_by}: {len(train_groups_check)} en train "
          f"({sorted(train_groups_check)}), {len(test_groups_check)} en test "
          f"({sorted(test_groups_check)}) — 0 solapados, verificado")

    train_window_ids_by_dir = defaultdict(set)
    test_window_ids_by_dir = defaultdict(set)
    for pos in train_pos:
        dir_idx, wid = all_window_ids[filtered_idx[pos]]
        train_window_ids_by_dir[dir_idx].add(wid)
    for pos in test_pos:
        dir_idx, wid = all_window_ids[filtered_idx[pos]]
        test_window_ids_by_dir[dir_idx].add(wid)

    train_datasets, test_datasets = [], []
    for i, cd in enumerate(args.cache_dirs):
        if train_window_ids_by_dir[i]:
            train_datasets.append(ActionWindowDataset(
                cd, le, args.clip_model, train_window_ids_by_dir[i], remap,
                augment=not args.no_augment))
        if test_window_ids_by_dir[i]:
            test_datasets.append(ActionWindowDataset(
                cd, le, args.clip_model, test_window_ids_by_dir[i], remap,
                augment=False))

    train_ds = ConcatDataset(train_datasets)
    test_ds = ConcatDataset(test_datasets)
    print(f"Train: {len(train_ds)} ventanas | Test: {len(test_ds)} ventanas")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=(device == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=2, pin_memory=(device == "cuda"))

    model = ActionClassifierFinetune(
        n_classes=len(valid_classes), clip_model_id=args.clip_model,
        n_unfrozen_layers=args.unfrozen_layers,
        use_lora=args.use_lora, lora_r=args.lora_r,
        lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
    ).to(device)
    print(f"Adaptación del backbone: {'LoRA (r=' + str(args.lora_r) + ')' if args.use_lora else f'{args.unfrozen_layers} capas descongeladas'}")
    print(f"Parámetros totales: {model.total_parameter_count():,}")
    print(f"Parámetros entrenables: {model.trainable_parameter_count():,} "
          f"({model.trainable_parameter_count()/model.total_parameter_count()*100:.1f}%)")

    if args.balance_mode == "full":
        class_weights = compute_class_weight(
            "balanced", classes=np.arange(len(valid_classes)),
            y=le.transform(filtered_labels))
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t, label_smoothing=args.label_smoothing)
        print("Balanceo de clases: COMPLETO")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        print("Balanceo de clases: NINGUNO (optimizando accuracy total)")
    print(f"Label smoothing: {args.label_smoothing} | Weight decay: {args.weight_decay}")

    backbone_params = [p for p in model.vision_backbone.parameters() if p.requires_grad]
    head_params = (list(model.temporal_pooling.parameters())
                   + list(model.classifier.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler("cuda", enabled=(args.fp16 and device == "cuda"))

    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    warmup_steps = min(args.warmup_steps, max(total_steps // 10, 1))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(progress, 1.0)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step = 0

    best_acc = 0.0
    best_true, best_preds = None, None
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device)
            motion_features = batch["motion_features"].to(device)
            labels = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=(args.fp16 and device == "cuda")):
                logits = model(pixel_values, motion_features)
                loss = criterion(logits, labels) / args.grad_accum_steps

            scaler.scale(loss).backward()
            total_loss += loss.item() * args.grad_accum_steps

            if (batch_idx + 1) % args.grad_accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

        avg_loss = total_loss / max(len(train_loader), 1)

        model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for batch in test_loader:
                pixel_values = batch["pixel_values"].to(device)
                motion_features = batch["motion_features"].to(device)
                labels = batch["label"]
                logits = model(pixel_values, motion_features)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(labels.numpy())

        acc = accuracy_score(all_true, all_preds)
        bacc = balanced_accuracy_score(all_true, all_preds)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | "
              f"val_acc={acc*100:.1f}% | val_balanced_acc={bacc*100:.1f}% | "
              f"lr={current_lr:.2e} | {elapsed:.0f}s")

        if acc > best_acc:
            best_acc = acc
            best_true, best_preds = list(all_true), list(all_preds)
            if (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "label_encoder_classes": list(le.classes_),
                    "epoch": epoch,
                    "val_accuracy": acc,
                }, args.output)
                print(f"  -> Checkpoint guardado (mejor val_acc hasta ahora: {acc*100:.1f}%)")

    print("\n── Reporte final (MEJOR checkpoint, no la última época) ──────────")
    print(classification_report(
        [le.classes_[i] for i in best_true],
        [le.classes_[i] for i in best_preds],
        zero_division=0))
    print(f"Mejor accuracy de validación: {best_acc*100:.1f}%")


if __name__ == "__main__":
    main()