#!/usr/bin/env python3
"""
calibrate_vla_thresholds.py
==============================
CRÍTICO: corre esto ANTES de confiar en los resultados de
benchmark_comparison.py. Los umbrales en VLA_THRESHOLDS son un punto
de partida sin calibrar -- este script corre OpenVLA sobre un puñado
de ventanas con ground truth conocido y muestra la distribución REAL
de (posición, rotación, gripper) por clase, para que ajustes los
umbrales con datos en vez de a ciegas.

Uso:
    python3 calibrate_vla_thresholds.py \
        --crops-dirs crops_cache/P01 crops_cache/P02 \
        --videos-dir kitchen/EPIC-KITCHENS \
        --samples-per-class 5
"""

import argparse
import csv
import random
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-dirs", nargs="+", required=True,
                        help="Carpetas con metadata.csv (para saber qué "
                             "video_id + frame_index corresponde a cada "
                             "clase real)")
    parser.add_argument("--model-id", default="openvla/openvla-7b")
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Cargando {args.model_id}...")
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_id, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True,
    ).to("cuda")

    rows_by_class = defaultdict(list)
    for crops_dir in args.crops_dirs:
        crops_dir = Path(crops_dir)
        meta_path = crops_dir / "metadata.csv"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            for row in csv.DictReader(f):
                row["_crops_dir"] = str(crops_dir)
                rows_by_class[row["label"]].append(row)

    print(f"\nClases disponibles: {sorted(rows_by_class.keys())}")

    stats_by_class = defaultdict(list)

    for label, rows in rows_by_class.items():
        sample = random.sample(rows, min(args.samples_per_class, len(rows)))
        print(f"\n── Clase '{label}' ({len(sample)} muestras) ──────────")

        for row in sample:
            crops_dir = Path(row["_crops_dir"])
            frame_files = row["frame_files"].split("|")
            first_frame_path = crops_dir / frame_files[0]
            crop = cv2.imread(str(first_frame_path))
            if crop is None:
                continue

            image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            prompt = f"In: What action should the robot take to {label}?\nOut:"
            inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)

            with torch.no_grad():
                action = model.predict_action(
                    **inputs, unnorm_key="bridge_orig", do_sample=False)
            action = np.asarray(action)

            dx, dy, dz, droll, dpitch, dyaw, gripper = action
            pos_mag = float(np.linalg.norm([dx, dy, dz]))
            rot_mag = float(np.linalg.norm([droll, dpitch, dyaw]))

            stats_by_class[label].append({
                "pos_mag": pos_mag, "rot_mag": rot_mag, "gripper": gripper,
            })
            print(f"  pos_mag={pos_mag:.4f}  rot_mag={rot_mag:.4f}  gripper={gripper:.4f}")

    print("\n" + "=" * 60)
    print("RESUMEN: distribución real por clase (media ± desv. estándar)")
    print("=" * 60)
    print("Usa estos números para ajustar VLA_THRESHOLDS en "
          "benchmark_comparison.py -- por ejemplo, si 'grasp' tiene "
          "gripper promedio -0.15 (no -0.3+), baja gripper_close a -0.15.")
    for label, stats in stats_by_class.items():
        if not stats:
            continue
        pos_vals = [s["pos_mag"] for s in stats]
        rot_vals = [s["rot_mag"] for s in stats]
        grip_vals = [s["gripper"] for s in stats]
        print(f"\n{label}:")
        print(f"  pos_mag : {np.mean(pos_vals):.4f} ± {np.std(pos_vals):.4f}")
        print(f"  rot_mag : {np.mean(rot_vals):.4f} ± {np.std(rot_vals):.4f}")
        print(f"  gripper : {np.mean(grip_vals):.4f} ± {np.std(grip_vals):.4f}")


if __name__ == "__main__":
    main()