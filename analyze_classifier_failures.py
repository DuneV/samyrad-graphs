#!/usr/bin/env python3
"""
analyze_classifier_failures.py
=================================
Corre tu clasificador entrenado sobre las ventanas YA CACHEADAS en
crops_cache/<participante>/ (no vuelve a procesar video), compara cada
predicción contra el ground truth real (columna "label" de metadata.csv,
ya viene del verbo real de EPIC-KITCHENS), y exporta los FALLOS
organizados en carpetas para que los revises visualmente.

Estructura de salida:
    failure_analysis/
        <real>_predicho_como_<predicho>/
            window_0001234/
                f0.jpg f1.jpg f2.jpg f3.jpg
                info.txt   (confianza, brillo promedio, metadata)
        summary.csv   (una fila por fallo, con diagnóstico automático)

El "diagnóstico automático" es solo un apoyo (brillo/contraste
promedio de los frames) para priorizar qué revisar primero -- el
análisis real de "por qué falla" (luminosidad, oclusión de mano,
ángulo...) lo haces tú viendo las imágenes exportadas.

Uso:
    python3 analyze_classifier_failures.py \
        --crops-dirs crops_cache/P01 crops_cache/P03 \
        --checkpoint action_classifier_finetuned_final.pth \
        --output-dir failure_analysis \
        --max-per-confusion 20
"""

import argparse
import csv
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from trained_classifier_inference import TrainedActionClassifier

DEFAULT_REMAP = {
    "pull": "displace", "push": "displace",
    "remove_from": "displace", "inspect": "displace",
    "close": "actuate", "open": "actuate",
}


def brightness_stats(crop_paths: List[Path]) -> Dict[str, float]:
    """Brillo y contraste promedio de los frames de la ventana -- pista
    rápida para priorizar revisión (no reemplaza mirar las imágenes)."""
    brightness_vals, contrast_vals = [], []
    for p in crop_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        brightness_vals.append(float(img.mean()))
        contrast_vals.append(float(img.std()))
    return {
        "brillo_promedio": round(np.mean(brightness_vals), 1) if brightness_vals else -1,
        "contraste_promedio": round(np.mean(contrast_vals), 1) if contrast_vals else -1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-dirs", nargs="+", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="failure_analysis")
    parser.add_argument("--max-per-confusion", type=int, default=20,
                        help="Máximo de ejemplos a exportar por cada par "
                             "(real, predicho) -- para no llenar el disco "
                             "si hay miles de fallos de un mismo tipo")
    parser.add_argument("--no-remap", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    remap = {} if args.no_remap else DEFAULT_REMAP
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Cargando clasificador desde {args.checkpoint}...")
    classifier = TrainedActionClassifier.load(args.checkpoint, device=args.device)

    exported_counts = defaultdict(int)
    summary_rows = []
    n_total, n_correct, n_failed, n_exported = 0, 0, 0, 0

    for crops_dir in args.crops_dirs:
        crops_dir = Path(crops_dir)
        metadata_path = crops_dir / "metadata.csv"
        if not metadata_path.exists():
            print(f"⚠ {metadata_path} no existe, se salta")
            continue

        print(f"\nProcesando {crops_dir}...")
        with open(metadata_path) as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            true_label = remap.get(row["label"], row["label"])
            frame_files = row["frame_files"].split("|")
            crop_paths = [crops_dir / fn for fn in frame_files]
            crops = [cv2.imread(str(p)) for p in crop_paths]
            if any(c is None for c in crops):
                continue

            motion_features = np.array([
                float(row["net_disp"]), float(row["curvature"]),
                float(row["area_change"]), float(row["angular_range"]),
            ], dtype=np.float32)

            pred_action, confidence = classifier.classify_window(crops, motion_features)

            n_total += 1
            if pred_action == true_label:
                n_correct += 1
                continue

            n_failed += 1
            confusion_key = f"{true_label}_predicho_como_{pred_action}"

            stats = brightness_stats(crop_paths)
            summary_rows.append({
                "window_id": row["window_id"], "video_id": row["video_id"],
                "real": true_label, "predicho": pred_action,
                "confianza": round(confidence, 3),
                **stats,
            })

            if exported_counts[confusion_key] < args.max_per_confusion:
                dest_dir = out_dir / confusion_key / f"window_{row['window_id']}"
                dest_dir.mkdir(parents=True, exist_ok=True)
                for i, src in enumerate(crop_paths):
                    shutil.copy(src, dest_dir / f"f{i}.jpg")
                (dest_dir / "info.txt").write_text(
                    f"video_id: {row['video_id']}\n"
                    f"real: {true_label}\n"
                    f"predicho: {pred_action}\n"
                    f"confianza: {confidence:.3f}\n"
                    f"brillo_promedio: {stats['brillo_promedio']}\n"
                    f"contraste_promedio: {stats['contraste_promedio']}\n"
                    f"epic_verb original: {row.get('epic_verb', '?')}\n"
                )
                exported_counts[confusion_key] += 1
                n_exported += 1

    csv_path = out_dir / "summary.csv"
    if summary_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"\n{'='*60}")
    print(f"Total evaluado: {n_total}")
    if n_total:
        print(f"Correctos: {n_correct} ({n_correct/n_total*100:.1f}%)")
    print(f"Fallos: {n_failed}")
    print(f"Fallos exportados a disco (con imágenes): {n_exported}")
    print(f"\nDesglose por tipo de confusión (real -> predicho), ordenado por frecuencia:")
    conf_counts = defaultdict(int)
    for row in summary_rows:
        conf_counts[(row["real"], row["predicho"])] += 1
    for (real, pred), count in sorted(conf_counts.items(), key=lambda x: -x[1]):
        print(f"  {real:12s} -> {pred:12s}: {count} veces")
    print(f"\nResumen completo en: {csv_path}")
    print(f"Imágenes de fallos en: {out_dir}/<real>_predicho_como_<predicho>/")


if __name__ == "__main__":
    main()