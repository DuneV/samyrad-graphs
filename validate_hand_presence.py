#!/usr/bin/env python3
"""
validate_hand_presence.py
============================
cache_crops.py, cuando una ventana tiene menos frames con detección real
de mano que --frames-per-window, RELLENA duplicando el último crop
válido para completar el número de frames guardados. Esto es necesario
para tener siempre el mismo tamaño de entrada, pero tiene un costo: si
la mayoría de los frames guardados de una ventana son duplicados, la
"trayectoria" que ve el modelo es prácticamente estática aunque la
acción real no lo haya sido -- la mano se detectó de forma muy
intermitente, y las features de movimiento (net_disp, curvature...)
quedan degeneradas.

Este script:
  1. Detecta, por cada ventana cacheada, cuántos de sus frames guardados
     son duplicados (píxeles casi idénticos) vs genuinamente distintos.
  2. Reporta estadísticas (por participante y por clase) de qué tan
     seguido pasa esto -- útil para saber si hay clases/participantes
     con detección de mano poco confiable.
  3. Escribe un metadata_filtrado.csv sin las ventanas por debajo del
     umbral de frames únicos, para poder reentrenar solo con ventanas
     donde la mano SÍ se seguía de forma consistente.

Uso:
    python3 validate_hand_presence.py \
        --crops-dirs crops_cache/P01 crops_cache/P02 \
        --min-unique-frames 3 \
        --output-suffix _filtrado
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import List

import cv2
import numpy as np


def count_unique_frames(crop_paths: List[Path], diff_threshold: float = 2.0) -> int:
    """
    Cuenta cuántos frames de la ventana son genuinamente distintos entre
    sí (diferencia media de píxeles > diff_threshold respecto a todos
    los anteriores ya vistos). Frames de relleno (duplicados exactos o
    casi exactos del último crop válido) no cuentan como nuevos.
    """
    images = [cv2.imread(str(p)) for p in crop_paths]
    if any(im is None for im in images):
        return 0

    unique_images = [images[0]]
    for img in images[1:]:
        is_duplicate = False
        for seen in unique_images:
            if img.shape != seen.shape:
                continue
            diff = float(np.mean(cv2.absdiff(img, seen)))
            if diff < diff_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_images.append(img)

    return len(unique_images)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-dirs", nargs="+", required=True)
    parser.add_argument("--min-unique-frames", type=int, default=3,
                        help="Ventanas con menos frames únicos que esto "
                             "se excluyen del CSV filtrado (mano detectada "
                             "de forma demasiado intermitente)")
    parser.add_argument("--diff-threshold", type=float, default=2.0,
                        help="Diferencia media de píxeles (escala 0-255) "
                             "por debajo de la cual dos frames se "
                             "consideran duplicados")
    parser.add_argument("--output-suffix", default="_filtrado",
                        help="Se escribe metadata{suffix}.csv junto al "
                             "metadata.csv original de cada carpeta")
    args = parser.parse_args()

    stats_by_class = defaultdict(lambda: {"total": 0, "bajo_umbral": 0})
    stats_by_dir = {}

    for crops_dir in args.crops_dirs:
        crops_dir = Path(crops_dir)
        metadata_path = crops_dir / "metadata.csv"
        if not metadata_path.exists():
            print(f"⚠ {metadata_path} no existe, se salta")
            continue

        print(f"\nProcesando {crops_dir}...")
        with open(metadata_path) as f:
            rows = list(csv.DictReader(f))

        kept_rows = []
        n_below = 0
        for row in rows:
            frame_files = row["frame_files"].split("|")
            crop_paths = [crops_dir / fn for fn in frame_files]
            n_unique = count_unique_frames(crop_paths, args.diff_threshold)

            label = row["label"]
            stats_by_class[label]["total"] += 1

            if n_unique < args.min_unique_frames:
                n_below += 1
                stats_by_class[label]["bajo_umbral"] += 1
                continue

            kept_rows.append(row)

        stats_by_dir[str(crops_dir)] = {
            "total": len(rows), "excluidas": n_below, "conservadas": len(kept_rows),
        }

        if kept_rows:
            out_path = crops_dir / f"metadata{args.output_suffix}.csv"
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(kept_rows[0].keys()))
                writer.writeheader()
                writer.writerows(kept_rows)
            print(f"  {len(rows)} ventanas totales, {n_below} excluidas "
                  f"(<{args.min_unique_frames} frames únicos), "
                  f"{len(kept_rows)} conservadas -> {out_path}")
        else:
            print(f"  {len(rows)} ventanas totales, ¡TODAS excluidas!")

    print(f"\n{'='*60}")
    print("RESUMEN POR CARPETA")
    print(f"{'='*60}")
    for d, s in stats_by_dir.items():
        pct = s["excluidas"] / s["total"] * 100 if s["total"] else 0
        print(f"  {d:30s} {s['excluidas']:5d}/{s['total']:5d} excluidas ({pct:.1f}%)")

    print(f"\n{'='*60}")
    print("RESUMEN POR CLASE (dónde falla más la detección de mano)")
    print(f"{'='*60}")
    for label, s in sorted(stats_by_class.items(), key=lambda x: -x[1]["bajo_umbral"]):
        pct = s["bajo_umbral"] / s["total"] * 100 if s["total"] else 0
        print(f"  {label:12s} {s['bajo_umbral']:5d}/{s['total']:5d} excluidas ({pct:.1f}%)")

    print(f"\nUsa metadata{args.output_suffix}.csv en vez de metadata.csv al "
          f"combinar/entrenar, para excluir las ventanas con detección de "
          f"mano poco confiable.")


if __name__ == "__main__":
    main()