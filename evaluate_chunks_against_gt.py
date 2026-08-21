#!/usr/bin/env python3
"""
evaluate_chunks_against_gt.py
==============================
Compara las acciones que detectó CLIP (ya guardadas en tus chunks
procesados, chunks/<video_id>/chunk_N.json) contra el ground truth real
de EPIC-KITCHENS — SIN reprocesar ningún video.

Aprovecha que la carpeta chunks/<video_id>/ ya usa el video_id real como
nombre (ej. chunks/P01_01/), así no hay que adivinarlo.

Uso:
    python3 evaluate_chunks_against_gt.py \
        --chunks-root chunks/ \
        --gt-csv annotations/EPIC_100_train.csv annotations/EPIC_100_validation.csv
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from epic_ground_truth import EpicGroundTruth
from robotAction import EPIC_TO_ROBOT

EPIC_VIDEO_ID_RE = re.compile(r"^(P\d{2}_\d{2,3})")


def compare_video(chunk_files: List[Path], gt: EpicGroundTruth, video_id: str) -> Optional[Dict]:
    pairs = []
    for cf in chunk_files:
        data = json.loads(cf.read_text())
        for event in data["scenario"].get("semantic_line", []):
            seg = gt.segment_at_frame(video_id, event["frame_index"])
            if seg is None:
                continue
            gt_action = EPIC_TO_ROBOT.get(seg.verb)
            if gt_action is None:   # verbo EPIC sin equivalente robótico
                continue
            pairs.append((event["action"], gt_action))

    if not pairs:
        return None

    total = len(pairs)
    correct = sum(1 for pred, gt_a in pairs if pred == gt_a)

    confusion = defaultdict(lambda: defaultdict(int))
    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)
    for pred, gt_a in pairs:
        confusion[gt_a][pred] += 1
        per_class_total[gt_a] += 1
        if pred == gt_a:
            per_class_correct[gt_a] += 1

    return {
        "video_id": video_id,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 3),
        "per_class_accuracy": {
            c: round(per_class_correct[c] / per_class_total[c], 3)
            for c in per_class_total
        },
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-root", default="chunks",
                        help="Carpeta que contiene chunks/<video_id>/chunk_N.json")
    parser.add_argument("--gt-csv", nargs="+", required=True)
    args = parser.parse_args()

    gt = EpicGroundTruth(args.gt_csv)
    chunks_root = Path(args.chunks_root)

    all_results = []
    for video_dir in sorted(chunks_root.iterdir()):
        if not video_dir.is_dir():
            continue

        match = EPIC_VIDEO_ID_RE.match(video_dir.name)
        if not match:
            print(f"{video_dir.name}: no parece un video_id de EPIC-KITCHENS, se salta")
            continue
        video_id = match.group(1)

        if not gt.has_video(video_id):
            print(f"{video_id}: no está en el/los CSV de ground truth cargados, se salta")
            continue

        chunk_files = sorted(video_dir.glob("chunk_*.json"),
                              key=lambda p: int(p.stem.split("_")[1]) if p.stem.split("_")[1].isdigit() else 0)
        chunk_files = [f for f in chunk_files if f.name.startswith("chunk_") and f.suffix == ".json"]

        result = compare_video(chunk_files, gt, video_id)
        if result is None:
            print(f"{video_id}: sin eventos comparables contra ground truth")
            continue

        all_results.append(result)
        print(f"\n{video_id}: accuracy {result['accuracy']*100:.1f}% "
              f"({result['correct']}/{result['total']})")
        for cls, acc in sorted(result["per_class_accuracy"].items()):
            print(f"    {cls:12s}: {acc*100:.1f}%")

    if all_results:
        total = sum(r["total"] for r in all_results)
        correct = sum(r["correct"] for r in all_results)
        print(f"\n{'='*50}")
        print(f"GLOBAL: {correct}/{total} = {correct/total*100:.1f}% "
              f"(sobre {len(all_results)} videos)")
        print(f"{'='*50}")

        # Matriz de confusión global (sumando todas las de cada video)
        global_confusion = defaultdict(lambda: defaultdict(int))
        for r in all_results:
            for gt_cls, preds in r["confusion_matrix"].items():
                for pred_cls, count in preds.items():
                    global_confusion[gt_cls][pred_cls] += count
        print("\nMatriz de confusión global (fila=ground truth, columna=predicción CLIP):")
        for gt_cls, preds in sorted(global_confusion.items()):
            print(f"  {gt_cls:12s}: {dict(preds)}")
    else:
        print("\nNo se pudo evaluar ningún video (revisa --chunks-root y --gt-csv)")


if __name__ == "__main__":
    main()