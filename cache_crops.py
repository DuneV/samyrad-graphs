#!/usr/bin/env python3
"""
cache_crops.py
================
Extrae las mismas ventanas que extract_training_embeddings_v4.py, pero
en vez de calcular embeddings de CLIP y descartar las imágenes, GUARDA
los recortes en disco como JPEGs chicos + un metadata.csv. Necesario
para fine-tuning real: hay que poder recorrer el backbone en cada época
de entrenamiento, no una sola vez como con embeddings congelados.

Estructura de salida:
    crops_cache/<run_name>/
        window_000001_f0.jpg
        window_000001_f1.jpg
        window_000001_f2.jpg
        window_000001_f3.jpg
        window_000002_f0.jpg
        ...
        metadata.csv   (window_id, label, epic_verb, net_disp, curvature,
                        area_change, angular_range, video_id, n_frames)

Uso:
    python3 cache_crops.py \
        --videos-dir kitchen/EPIC-KITCHENS/P01/videos/ \
        --gt-csv annotations/EPIC_100_train.csv annotations/EPIC_100_validation.csv \
        --model-path yoloe-11l-seg-pf.pt \
        --output-dir crops_cache/P01 \
        --window-size 8 --windows-per-segment 4 --segments-per-class-cap 3000 \
        --frames-per-window 4
"""

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from ultralytics import YOLO

from epic_ground_truth import EpicGroundTruth
from robotAction import EPIC_TO_ROBOT

EPIC_VIDEO_ID_RE = re.compile(r"^(P\d{2}_\d{2,3})")


def make_crop(frame_bgr, hand_wrist, obj_bbox, padding=80):
    h, w = frame_bgr.shape[:2]
    wx, wy = hand_wrist
    xmin, ymin, xmax, ymax = obj_bbox
    x0 = max(0, min(wx, xmin) - padding)
    y0 = max(0, min(wy, ymin) - padding)
    x1 = min(w, max(wx, xmax) + padding)
    y1 = min(h, max(wy, ymax) + padding)
    return frame_bgr[y0:y1, x0:x1]


def angular_range(angles: List[float]) -> float:
    if len(angles) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(angles[:-1], angles[1:]):
        d = b - a
        d = (d + math.pi) % (2 * math.pi) - math.pi
        total += abs(d)
    return math.degrees(total)


def motion_features(positions, areas, angles) -> Tuple[float, float, float, float]:
    pts = np.array(positions, dtype=float)
    net_disp = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) >= 2 else 0.0
    path_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) >= 2 else 0.0
    curvature = 0.0 if net_disp == 0 else (path_length / net_disp) - 1.0
    area_change = 0.0
    if len(areas) >= 2 and areas[0] > 0:
        area_change = (areas[-1] - areas[0]) / areas[0]
    ang_range = angular_range(angles) if len(angles) >= 2 else 0.0
    return net_disp, curvature, area_change, ang_range


def extract_window(cap, start_frame, window_size, yolo, hands, width, height,
                   confidence, max_missing_frames=3, dist_threshold_frac=0.20):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    diagonal = (width ** 2 + height ** 2) ** 0.5
    dist_threshold = diagonal * dist_threshold_frac

    positions, areas, angles, crops = [], [], [], []
    missing = 0

    for i in range(window_size):
        ret, frame = cap.read()
        if not ret:
            return None

        yolo_result = yolo(frame, conf=confidence, verbose=False)[0]
        if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            missing += 1
            if missing > max_missing_frames:
                return None
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_result = hands.detect(mp_image)
        if not mp_result.hand_landmarks:
            missing += 1
            if missing > max_missing_frames:
                return None
            continue

        boxes = yolo_result.boxes.xyxy.cpu().numpy()
        landmarks = mp_result.hand_landmarks[0]
        wrist = landmarks[0]
        mid_mcp = landmarks[9]
        wx, wy = wrist.x * width, wrist.y * height
        ox, oy = mid_mcp.x * width, mid_mcp.y * height

        best_bbox, best_dist = None, float("inf")
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box)
            cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
            dist = ((wx - cx) ** 2 + (wy - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best_bbox = dist, (xmin, ymin, xmax, ymax)

        if best_bbox is None or best_dist > dist_threshold:
            missing += 1
            if missing > max_missing_frames:
                return None
            continue

        positions.append((wx, wy))
        xmin, ymin, xmax, ymax = best_bbox
        areas.append(max(0, xmax - xmin) * max(0, ymax - ymin))
        angles.append(math.atan2(oy - wy, ox - wx))
        crops.append(make_crop(frame, (int(wx), int(wy)), best_bbox))

    if len(positions) < max(3, window_size - max_missing_frames):
        return None

    return crops, positions, areas, angles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--gt-csv", nargs="+", required=True)
    parser.add_argument("--model-path", default="yoloe-11l-seg-pf.pt")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--windows-per-segment", type=int, default=4)
    parser.add_argument("--segments-per-class-cap", type=int, default=3000)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--frames-per-window", type=int, default=4,
                        help="Cuántos frames de cada ventana de 8 guardar "
                             "en disco (equiespaciados). Menos = menos disco "
                             "y entrenamiento más rápido, pero menos contexto "
                             "temporal por muestra.")
    parser.add_argument("--crop-size", type=int, default=224,
                        help="Los recortes se guardan redimensionados a "
                             "crop_size x crop_size (224 = tamaño nativo de "
                             "CLIP ViT-B/32, evita redimensionar en cada "
                             "época durante el entrenamiento)")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gt = EpicGroundTruth(args.gt_csv)
    yolo = YOLO(args.model_path)

    hand_options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=args.confidence,
        min_tracking_confidence=args.confidence,
    )
    hands = HandLandmarker.create_from_options(hand_options)

    video_files = sorted(Path(args.videos_dir).glob("*.MP4")) + \
                  sorted(Path(args.videos_dir).glob("*.mp4"))

    metadata_rows = []
    window_id = 0
    class_counts: Dict[str, int] = defaultdict(int)

    for vf in video_files:
        match = EPIC_VIDEO_ID_RE.match(vf.stem)
        if not match:
            continue
        video_id = match.group(1)
        if not gt.has_video(video_id):
            print(f"{video_id}: sin ground truth, se salta")
            continue

        print(f"Procesando {video_id}...")
        cap = cv2.VideoCapture(str(vf))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        n_extracted, n_attempted = 0, 0
        for seg in gt.segments_for_video(video_id):
            label = EPIC_TO_ROBOT.get(seg.verb)
            if label is None or class_counts[label] >= args.segments_per_class_cap:
                continue

            seg_len = seg.stop_frame - seg.start_frame
            if seg_len < args.window_size:
                continue

            max_start = max(seg.start_frame, seg.stop_frame - args.window_size)
            if args.windows_per_segment == 1:
                starts = [(seg.start_frame + seg.stop_frame)//2 - args.window_size//2]
            else:
                starts = np.linspace(seg.start_frame, max_start,
                                     num=args.windows_per_segment, dtype=int)
                starts = sorted(set(int(s) for s in starts))

            for start in starts:
                if class_counts[label] >= args.segments_per_class_cap:
                    break
                start = max(seg.start_frame, min(start, max_start))

                n_attempted += 1
                result = extract_window(cap, start, args.window_size, yolo, hands,
                                        width, height, args.confidence)
                if result is None:
                    continue
                crops, positions, areas, angles = result

                if len(crops) > args.frames_per_window:
                    idx = np.linspace(0, len(crops) - 1, args.frames_per_window, dtype=int)
                    crops_to_save = [crops[i] for i in sorted(set(idx))]
                else:
                    crops_to_save = crops
                # Si dedup dejó menos de frames_per_window (ventanas cortas),
                # repite el último para mantener siempre el mismo conteo
                while len(crops_to_save) < args.frames_per_window:
                    crops_to_save.append(crops_to_save[-1])

                frame_paths = []
                for fi, crop in enumerate(crops_to_save):
                    resized = cv2.resize(crop, (args.crop_size, args.crop_size))
                    fname = f"window_{window_id:07d}_f{fi}.jpg"
                    fpath = out_dir / fname
                    cv2.imwrite(str(fpath), resized,
                               [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                    frame_paths.append(fname)

                net_disp, curvature, area_change, ang_range = motion_features(
                    positions, areas, angles)

                metadata_rows.append({
                    "window_id": window_id,
                    "video_id": video_id,
                    "label": label,
                    "epic_verb": seg.verb,
                    "frame_files": "|".join(frame_paths),
                    "net_disp": net_disp,
                    "curvature": curvature,
                    "area_change": area_change,
                    "angular_range": ang_range,
                })

                window_id += 1
                class_counts[label] += 1
                n_extracted += 1

        cap.release()
        pct = (n_extracted / n_attempted * 100) if n_attempted else 0
        print(f"  {n_extracted}/{n_attempted} ventanas válidas ({pct:.0f}%). "
              f"Totales: {dict(class_counts)}")

    csv_path = out_dir / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metadata_rows[0].keys()) if metadata_rows else [])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"\nGuardadas {len(metadata_rows)} ventanas "
          f"({len(metadata_rows) * args.frames_per_window} imágenes JPEG) en {out_dir}")
    print(f"Metadata: {csv_path}")
    print(f"Distribución final: {dict(class_counts)}")


if __name__ == "__main__":
    main()