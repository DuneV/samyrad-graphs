#!/usr/bin/env python3
"""
extract_training_embeddings_v2.py
===================================
Versión reforzada: para cada segmento de ground truth, en vez de tomar 1-2
frames sueltos, toma una VENTANA de 8 frames consecutivos (igual que
HandTrajectory en vivo) para calcular features de movimiento explícitas
(desplazamiento, curvatura, cambio de área del objeto, cambio de ángulo
de la mano), y las concatena al embedding visual de CLIP ViT-L/14 (más
pesado que ViT-B/32, pero con features mucho más ricas).

Vector final por muestra: [embedding_CLIP (768-dim) | net_disp | curvature |
                           area_change | angular_range]  = 772 dims

Uso:
    python3 extract_training_embeddings_v2.py \
        --videos-dir kitchen/EPIC-KITCHENS/P01/videos/ \
        --gt-csv annotations/EPIC_100_train.csv annotations/EPIC_100_validation.csv \
        --model-path yoloe-11l-seg-pf.pt \
        --output embeddings_P01_v2.npz \
        --window-size 8 \
        --segments-per-class-cap 3000 \
        --clip-model openai/clip-vit-large-patch14
"""

import argparse
import math
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from ultralytics import YOLO

from epic_ground_truth import EpicGroundTruth
from robotAction import EPIC_TO_ROBOT

EPIC_VIDEO_ID_RE = re.compile(r"^(P\d{2}_\d{2,3})")


def get_clip_image_embedding(clip_model: CLIPModel, inputs: dict) -> np.ndarray:
    """
    clip_model.get_image_features() debería devolver un tensor plano, pero
    en algunas versiones de transformers devuelve un objeto envuelto
    (BaseModelOutputWithPooling) en vez del tensor directamente. Maneja
    ambos casos para no depender de la versión exacta instalada.
    """
    with torch.no_grad():
        output = clip_model.get_image_features(**inputs)
    if isinstance(output, torch.Tensor):
        tensor = output
    elif hasattr(output, "image_embeds"):
        tensor = output.image_embeds
    elif hasattr(output, "pooler_output"):
        tensor = output.pooler_output
    else:
        raise TypeError(f"No se pudo extraer el embedding de {type(output)}")
    return tensor.squeeze().cpu().numpy()


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


def motion_features(positions: List[Tuple[float, float]],
                    areas: List[float], angles: List[float]) -> np.ndarray:
    """Las mismas 4 features que usa HandTrajectory en vivo, calculadas
    sobre una ventana completa de frames (no en tiempo real)."""
    pts = np.array(positions, dtype=float)
    net_disp = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) >= 2 else 0.0
    path_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) >= 2 else 0.0
    curvature = 0.0 if net_disp == 0 else (path_length / net_disp) - 1.0

    area_change = 0.0
    if len(areas) >= 2 and areas[0] > 0:
        area_change = (areas[-1] - areas[0]) / areas[0]

    ang_range = angular_range(angles) if len(angles) >= 2 else 0.0

    return np.array([net_disp, curvature, area_change, ang_range], dtype=np.float32)


def extract_window(cap, start_frame: int, window_size: int,
                    yolo: YOLO, hands: HandLandmarker,
                    width: int, height: int, confidence: float,
                    max_missing_frames: int = 3,
                    dist_threshold_frac: float = 0.20,
                    ) -> Optional[Tuple[np.ndarray, List, List, List]]:
    """
    Procesa `window_size` frames consecutivos desde start_frame.
    TOLERANTE a frames individuales fallidos (sin detección de YOLO, sin
    mano, o mano muy lejos de cualquier objeto): esos frames se saltan
    en vez de descartar toda la ventana, siempre que no se pierdan más
    de `max_missing_frames`. Antes exigía los 8/8 frames válidos, lo que
    descartaba casi todas las ventanas.

    dist_threshold_frac: distancia máxima mano-objeto como fracción de la
    diagonal del frame (en vez de un valor fijo en píxeles, que no
    escala bien entre videos de distinta resolución).

    Devuelve (crop_frame_más_cercano_al_medio, positions, areas, angles)
    o None si se perdieron demasiados frames.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    diagonal = (width ** 2 + height ** 2) ** 0.5
    dist_threshold = diagonal * dist_threshold_frac

    positions, areas, angles = [], [], []
    crops_by_frame: Dict[int, np.ndarray] = {}
    missing = 0
    mid_idx = window_size // 2

    for i in range(window_size):
        ret, frame = cap.read()
        if not ret:
            return None   # esto sí es fatal: se acabó el video

        yolo_result = yolo(frame, conf=confidence, verbose=False)[0]
        if yolo_result.boxes is None or len(yolo_result.boxes) == 0:
            missing += 1
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_result = hands.detect(mp_image)
        if not mp_result.hand_landmarks:
            missing += 1
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
            continue

        positions.append((wx, wy))
        xmin, ymin, xmax, ymax = best_bbox
        areas.append(max(0, xmax - xmin) * max(0, ymax - ymin))
        angles.append(math.atan2(oy - wy, ox - wx))
        crops_by_frame[i] = make_crop(frame, (int(wx), int(wy)), best_bbox)

        if missing > max_missing_frames:
            return None

    # Necesitamos al menos la mitad de la ventana con datos válidos para
    # que las features de movimiento (desplazamiento, curvatura) tengan sentido
    if len(positions) < max(3, window_size - max_missing_frames):
        return None

    # Crop más cercano al frame del medio de los que sí se lograron capturar
    closest_i = min(crops_by_frame.keys(), key=lambda i: abs(i - mid_idx))
    mid_crop = crops_by_frame[closest_i]
    if mid_crop is None or mid_crop.size == 0:
        return None

    return mid_crop, positions, areas, angles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--gt-csv", nargs="+", required=True)
    parser.add_argument("--model-path", default="yoloe-11l-seg-pf.pt")
    parser.add_argument("--output", default="embeddings_v2.npz")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--windows-per-segment", type=int, default=1)
    parser.add_argument("--segments-per-class-cap", type=int, default=3000)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14",
                        help="ViT-L/14 (768-dim, más lento/preciso) o "
                             "ViT-B/32 (512-dim, más rápido)")
    parser.add_argument("--max-missing-frames", type=int, default=3,
                        help="Frames individuales que se toleran perder dentro "
                             "de la ventana antes de descartarla (antes era 0: "
                             "un solo frame fallido descartaba toda la ventana)")
    parser.add_argument("--dist-threshold-frac", type=float, default=0.20,
                        help="Distancia máxima mano-objeto como fracción de la "
                             "diagonal del frame (antes era un valor fijo en "
                             "píxeles que no escalaba con la resolución)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | CLIP: {args.clip_model}")

    gt = EpicGroundTruth(args.gt_csv)
    yolo = YOLO(args.model_path).to(device)

    hand_options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=args.confidence,
        min_tracking_confidence=args.confidence,
    )
    hands = HandLandmarker.create_from_options(hand_options)

    clip_model = CLIPModel.from_pretrained(args.clip_model).to(device)
    clip_model.eval()
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

    video_files = sorted(Path(args.videos_dir).glob("*.MP4")) + \
                  sorted(Path(args.videos_dir).glob("*.mp4"))

    all_features: List[np.ndarray] = []
    all_labels: List[str] = []
    class_counts: Dict[str, int] = defaultdict(int)

    for vf in video_files:
        match = EPIC_VIDEO_ID_RE.match(vf.stem)
        if not match:
            continue
        video_id = match.group(1)
        if not gt.has_video(video_id):
            print(f"{video_id}: sin ground truth, se salta")
            continue

        print(f"Extrayendo {video_id}...")
        cap = cv2.VideoCapture(str(vf))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        n_extracted = 0
        n_attempted = 0
        for seg in gt.segments_for_video(video_id):
            label = EPIC_TO_ROBOT.get(seg.verb)
            if label is None or class_counts[label] >= args.segments_per_class_cap:
                continue

            seg_len = seg.stop_frame - seg.start_frame
            if seg_len < args.window_size:
                continue

            n_attempted += 1
            mid = (seg.start_frame + seg.stop_frame) // 2
            start = max(seg.start_frame, mid - args.window_size // 2)

            result = extract_window(cap, start, args.window_size, yolo, hands,
                                    width, height, args.confidence,
                                    max_missing_frames=args.max_missing_frames,
                                    dist_threshold_frac=args.dist_threshold_frac)
            if result is None:
                continue
            mid_crop, positions, areas, angles = result

            pil_img = Image.fromarray(cv2.cvtColor(mid_crop, cv2.COLOR_BGR2RGB))
            inputs = clip_processor(images=pil_img, return_tensors="pt").to(device)
            clip_embedding = get_clip_image_embedding(clip_model, inputs)

            mfeats = motion_features(positions, areas, angles)
            combined = np.concatenate([clip_embedding, mfeats])

            all_features.append(combined)
            all_labels.append(label)
            class_counts[label] += 1
            n_extracted += 1

        cap.release()
        pct = (n_extracted / n_attempted * 100) if n_attempted else 0
        print(f"  {n_extracted}/{n_attempted} ventanas válidas ({pct:.0f}%). "
              f"Totales: {dict(class_counts)}")

    if not all_features:
        print("No se extrajo ninguna muestra.")
        return

    X = np.stack(all_features)
    y = np.array(all_labels)
    np.savez(args.output, X=X, y=y, clip_dim=X.shape[1] - 4)
    print(f"\nGuardado {len(y)} muestras ({X.shape[1]}-dim: "
          f"{X.shape[1]-4} de CLIP + 4 de movimiento) en {args.output}")
    print(f"Distribución final: {dict(class_counts)}")


if __name__ == "__main__":
    main()