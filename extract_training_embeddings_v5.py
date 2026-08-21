#!/usr/bin/env python3
"""
extract_training_embeddings_v5.py
====================================
Sobre v4 (CLIP ViT-L/14, 4 frames promediados -- la mejor configuracion
confirmada hasta ahora, 56.8% de accuracy con 9 participantes), agrega
una feature nueva: APERTURA DE MANO (que tan abiertos estan los dedos
respecto a la muneca, normalizado por tamano de mano).

Motivo: el analisis con analyze_move_to.py descarto que dividir move_to
por verbo ayude (los clusters visuales no correlacionan con el verbo).
La hipotesis siguiente: grasp (agarrar, mano CERRANDOSE) y move_to/
put-down (soltar, mano ABRIENDOSE) se ven casi identicos en posicion y
bbox -- la senal que falta es el estado de los dedos, que nunca se le
dio al modelo (solo se usaban WRIST y MIDDLE_FINGER_MCP de los 21
keypoints disponibles).

Vector final: [embedding_CLIP_promediado (768) | net_disp | curvature |
              area_change | angular_range | aperture_mean |
              aperture_change] = 774 dims (2 mas que v4 -- NO mezclar
              con .npz de v2/v3/v4 en merge_embeddings.py, el numero de
              columnas no calza)

Uso: identico a v4.
    python3 extract_training_embeddings_v5.py \
        --videos-dir kitchen/EPIC-KITCHENS/P01/videos/ \
        --gt-csv annotations/EPIC_100_train.csv annotations/EPIC_100_validation.csv \
        --model-path yoloe-11l-seg-pf.pt \
        --output embeddings_P01_v5.npz \
        --window-size 8 --windows-per-segment 4 --segments-per-class-cap 3000
"""
import argparse
import math
import re
from collections import defaultdict
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


def hand_aperture(wrist: Tuple[float, float], mid_mcp: Tuple[float, float],
                  fingertips: List[Tuple[float, float]]) -> float:
    """
    Qué tan abierta está la mano: distancia promedio muñeca->punta de cada
    dedo, normalizada por el tamaño de la mano (muñeca->nudillo medio) para
    que no dependa de qué tan cerca esté la mano de la cámara.

    Es la señal que le faltaba al modelo para distinguir grasp (mano
    cerrándose sobre el objeto) de move_to/release (mano abriéndose al
    soltar) — con solo posición+bbox ambas se ven casi idénticas.
    """
    scale = math.hypot(mid_mcp[0] - wrist[0], mid_mcp[1] - wrist[1])
    if scale < 1e-6:
        return 0.0
    dists = [math.hypot(tip[0] - wrist[0], tip[1] - wrist[1]) for tip in fingertips]
    return float(np.mean(dists) / scale)


def motion_features(positions, areas, angles, apertures=None) -> np.ndarray:
    pts = np.array(positions, dtype=float)
    net_disp = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) >= 2 else 0.0
    path_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) >= 2 else 0.0
    curvature = 0.0 if net_disp == 0 else (path_length / net_disp) - 1.0
    area_change = 0.0
    if len(areas) >= 2 and areas[0] > 0:
        area_change = (areas[-1] - areas[0]) / areas[0]
    ang_range = angular_range(angles) if len(angles) >= 2 else 0.0

    aperture_mean = 0.0
    aperture_change = 0.0
    if apertures and len(apertures) >= 1:
        aperture_mean = float(np.mean(apertures))
        if len(apertures) >= 2:
            aperture_change = apertures[-1] - apertures[0]

    return np.array([net_disp, curvature, area_change, ang_range,
                     aperture_mean, aperture_change], dtype=np.float32)


def extract_window(cap, start_frame: int, window_size: int,
                    yolo: YOLO, hands: HandLandmarker,
                    width: int, height: int, confidence: float,
                    max_missing_frames: int = 3,
                    dist_threshold_frac: float = 0.20,
                    ) -> Optional[Tuple[List[np.ndarray], List, List, List]]:
    """
    Igual que v2, pero devuelve la lista COMPLETA de crops válidos de la
    ventana (no solo el del medio), para promediar sus embeddings después.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    diagonal = (width ** 2 + height ** 2) ** 0.5
    dist_threshold = diagonal * dist_threshold_frac

    positions, areas, angles, apertures, crops = [], [], [], [], []
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
        # Puntas de los 5 dedos (índices MediaPipe: pulgar=4, índice=8,
        # medio=12, anular=16, meñique=20)
        fingertip_idx = [4, 8, 12, 16, 20]
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

        fingertips_px = [(landmarks[i].x * width, landmarks[i].y * height)
                         for i in fingertip_idx]
        apertures.append(hand_aperture((wx, wy), (ox, oy), fingertips_px))

        crops.append(make_crop(frame, (int(wx), int(wy)), best_bbox))

    if len(positions) < max(3, window_size - max_missing_frames):
        return None

    return crops, positions, areas, angles, apertures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--gt-csv", nargs="+", required=True)
    parser.add_argument("--model-path", default="yoloe-11l-seg-pf.pt")
    parser.add_argument("--output", default="embeddings_v3.npz")
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--windows-per-segment", type=int, default=1)
    parser.add_argument("--segments-per-class-cap", type=int, default=3000)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--max-missing-frames", type=int, default=3)
    parser.add_argument("--dist-threshold-frac", type=float, default=0.20)
    parser.add_argument("--max-frames-to-embed", type=int, default=4,
                        help="Máximo de frames de la ventana a promediar. "
                             "4 confirmado empíricamente como mejor que 8 "
                             "(promediar de más diluye la señal en vez de "
                             "sumarla: 8 frames dio 47.7% vs 50.6% con 4, "
                             "sobre P01 solo).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | CLIP: {args.clip_model} | "
          f"promediando hasta {args.max_frames_to_embed} frames por ventana")

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
    all_epic_verbs: List[str] = []
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

        n_extracted, n_attempted = 0, 0
        for seg in gt.segments_for_video(video_id):
            label = EPIC_TO_ROBOT.get(seg.verb)
            if label is None or class_counts[label] >= args.segments_per_class_cap:
                continue
            epic_verb = seg.verb   # se guarda aparte, para poder re-particionar después

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
                                        width, height, args.confidence,
                                        max_missing_frames=args.max_missing_frames,
                                        dist_threshold_frac=args.dist_threshold_frac)
                if result is None:
                    continue
                crops, positions, areas, angles, apertures = result

                # Muestrea hasta max_frames_to_embed crops equiespaciados
                # de la ventana (no todos, para no disparar el costo de CLIP)
                if len(crops) > args.max_frames_to_embed:
                    idx = np.linspace(0, len(crops) - 1, args.max_frames_to_embed, dtype=int)
                    crops_to_embed = [crops[i] for i in sorted(set(idx))]
                else:
                    crops_to_embed = crops

                embeddings = []
                for crop in crops_to_embed:
                    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    inputs = clip_processor(images=pil_img, return_tensors="pt").to(device)
                    embeddings.append(get_clip_image_embedding(clip_model, inputs))

                clip_embedding = np.mean(embeddings, axis=0)   # <-- el cambio clave vs v2

                mfeats = motion_features(positions, areas, angles, apertures)
                combined = np.concatenate([clip_embedding, mfeats])

                all_features.append(combined)
                all_labels.append(label)
                all_epic_verbs.append(epic_verb)
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
    epic_verbs = np.array(all_epic_verbs)
    np.savez(args.output, X=X, y=y, epic_verb=epic_verbs, clip_dim=X.shape[1] - 6)
    print(f"\nGuardado {len(y)} muestras ({X.shape[1]}-dim, con verbo EPIC original) en {args.output}")
    print(f"Distribución final: {dict(class_counts)}")


if __name__ == "__main__":
    main()