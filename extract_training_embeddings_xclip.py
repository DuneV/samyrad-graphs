#!/usr/bin/env python3
"""
extract_training_embeddings_xclip.py
======================================
Reemplaza CLIP (imagen estática, promediada sobre frames) por X-CLIP
(microsoft/xclip-base-patch32) — un modelo entrenado ESPECÍFICAMENTE
para reconocer acciones en video (Kinetics-400), con un Multi-frame
Integration Transformer que modela la relación temporal REAL entre
frames, en vez de simplemente promediar embeddings independientes.

~193M parámetros, cabe cómodamente en 8GB de VRAM (RTX 3070 o similar).
Entrenado con clips de 8 frames — por eso --window-size debe quedarse en 8
para que coincida con lo que el modelo espera nativamente.

A diferencia de v2-v4, extract_window_fixed_length() SIEMPRE devuelve
exactamente `window_size` frames en orden temporal (rellenando huecos
con el último frame válido) — X-CLIP necesita una secuencia ordenada de
longitud fija, no puede saltarse frames como hacían las versiones
anteriores (que solo promediaban, sin importar el orden).

Vector final: [embedding_XCLIP (512) | net_disp | curvature |
              area_change | angular_range] = 516 dims
NOTA: la dimensión del embedding de X-CLIP (512) es distinta a la de
CLIP ViT-L/14 (768) — no mezcles este .npz con los de v2/v3/v4 en
merge_embeddings.py (ya detecta el mismatch de clip_dim y lo descarta).

Uso:
    python3 extract_training_embeddings_xclip.py \
        --videos-dir kitchen/EPIC-KITCHENS/P01/videos/ \
        --gt-csv annotations/EPIC_100_train.csv annotations/EPIC_100_validation.csv \
        --model-path yoloe-11l-seg-pf.pt \
        --output embeddings_P01_xclip.npz \
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
from transformers import XCLIPModel, XCLIPProcessor
import mediapipe as mp
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from ultralytics import YOLO

from epic_ground_truth import EpicGroundTruth
from robotAction import EPIC_TO_ROBOT

EPIC_VIDEO_ID_RE = re.compile(r"^(P\d{2}_\d{2,3})")


def get_xclip_video_embedding(xclip_model: XCLIPModel, inputs: dict) -> np.ndarray:
    """
    xclip_model.get_video_features() debería devolver un tensor plano,
    pero en algunas versiones de transformers devuelve un objeto envuelto
    (BaseModelOutputWithPooling) en vez del tensor directamente — mismo
    comportamiento que ya vimos con CLIPModel.get_image_features().
    """
    with torch.no_grad():
        output = xclip_model.get_video_features(**inputs)
    if isinstance(output, torch.Tensor):
        tensor = output
    elif hasattr(output, "video_embeds"):
        tensor = output.video_embeds
    elif hasattr(output, "image_embeds"):
        tensor = output.image_embeds
    elif hasattr(output, "pooler_output"):
        tensor = output.pooler_output
    elif hasattr(output, "last_hidden_state"):
        # último recurso: promedia sobre la dimensión de secuencia
        tensor = output.last_hidden_state.mean(dim=1)
    else:
        raise TypeError(f"No se pudo extraer el embedding de {type(output)}: "
                        f"atributos disponibles: {dir(output)}")
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


def motion_features(positions, areas, angles) -> np.ndarray:
    pts = np.array(positions, dtype=float)
    net_disp = float(np.linalg.norm(pts[-1] - pts[0])) if len(pts) >= 2 else 0.0
    path_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) >= 2 else 0.0
    curvature = 0.0 if net_disp == 0 else (path_length / net_disp) - 1.0
    area_change = 0.0
    if len(areas) >= 2 and areas[0] > 0:
        area_change = (areas[-1] - areas[0]) / areas[0]
    ang_range = angular_range(angles) if len(angles) >= 2 else 0.0
    return np.array([net_disp, curvature, area_change, ang_range], dtype=np.float32)


def extract_window_fixed_length(cap, start_frame: int, window_size: int,
                                yolo: YOLO, hands: HandLandmarker,
                                width: int, height: int, confidence: float,
                                dist_threshold_frac: float = 0.20,
                                ) -> Optional[Tuple[List[np.ndarray], List, List, List]]:
    """
    SIEMPRE devuelve exactamente `window_size` crops en orden temporal
    (a diferencia de v2-v4, que se saltaban frames fallidos). Los huecos
    se rellenan repitiendo el último crop válido — necesario porque
    X-CLIP espera una secuencia ordenada de longitud fija, no un
    conjunto de frames sueltos para promediar.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    diagonal = (width ** 2 + height ** 2) ** 0.5
    dist_threshold = diagonal * dist_threshold_frac

    positions, areas, angles = [], [], []
    crops: List[Optional[np.ndarray]] = []
    last_valid_crop = None

    for i in range(window_size):
        ret, frame = cap.read()
        if not ret:
            return None

        crop = None
        yolo_result = yolo(frame, conf=confidence, verbose=False)[0]
        if yolo_result.boxes is not None and len(yolo_result.boxes) > 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mp_result = hands.detect(mp_image)
            if mp_result.hand_landmarks:
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

                if best_bbox is not None and best_dist <= dist_threshold:
                    positions.append((wx, wy))
                    xmin, ymin, xmax, ymax = best_bbox
                    areas.append(max(0, xmax - xmin) * max(0, ymax - ymin))
                    angles.append(math.atan2(oy - wy, ox - wx))
                    crop = make_crop(frame, (int(wx), int(wy)), best_bbox)

        if crop is not None:
            last_valid_crop = crop
            crops.append(crop)
        elif last_valid_crop is not None:
            crops.append(last_valid_crop)   # repite el último válido (mantiene el orden)
        else:
            crops.append(None)   # se rellena más abajo con el primero que aparezca

    first_valid = next((c for c in crops if c is not None), None)
    if first_valid is None:
        return None   # ninguna detección válida en toda la ventana
    crops = [c if c is not None else first_valid for c in crops]

    if len(positions) < 3:
        return None   # muy poca señal real de movimiento

    return crops, positions, areas, angles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--gt-csv", nargs="+", required=True)
    parser.add_argument("--model-path", default="yoloe-11l-seg-pf.pt")
    parser.add_argument("--output", default="embeddings_xclip.npz")
    parser.add_argument("--window-size", type=int, default=8,
                        help="Debe quedarse en 8: xclip-base-patch32 fue "
                             "entrenado con clips de exactamente 8 frames")
    parser.add_argument("--windows-per-segment", type=int, default=4)
    parser.add_argument("--segments-per-class-cap", type=int, default=3000)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--xclip-model", default="microsoft/xclip-base-patch32")
    parser.add_argument("--dist-threshold-frac", type=float, default=0.20)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | X-CLIP: {args.xclip_model}")

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

    xclip_model = XCLIPModel.from_pretrained(args.xclip_model).to(device)
    xclip_model.eval()
    xclip_processor = XCLIPProcessor.from_pretrained(args.xclip_model)

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
            epic_verb = seg.verb

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
                result = extract_window_fixed_length(
                    cap, start, args.window_size, yolo, hands,
                    width, height, args.confidence,
                    dist_threshold_frac=args.dist_threshold_frac)
                if result is None:
                    continue
                crops, positions, areas, angles = result

                pil_frames = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                             for c in crops]
                inputs = xclip_processor(videos=[pil_frames], return_tensors="pt").to(device)
                video_embedding = get_xclip_video_embedding(xclip_model, inputs)

                mfeats = motion_features(positions, areas, angles)
                combined = np.concatenate([video_embedding, mfeats])

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
    np.savez(args.output, X=X, y=y, epic_verb=epic_verbs, clip_dim=X.shape[1] - 4)
    print(f"\nGuardado {len(y)} muestras ({X.shape[1]}-dim, X-CLIP) en {args.output}")
    print(f"Distribución final: {dict(class_counts)}")


if __name__ == "__main__":
    main()