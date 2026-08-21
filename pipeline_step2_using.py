"""
pipeline_step2_using.py
==========================
STEP 2: USING -- clasifica las acciones de un video con DOS fuentes:
  (a) el modelo entrenado (TrainedActionClassifier, STEP 1)
  (b) el ground truth real de EPIC-KITCHENS

Ambas ramas usan EXACTAMENTE las mismas ventanas (mismos frames, mismo
objeto, misma posición) -- lo único que cambia es de dónde sale la
acción. Esto es lo que faltaba: benchmark_comparison.py todavía usaba
CLIP zero-shot en vivo para la rama "clasificador", nunca el modelo
realmente entrenado en STEP 1.
"""

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from epic_ground_truth import EpicGroundTruth
from robotAction import EPIC_TO_ROBOT
from trained_classifier_inference import TrainedActionClassifier


def make_crop(frame_bgr, hand_wrist, obj_bbox, padding=80, max_size=400,
              giant_object_frac=0.15):
    """
    Recorte de primer plano mano-objeto.

    v1 (original): unión mano+objeto+padding, sin límite -- con objetos
    gigantes (neveras, lavadoras) el crop terminaba siendo casi la
    imagen completa.

    v2 (primer intento): cap fijo de max_size centrado en la muñeca
    para TODOS los objetos -- sobrecorrigió: con objetos medianos (una
    olla, agarrada del soporte) recortaba de más y perdía el contexto
    visual completo del objeto, degradando la clasificación.

    v3 (esta versión): el cap SOLO se aplica cuando el objeto es
    realmente desproporcionado (bbox > giant_object_frac del área del
    frame, default 15% -- neveras/lavadoras/muebles grandes). Para
    objetos normales o medianos (ollas, tazas, tablas de cortar) se
    mantiene el comportamiento original (unión mano+objeto+padding,
    sin capar), preservando el contexto visual completo.
    """
    h, w = frame_bgr.shape[:2]
    frame_area = h * w
    wx, wy = hand_wrist
    xmin, ymin, xmax, ymax = obj_bbox
    obj_area = max(0, xmax - xmin) * max(0, ymax - ymin)
    is_giant = obj_area > giant_object_frac * frame_area

    if is_giant:
        half = max_size / 2
        x0 = max(0, int(wx - half))
        y0 = max(0, int(wy - half))
        x1 = min(w, int(wx + half))
        y1 = min(h, int(wy + half))
    else:
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


def extract_window(cap, start_frame, window_size, yolo, hands, width, height,
                   confidence, max_missing_frames=3, dist_threshold_frac=0.20):
    """Idéntico al extract_window ya probado en cache_crops.py."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    diagonal = (width ** 2 + height ** 2) ** 0.5
    dist_threshold = diagonal * dist_threshold_frac

    positions, areas, angles, crops = [], [], [], []
    missing = 0
    last_matched_center = None   # para el seguimiento pegajoso dentro de la ventana

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

        import mediapipe as mp
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

        # Emparejamiento mano-objeto en tres fases:
        # 1) Candidatos: objetos donde la muñeca está LITERALMENTE DENTRO
        #    del bbox (no solo "cerca del centro") -- en escenas con
        #    muchos objetos próximos (ej. un cajón con cubiertos, varias
        #    zanahorias en fila), "el centro más cercano" puede enganchar
        #    el objeto equivocado aunque la mano esté físicamente tocando
        #    otro.
        # 2) SEGUIMIENTO PEGAJOSO: si ya veníamos siguiendo un objeto en
        #    frames anteriores de esta misma ventana, se prefiere seguir
        #    con el candidato más cercano a ese centro anterior -- evita
        #    que la ventana "salte" entre zanahorias/cubiertos distintos
        #    frame a frame por el jitter normal de la detección, lo cual
        #    corrompía tanto el crop como las features de movimiento.
        # 3) Sin objeto previo (primer frame) o ninguno cerca de él: se
        #    usa el candidato de área más chica (el más específico).
        # 4) Respaldo si nada contiene la muñeca: el centro más cercano
        #    (comportamiento original).
        containing_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = map(int, box)
            if xmin <= wx <= xmax and ymin <= wy <= ymax:
                area = (xmax - xmin) * (ymax - ymin)
                cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                containing_boxes.append((area, (xmin, ymin, xmax, ymax), (cx, cy)))

        sticky_radius = diagonal * 0.03   # ~3% de la diagonal del frame

        if containing_boxes:
            chosen = None
            if last_matched_center is not None:
                for area, bbox, center in sorted(containing_boxes, key=lambda c: c[0]):
                    dist_to_last = ((center[0] - last_matched_center[0]) ** 2 +
                                    (center[1] - last_matched_center[1]) ** 2) ** 0.5
                    if dist_to_last < sticky_radius:
                        chosen = (bbox, center)
                        break
            if chosen is None:
                containing_boxes.sort(key=lambda x: x[0])
                chosen = (containing_boxes[0][1], containing_boxes[0][2])
            best_bbox, chosen_center = chosen[0], chosen[1]
            # Distancia REAL al centro del objeto elegido (aunque la
            # muñeca esté contenida en su bbox) -- así el filtro de
            # calidad (dist_threshold) sigue siendo significativo. Antes
            # esto se forzaba a 0.0 siempre que hubiera contención, lo
            # que dejaba pasar objetos de fondo irrelevantes (ej. un
            # mesón gigante que "contiene" la muñeca pero no es lo que
            # realmente se está manipulando) sin ningún filtro real.
            best_dist = ((wx - chosen_center[0]) ** 2 +
                        (wy - chosen_center[1]) ** 2) ** 0.5
            last_matched_center = chosen[1]
        else:
            best_bbox, best_dist = None, float("inf")
            for box in boxes:
                xmin, ymin, xmax, ymax = map(int, box)
                cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
                dist = ((wx - cx) ** 2 + (wy - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_bbox = dist, (xmin, ymin, xmax, ymax)
            if best_bbox is not None and best_dist <= dist_threshold:
                last_matched_center = ((best_bbox[0] + best_bbox[2]) / 2,
                                       (best_bbox[1] + best_bbox[3]) / 2)

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


def build_scenarios_dual(video_path: str, video_id: str, gt: EpicGroundTruth,
                         classifier: TrainedActionClassifier,
                         yolo, hands, confidence: float = 0.5,
                         window_size: int = 8, windows_per_segment: int = 2,
                         on_window=None
                         ) -> Tuple[Dict, Dict]:
    """
    Recorre los segmentos anotados de video_id y construye DOS escenarios
    en paralelo, sobre EXACTAMENTE las mismas ventanas:
      - scenario_classifier: acción = predicción de TrainedActionClassifier
      - scenario_ground_truth: acción = verbo real de EPIC-KITCHENS

    on_window: callback opcional, se llama para CADA ventana clasificada
        con (crops, pred_action, pred_conf, gt_action, obj, video_id,
        window_idx) -- el caller decide qué hacer con eso (ej. guardar
        una muestra de los fallos a disco). No se llama si es None (sin
        costo extra si no lo necesitas).

    Devuelve (scenario_classifier, scenario_ground_truth).
    """
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    object_positions: Dict[str, Tuple] = {}
    semantic_line_clf: List[Dict] = []
    semantic_line_gt: List[Dict] = []
    detected_objects, target_objects_clf, target_objects_gt = set(), set(), set()
    required_actions_clf, required_actions_gt = set(), set()
    window_idx = 0

    for seg in gt.segments_for_video(video_id):
        gt_action = EPIC_TO_ROBOT.get(seg.verb)
        if gt_action is None:
            continue

        seg_len = seg.stop_frame - seg.start_frame
        if seg_len < window_size:
            continue
        max_start = max(seg.start_frame, seg.stop_frame - window_size)
        starts = np.linspace(seg.start_frame, max_start, num=windows_per_segment, dtype=int)
        starts = sorted(set(int(s) for s in starts))

        for start in starts:
            result = extract_window(cap, start, window_size, yolo, hands,
                                    width, height, confidence)
            if result is None:
                continue
            crops, positions, areas, angles = result

            wx, wy = positions[-1]
            obj = seg.noun
            object_positions.setdefault(obj, (wx, wy, 0.0))
            object_positions.setdefault("robot_right_hand", (0.0, 0.0, 0.0))
            detected_objects.add(obj)

            mfeats = motion_features(positions, areas, angles)
            pred_action, pred_conf = classifier.classify_window(crops, mfeats)

            if on_window is not None:
                on_window(crops, pred_action, pred_conf, gt_action, obj,
                         video_id, window_idx)
            window_idx += 1

            semantic_line_clf.append({
                "action": pred_action, "object": obj, "confidence": pred_conf})
            required_actions_clf.add(pred_action)
            target_objects_clf.add(obj)

            semantic_line_gt.append({"action": gt_action, "object": obj})
            required_actions_gt.add(gt_action)
            target_objects_gt.add(obj)

    cap.release()

    def edges_from(semantic_line):
        return {("robot_right_hand", e["action"], e["object"]): 0.3
                for e in semantic_line}

    scenario_classifier = {
        "goal": "Perform robot task", "detected_objects": list(detected_objects),
        "target_objects": list(target_objects_clf),
        "required_actions": list(required_actions_clf),
        "object_positions": object_positions,
        "edge_labels": edges_from(semantic_line_clf),
        "semantic_line": semantic_line_clf,
    }
    scenario_ground_truth = {
        "goal": "Perform robot task", "detected_objects": list(detected_objects),
        "target_objects": list(target_objects_gt),
        "required_actions": list(required_actions_gt),
        "object_positions": object_positions,
        "edge_labels": edges_from(semantic_line_gt),
        "semantic_line": semantic_line_gt,
    }
    return scenario_classifier, scenario_ground_truth