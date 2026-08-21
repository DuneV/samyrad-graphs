from tqdm import tqdm
from ultralytics import YOLO
import cv2
import json
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from collections import deque
import torch
import numpy as np
from PIL import Image
from robotAction import *
from epic_ground_truth import *
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
from transformers import CLIPModel, CLIPProcessor

HAND_KEYPOINT_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]

CANDIDATE_ACTIONS = [
    "a hand grasping an object",
    "a hand moving an object to another location",
    "a hand pouring liquid from a container",
    "a hand cutting with a knife",
    "a hand opening a door or cabinet",
    "a hand closing a door or cabinet",
    "a hand pushing an object",
    "a hand pulling an object",
    "a hand pressing a button",
    "a hand rotating a knob",
    "a hand inspecting an object",
    "a hand removing an object from a surface",
]

CLIP_TO_ACTION: Dict[str, str] = {
    "a hand grasping an object":                "grasp",
    "a hand moving an object to another location": "move_to",
    "a hand pouring liquid from a container":   "pour",
    "a hand cutting with a knife":              "cut",
    "a hand opening a door or cabinet":         "open",
    "a hand closing a door or cabinet":         "close",
    "a hand pushing an object":                 "push",
    "a hand pulling an object":                 "pull",
    "a hand pressing a button":                 "press",
    "a hand rotating a knob":                   "rotate",
    "a hand inspecting an object":              "inspect",
    "a hand removing an object from a surface": "remove_from",
}


@dataclass
class Node:
    name: str
    position: Tuple[float, float, float]
    is_target: bool


@dataclass
class Action:
    source: str
    action_type: str
    target: str
    weight: float


@dataclass
class Graph:
    nodes: List[Node] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    relations: List[Tuple[str, str, str]] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# HandTrajectory
# ──────────────────────────────────────────────────────────────────────────────

class HandTrajectory:
    """
    Rastrea, por mano, la posición 2D de la muñeca, el tamaño del bbox del
    objeto con el que interactúa, y la ORIENTACIÓN de la mano (ángulo del
    vector muñeca -> nudillo medio), en una ventana deslizante. Se usa para
    desambiguar grasp/push/pull/rotate por MOVIMIENTO, donde CLIP zero-shot
    demostró ser poco confiable (ver evaluate_chunks_against_gt.py).

    Heurística:
      - Poco desplazamiento neto + poco cambio de ángulo -> 'grasp'
        (mano de verdad quieta)
      - Poco desplazamiento neto + cambio de ángulo grande -> 'rotate'
        (mano gira en el sitio: perilla, tapa, muñeca girando el objeto)
      - Camino curvo (traslación) respecto al desplazamiento neto -> 'rotate'
      - Camino lineal + bbox del objeto CRECE (se acerca a cámara) -> 'pull'
      - Camino lineal + bbox del objeto DECRECE (se aleja) -> 'push'
      - Si no hay suficientes frames -> 'unknown' (CLIP decide)
    """

    def __init__(self, window_size: int = 8,
                 static_threshold_px: float = 12.0,
                 curvature_threshold: float = 0.5,
                 area_change_threshold: float = 0.08,
                 angle_threshold_deg: float = 15.0):
        self.window_size = window_size
        self.static_threshold_px = static_threshold_px
        self.curvature_threshold = curvature_threshold
        self.area_change_threshold = area_change_threshold
        self.angle_threshold_deg = angle_threshold_deg
        self._positions: Dict[str, deque] = {}
        self._areas: Dict[str, deque] = {}
        self._angles: Dict[str, deque] = {}

    def update(self, hand_label: str, wx: int, wy: int,
               obj_bbox: Optional[Tuple[int, int, int, int]] = None,
               orientation_point: Optional[Tuple[int, int]] = None) -> None:
        if hand_label not in self._positions:
            self._positions[hand_label] = deque(maxlen=self.window_size)
            self._areas[hand_label] = deque(maxlen=self.window_size)
            self._angles[hand_label] = deque(maxlen=self.window_size)
        self._positions[hand_label].append((wx, wy))
        if obj_bbox is not None:
            xmin, ymin, xmax, ymax = obj_bbox
            area = max(0, xmax - xmin) * max(0, ymax - ymin)
            self._areas[hand_label].append(area)
        if orientation_point is not None:
            ox, oy = orientation_point
            angle = math.atan2(oy - wy, ox - wx)
            self._angles[hand_label].append(angle)

    def reset(self, hand_label: str) -> None:
        self._positions.pop(hand_label, None)
        self._areas.pop(hand_label, None)
        self._angles.pop(hand_label, None)

    @staticmethod
    def _angular_range(angles: List[float]) -> float:
        """Suma de los cambios angulares absolutos frame a frame (grados),
        manejando correctamente el wraparound de -pi/pi."""
        if len(angles) < 2:
            return 0.0
        total = 0.0
        for a, b in zip(angles[:-1], angles[1:]):
            d = b - a
            d = (d + math.pi) % (2 * math.pi) - math.pi
            total += abs(d)
        return math.degrees(total)

    def classify_motion(self, hand_label: str) -> Tuple[str, float]:
        positions = self._positions.get(hand_label)
        if positions is None or len(positions) < self.window_size:
            return "unknown", 0.0

        pts = np.array(positions, dtype=float)
        net_disp = float(np.linalg.norm(pts[-1] - pts[0]))
        path_length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

        angles = list(self._angles.get(hand_label, []))
        angular_range = (self._angular_range(angles)
                         if len(angles) >= self.window_size else None)

        if net_disp < self.static_threshold_px:
            # Mano no se traslada, pero puede estar rotando en el sitio
            if angular_range is not None and angular_range > self.angle_threshold_deg:
                confidence = min(angular_range / (self.angle_threshold_deg * 3), 1.0)
                return "rotate", round(confidence, 2)
            confidence = 1.0 - (net_disp / self.static_threshold_px)
            return "grasp", round(max(confidence, 0.3), 2)

        curvature = 0.0 if net_disp == 0 else (path_length / net_disp) - 1.0
        if curvature > self.curvature_threshold:
            confidence = min(curvature / (self.curvature_threshold * 2), 1.0)
            return "rotate", round(confidence, 2)

        areas = self._areas.get(hand_label)
        if areas and len(areas) >= 2 and areas[0] > 0:
            area_change = (areas[-1] - areas[0]) / areas[0]
            if abs(area_change) >= self.area_change_threshold:
                action = "pull" if area_change > 0 else "push"
                confidence = min(abs(area_change) / (self.area_change_threshold * 3), 1.0)
                return action, round(confidence, 2)

        return "unknown", 0.0


# ──────────────────────────────────────────────────────────────────────────────
# VLMEncoder
# ──────────────────────────────────────────────────────────────────────────────

class VLMEncoder:
    def __init__(self, backend: str = "clip", device: str = "cuda",
                 model_id: str = "openai/clip-vit-base-patch32"):
        self.backend = backend
        self.device = device

        if backend == "clip":
            self.model_id = model_id
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id).to(device)
            self.model.eval()

        elif backend == "llava":
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
            self.processor = LlavaNextProcessor.from_pretrained(model_id)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=torch.float16,
                load_in_4bit=True, device_map="auto",
            )
        else:
            raise ValueError(f"Backend no soportado: '{backend}'.")

    @torch.no_grad()
    def classify_action(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        if self.backend == "clip":
            inputs = self.processor(
                text=CANDIDATE_ACTIONS, images=pil_img,
                return_tensors="pt", padding=True,
            ).to(self.device)
            logits = self.model(**inputs).logits_per_image
            probs = logits.softmax(dim=-1).squeeze()
            best_idx = int(probs.argmax())
            description = CANDIDATE_ACTIONS[best_idx]
            return CLIP_TO_ACTION[description], float(probs[best_idx])
        elif self.backend == "llava":
            prompt = (
                "[INST] <image>\nIn ONE short phrase (max 8 words), what "
                "action is the hand performing? Choose from: grasp, move_to, "
                "pour, cut, open, close, push, pull, press, rotate, inspect, "
                "remove_from. [/INST]"
            )
            inputs = self.processor(prompt, pil_img, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=20)
            text = self.processor.decode(out[0], skip_special_tokens=True).strip().lower()
            action = "grasp"
            for act in CLIP_TO_ACTION.values():
                if act.replace("_", " ") in text:
                    action = act
                    break
            return action, 0.0

    @staticmethod
    def make_crop(frame_bgr: np.ndarray, hand_wrist: Tuple[int, int],
                  obj_bbox: Tuple[int, int, int, int], padding: int = 80) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        wx, wy = hand_wrist
        xmin, ymin, xmax, ymax = obj_bbox
        x0 = max(0, min(wx, xmin) - padding)
        y0 = max(0, min(wy, ymin) - padding)
        x1 = min(w, max(wx, xmax) + padding)
        y1 = min(h, max(wy, ymax) + padding)
        return frame_bgr[y0:y1, x0:x1]


# ──────────────────────────────────────────────────────────────────────────────
# StoryTelling
# ──────────────────────────────────────────────────────────────────────────────

class StoryTelling:

    def __init__(self, video, path, output_path, model_path, confidence,
                 N=5, run_name="prueba1", device="cuda",
                 vlm_backend: str = "clip"):

        self.gt: Optional[EpicGroundTruth] = None
        self.video_id: Optional[str] = None
        self.video = video
        self.path = path
        self.output = output_path
        self.device = device
        self.modelpose = YOLO(model_path).to(self.device)
        self.confidence = confidence
        self.N = N
        self.run_name = run_name
        self.confirmed = {}
        self.scene = []
        self.semantic_line: List[Dict] = []

        self._frame_counter: Dict[str, int] = {}

        self.vlm = VLMEncoder(backend=vlm_backend, device=device)

        self.trajectory = HandTrajectory()
        self._motion_refined_actions = {"grasp", "push", "pull", "rotate"}

        options = HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self._hand_options = options
        self.hands = HandLandmarker.create_from_options(options)
        self.reinit_hands_every = 2000
        self._frames_since_hand_reinit = 0

    def get_video_info(self):
        self.cap = cv2.VideoCapture(self.path)
        self.ret, self.frame = self.cap.read()
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.output, fourcc, self.fps, (self.width, self.height))

    def _maybe_reinit_hand_landmarker(self) -> None:
        if not self.reinit_hands_every:
            return
        self._frames_since_hand_reinit += 1
        if self._frames_since_hand_reinit >= self.reinit_hands_every:
            try:
                self.hands.close()
            except Exception:
                pass
            self.hands = HandLandmarker.create_from_options(self._hand_options)
            self._frames_since_hand_reinit = 0
            import gc
            gc.collect()

    def process_frame_hands(self, frame: np.ndarray, yolo_result,
                             framecounter: int) -> Optional[Dict]:
        self._maybe_reinit_hand_landmarker()
        timestamp_ms = int(framecounter * 1000 / self.fps)

        current_frame_data = {
            "frame_index": framecounter, "timestamp_ms": timestamp_ms,
            "hands": [], "objects": [],
        }

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_results = self.hands.detect_for_video(mp_image, timestamp_ms)

        if not mp_results.hand_landmarks:
            return None

        boxes, clss, confs, masks_xy = [], [], [], None
        if yolo_result.boxes is not None:
            boxes    = yolo_result.boxes.xyxy.cpu().numpy()
            clss     = yolo_result.boxes.cls.cpu().numpy()
            confs    = yolo_result.boxes.conf.cpu().numpy()
            masks_xy = yolo_result.masks.xy if yolo_result.masks is not None else None

        seen_hands: set = set()

        for i, hand_landmarks in enumerate(mp_results.hand_landmarks):
            handedness = mp_results.handedness[i][0]
            label = f"{handedness.category_name.lower()}_hand"
            seen_hands.add(label)
            self._frame_counter[label] = self._frame_counter.get(label, 0) + 1
            if self._frame_counter[label] < self.N:
                continue

            keypoints = {}
            hand_px: List[Tuple[int, int]] = []
            for idx, kname in enumerate(HAND_KEYPOINT_NAMES):
                lm = hand_landmarks[idx]
                px = int(lm.x * self.width)
                py = int(lm.y * self.height)
                hand_px.append((px, py))
                keypoints[kname] = {
                    "x_pixel": px, "y_pixel": py, "z_depth": float(lm.z),
                    "x_norm": float(lm.x), "y_norm": float(lm.y),
                }

            wx = int(hand_landmarks[0].x * self.width)
            wy = int(hand_landmarks[0].y * self.height)
            # Punto de referencia para el ángulo de orientación de la mano
            mid_mcp = keypoints["MIDDLE_FINGER_MCP"]
            orientation_point = (mid_mcp["x_pixel"], mid_mcp["y_pixel"])

            hand_entry = {
                "label": label, "wrist_center": [wx, wy],
                "confidence": round(float(handedness.score), 4),
                "keypoints": keypoints,
            }

            interactions = []
            for idx, box in enumerate(boxes):
                xmin, ymin, xmax, ymax = map(int, box)
                obj_label = yolo_result.names[int(clss[idx])]
                if obj_label == "person":
                    continue
                margin = 50
                touching = any(
                    (xmin - margin) <= px <= (xmax + margin)
                    and (ymin - margin) <= py <= (ymax + margin)
                    for px, py in hand_px
                )
                if touching:
                    self.trajectory.update(
                        label, wx, wy, (xmin, ymin, xmax, ymax),
                        orientation_point=orientation_point,
                    )

                    crop = VLMEncoder.make_crop(frame, (wx, wy),
                                                (xmin, ymin, xmax, ymax))
                    action, action_conf = self.vlm.classify_action(crop)

                    if action in self._motion_refined_actions:
                        motion_action, motion_conf = self.trajectory.classify_motion(label)
                        if motion_action != "unknown":
                            action, action_conf = motion_action, motion_conf

                    gt_action = None
                    if self.gt is not None and self.video_id is not None:
                        seg = self.gt.segment_at_frame(self.video_id, framecounter)
                        if seg is not None:
                            gt_action = EPIC_TO_ROBOT.get(seg.verb)

                    interactions.append({
                        "object": obj_label,
                        "confidence": round(float(confs[idx]), 4),
                        "bbox": [xmin, ymin, xmax, ymax],
                        "polygon": (masks_xy[idx].astype(int).tolist()
                                   if masks_xy is not None else []),
                        "action": action,
                        "action_conf": round(action_conf, 4),
                        "gt_action": gt_action,
                    })

                    self._add_semantic_event(
                        framecounter, timestamp_ms, label, action, obj_label,
                        action_conf, gt_action,
                    )

            if interactions:
                hand_entry["interacting_with"] = interactions
            else:
                self.trajectory.reset(label)

            current_frame_data["hands"].append(hand_entry)

            for lm in hand_landmarks:
                cv2.circle(frame, (int(lm.x * self.width), int(lm.y * self.height)),
                           4, (0, 255, 0), -1)

        for lbl in list(self._frame_counter):
            if lbl not in seen_hands:
                self._frame_counter[lbl] = 0

        if not current_frame_data["hands"]:
            return None
        return current_frame_data

    def _add_semantic_event(self, frame_index, timestamp_ms, hand, action,
                             obj_label, action_conf,
                             gt_action: Optional[str] = None) -> None:
        if self.semantic_line:
            last = self.semantic_line[-1]
            if (last["hand"] == hand and last["action"] == action
                    and last["object"] == obj_label):
                return
        self.semantic_line.append({
            "frame_index": frame_index, "timestamp_ms": timestamp_ms,
            "hand": hand, "action": action, "object": obj_label,
            "action_conf": round(action_conf, 4), "gt_action": gt_action,
        })

    def print_semantic_line(self) -> None:
        print("\n── Línea Semántica ───────────────────────────────────────")
        for e in self.semantic_line:
            t = e["timestamp_ms"] / 1000
            marca = f"  (gt={e['gt_action']})" if e.get("gt_action") else ""
            print(f"  [{t:6.2f}s] {e['hand']:12s} "
                  f"--({e['action']:12s})--> {e['object']:15s}  "
                  f"conf={e['action_conf']:.2f}{marca}")
        print("──────────────────────────────────────────────────────────\n")

    def evaluate_detector_vs_gt(self) -> Optional[Dict]:
        pairs = [(e["action"], e["gt_action"]) for e in self.semantic_line
                 if e.get("gt_action") is not None]
        if not pairs:
            return None
        from collections import defaultdict
        total = len(pairs)
        correct = sum(1 for pred, gt in pairs if pred == gt)
        confusion = defaultdict(lambda: defaultdict(int))
        per_class_total = defaultdict(int)
        per_class_correct = defaultdict(int)
        for pred, gt in pairs:
            confusion[gt][pred] += 1
            per_class_total[gt] += 1
            if pred == gt:
                per_class_correct[gt] += 1
        return {
            "total": total, "correct": correct,
            "accuracy": round(correct / total, 3),
            "per_class_accuracy": {
                c: round(per_class_correct[c] / per_class_total[c], 3)
                for c in per_class_total
            },
            "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
        }

    def print_evaluation(self) -> None:
        result = self.evaluate_detector_vs_gt()
        if result is None:
            print("Sin ground truth para evaluar.")
            return
        print(f"\nAccuracy global: {result['accuracy']*100:.1f}% "
              f"({result['correct']}/{result['total']})")
        for cls, acc in sorted(result["per_class_accuracy"].items()):
            print(f"    {cls:12s}: {acc*100:.1f}%")

    def create_story(self, data: Dict) -> Graph:
        graph = Graph()
        for obj_label, pos in data.get("object_positions", {}).items():
            graph.nodes.append(Node(
                name=obj_label, position=tuple(pos),
                is_target=(obj_label in data.get("target_objects", [])),
            ))
        robot_parts = {src for (src, _, _) in data.get("edge_labels", {})
                       if src.startswith("robot_")}
        for rp in robot_parts:
            graph.nodes.append(Node(name=rp, position=(0.0, 0.0, 0.0), is_target=False))
        for (src, action_type, tgt), weight in data.get("edge_labels", {}).items():
            graph.actions.append(Action(source=src, action_type=action_type,
                                        target=tgt, weight=weight))
        detected = data.get("detected_objects", [])
        for i, a in enumerate(detected):
            for b in detected[i + 1:]:
                graph.relations.append((a, "co_detected", b))
        return graph

    def pipeline(self):
        self.get_video_info()
        framecounter = 0
        with tqdm(total=self.total_frames, desc="StoryTelling", unit="frame") as pbar:
            while self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
                if not self.ret:
                    break
                yolo_result = self.modelpose(self.frame, conf=self.confidence, verbose=False)[0]
                frame_data = self.process_frame_hands(self.frame, yolo_result, framecounter)
                if frame_data is not None:
                    self.scene.append(frame_data)
                annotated = yolo_result.plot(img=self.frame)
                self.out.write(annotated)
                pbar.update(1)
                framecounter += 1
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        self.print_semantic_line()
        self.print_evaluation()

    def save_scene(self):
        Path(f"output/{self.run_name}").mkdir(parents=True, exist_ok=True)
        with open(f"output/{self.run_name}/scene.json", "w") as f:
            json.dump(self.scene, f, indent=4)
        with open(f"output/{self.run_name}/semantic_line.json", "w") as f:
            json.dump(self.semantic_line, f, indent=4)


if __name__ == "__main__":
    base_folder = Path(__file__).resolve().parent
    story = StoryTelling(
        video="test.mp4", path=str(base_folder / "test.mp4"),
        output_path=str(base_folder / "output_mediapipe.mp4"),
        model_path="yoloe-11l-seg-pf.pt", confidence=0.5, N=5,
        run_name="prueba_hibrida", vlm_backend="clip",
    )
    story.pipeline()
    story.save_scene()