from tqdm import tqdm
from ultralytics import YOLO
import cv2
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
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
# VLMEncoder
# ──────────────────────────────────────────────────────────────────────────────

class VLMEncoder:
    """
    Clasifica la acción que realiza una mano sobre un objeto.

    backend='clip'  : zero-shot, rápido (~30 ms/frame), ~1 GB VRAM.
    backend='llava' : LLaVA-1.5-7B, lento (~1 s/frame), ~8 GB VRAM.
                      Requiere: pip install accelerate bitsandbytes
    """

    def __init__(self, backend: str = "clip", device: str = "cuda"):
        self.backend = backend
        self.device = device

        if backend == "clip":
            model_id = "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(model_id)
            self.model = CLIPModel.from_pretrained(model_id).to(device)
            self.model.eval()

        elif backend == "llava":
            from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
            model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
            self.processor = LlavaNextProcessor.from_pretrained(model_id)
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                load_in_4bit=True,
                device_map="auto",
            )

        else:
            raise ValueError(f"Backend no soportado: '{backend}'. Usa 'clip' o 'llava'.")

    @torch.no_grad()
    def classify_action(self, crop_bgr: np.ndarray) -> Tuple[str, float]:
        """
        Recibe un crop BGR, devuelve (action_type, confidence).
        action_type coincide con los tipos usados en edge_labels.
        """
        pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        if self.backend == "clip":
            inputs = self.processor(
                text=CANDIDATE_ACTIONS,
                images=pil_img,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
            logits = self.model(**inputs).logits_per_image
            probs = logits.softmax(dim=-1).squeeze()
            best_idx = int(probs.argmax())
            description = CANDIDATE_ACTIONS[best_idx]
            return CLIP_TO_ACTION[description], float(probs[best_idx])

        elif self.backend == "llava":
            prompt = (
                "[INST] <image>\n"
                "In ONE short phrase (max 8 words), what action is the hand performing? "
                "Choose from: grasp, move_to, pour, cut, open, close, push, pull, "
                "press, rotate, inspect, remove_from. [/INST]"
            )
            inputs = self.processor(prompt, pil_img, return_tensors="pt").to(self.device)
            out = self.model.generate(**inputs, max_new_tokens=20)
            text = self.processor.decode(out[0], skip_special_tokens=True).strip().lower()
            action = "grasp"
            for act in CLIP_TO_ACTION.values():
                if act.replace("_", " ") in text:
                    action = act
                    break
            return action, 0.0   # LLaVA no da probabilidad directa

    @staticmethod
    def make_crop(frame_bgr: np.ndarray,
                  hand_wrist: Tuple[int, int],
                  obj_bbox: Tuple[int, int, int, int],
                  padding: int = 80) -> np.ndarray:
        """Crop mínimo que encierra la muñeca y el bbox del objeto."""
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
        self.semantic_line: List[Dict] = []   # eventos clasificados por VLM

        # Estado de confirmación de manos entre frames
        self._frame_counter: Dict[str, int] = {}

        # VLMEncoder para clasificar acciones (SIEMPRE se usa para predecir)
        self.vlm = VLMEncoder(backend=vlm_backend, device=device)

        options = HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=confidence,
            min_tracking_confidence=confidence,
        )
        self.hands = HandLandmarker.create_from_options(options)

    def get_video_info(self):
        self.cap = cv2.VideoCapture(self.path)
        self.ret, self.frame = self.cap.read()
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.out = cv2.VideoWriter(self.output, fourcc, self.fps, (self.width, self.height))

    def process_frame_hands(
        self,
        frame: np.ndarray,
        yolo_result,
        framecounter: int,
    ) -> Optional[Dict]:
        """
        Corre MediaPipe sobre el frame.
        Recibe yolo_result ya calculado (de VideoAnalyzer o del pipeline propio).
        Si hay interacción mano-objeto, SIEMPRE clasifica la acción con
        VLMEncoder (CLIP/LLaVA). Si hay ground truth de EPIC-KITCHENS
        disponible para ese frame, se guarda aparte (gt_action) solo para
        comparar después con evaluate_detector_vs_gt() — nunca reemplaza
        la predicción del detector.
        Devuelve dict de frame o None si no hay manos confirmadas.
        """
        timestamp_ms = int(framecounter * 1000 / self.fps)

        current_frame_data = {
            "frame_index": framecounter,
            "timestamp_ms": timestamp_ms,
            "hands": [],
            "objects": [],
        }

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        mp_results = self.hands.detect_for_video(mp_image, timestamp_ms)

        if not mp_results.hand_landmarks:
            return None

        # Extrae bboxes del yolo_result recibido
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
                    "x_pixel": px, "y_pixel": py,
                    "z_depth": float(lm.z),
                    "x_norm": float(lm.x), "y_norm": float(lm.y),
                }

            wx = int(hand_landmarks[0].x * self.width)
            wy = int(hand_landmarks[0].y * self.height)

            hand_entry = {
                "label": label,
                "wrist_center": [wx, wy],
                "confidence": round(float(handedness.score), 4),
                "keypoints": keypoints,
            }

            # ── Interacciones con objetos ──────────────────────────────
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
                    # ── Clasificación de acción con VLM (SIEMPRE) ──────
                    crop = VLMEncoder.make_crop(frame, (wx, wy),
                                                (xmin, ymin, xmax, ymax))
                    action, action_conf = self.vlm.classify_action(crop)

                    # ── Ground truth SOLO para comparar, no reemplaza ──
                    gt_action = None
                    if self.gt is not None and self.video_id is not None:
                        seg = self.gt.segment_at_frame(self.video_id, framecounter)
                        if seg is not None:
                            gt_action = EPIC_TO_ROBOT.get(seg.verb)

                    interactions.append({
                        "object":         obj_label,
                        "confidence":     round(float(confs[idx]), 4),
                        "bbox":           [xmin, ymin, xmax, ymax],
                        "polygon":        (masks_xy[idx].astype(int).tolist()
                                           if masks_xy is not None else []),
                        "action":         action,
                        "action_conf":    round(action_conf, 4),
                        "gt_action":      gt_action,
                    })

                    # Agrega a la línea semántica (deduplica eventos iguales)
                    self._add_semantic_event(
                        framecounter, timestamp_ms, label, action, obj_label,
                        action_conf, gt_action
                    )

            if interactions:
                hand_entry["interacting_with"] = interactions

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

    def _add_semantic_event(self, frame_index, timestamp_ms,
                             hand, action, obj_label, action_conf,
                             gt_action: Optional[str] = None) -> None:
        """Agrega un evento a self.semantic_line evitando repetidos consecutivos.
        action     : predicción del detector (CLIP/LLaVA) — la que usa el resto
                     del pipeline (edge_labels, entrenamiento del GNN).
        gt_action  : ground truth de EPIC-KITCHENS si está disponible — SOLO
                     para evaluate_detector_vs_gt(), nunca sustituye a action.
        """
        if self.semantic_line:
            last = self.semantic_line[-1]
            if (last["hand"] == hand and last["action"] == action
                    and last["object"] == obj_label):
                return
        self.semantic_line.append({
            "frame_index":  frame_index,
            "timestamp_ms": timestamp_ms,
            "hand":         hand,
            "action":       action,
            "object":       obj_label,
            "action_conf":  round(action_conf, 4),
            "gt_action":    gt_action,
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
        """
        Compara las predicciones del detector (self.vlm) contra el ground
        truth de EPIC-KITCHENS. Solo evalúa eventos donde hay gt_action
        disponible (es decir, videos de EPIC-KITCHENS con self.gt/
        self.video_id seteados). Devuelve None si no hay nada que comparar.
        """
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
            "total": total,
            "correct": correct,
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
            print("Sin ground truth para evaluar (video no es de EPIC-KITCHENS "
                  "o self.gt/self.video_id no están seteados).")
            return
        print("\n── Evaluación detector vs. ground truth ─────────────────")
        print(f"  Accuracy global: {result['accuracy']*100:.1f}%  "
              f"({result['correct']}/{result['total']})")
        print("  Accuracy por acción (ground truth):")
        for cls, acc in sorted(result["per_class_accuracy"].items()):
            print(f"    {cls:12s}: {acc*100:.1f}%")
        print("  Matriz de confusión (fila=ground truth, columna=predicción):")
        for gt_cls, preds in sorted(result["confusion_matrix"].items()):
            print(f"    {gt_cls:12s}: {dict(preds)}")
        print("──────────────────────────────────────────────────────────\n")

    def create_story(self, data: Dict) -> Graph:
        """
        Construye un Graph a partir de un dict de escenario.
        data debe tener: object_positions, target_objects,
                         edge_labels, detected_objects.
        """
        graph = Graph()

        for obj_label, pos in data.get("object_positions", {}).items():
            graph.nodes.append(Node(
                name=obj_label,
                position=tuple(pos),
                is_target=(obj_label in data.get("target_objects", [])),
            ))

        robot_parts = {
            src
            for (src, _, _) in data.get("edge_labels", {})
            if src.startswith("robot_")
        }
        for rp in robot_parts:
            graph.nodes.append(Node(name=rp, position=(0.0, 0.0, 0.0), is_target=False))

        for (src, action_type, tgt), weight in data.get("edge_labels", {}).items():
            graph.actions.append(Action(
                source=src, action_type=action_type, target=tgt, weight=weight,
            ))

        detected = data.get("detected_objects", [])
        for i, a in enumerate(detected):
            for b in detected[i + 1:]:
                graph.relations.append((a, "co_detected", b))

        return graph

    def pipeline(self):
        """Pipeline standalone de StoryTelling (sin VideoAnalyzer)."""
        self.get_video_info()
        framecounter = 0

        with tqdm(total=self.total_frames, desc="StoryTelling", unit="frame") as pbar:
            while self.cap.isOpened():
                self.ret, self.frame = self.cap.read()
                if not self.ret:
                    break

                # YOLO corre una sola vez por frame
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
        print(f"Scene guardado: output/{self.run_name}/scene.json")
        print(f"Línea semántica: output/{self.run_name}/semantic_line.json")


if __name__ == "__main__":
    base_folder = Path(__file__).resolve().parent
    ws = base_folder.parent.parent

    story = StoryTelling(
        video="test.mp4",
        path=str(base_folder / "test.mp4"),
        output_path=str(base_folder / "output_mediapipe.mp4"),
        model_path="yoloe-11l-seg-pf.pt",
        confidence=0.5,
        N=5,
        run_name="prueba_hibrida",
        vlm_backend="clip",   # cambia a "llava" si tienes 8 GB VRAM
    )
    story.pipeline()
    story.save_scene()