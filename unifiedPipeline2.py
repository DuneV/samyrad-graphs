"""
unified_pipeline.py
===================
Un único pase por video usando VideoAnalyzer + StoryTelling.

  VideoAnalyzer.process_frame()      → YOLO + Depth (una vez)
  StoryTelling.process_frame_hands() → MediaPipe + CLIP (usa YOLO de arriba)

Al terminar, generate_scenario() produce un dict con el mismo formato
que training_data.py, listo para pasarse a GNNTrainer.train_supervised().
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import networkx as nx
from tqdm import tqdm

from videoAnalyzer import VideoAnalyzer
from storyTelling import StoryTelling, Graph
from epic_ground_truth import EpicGroundTruth

# PXX_YY o PXX_1YY (extensión de EPIC-KITCHENS-100)
EPIC_VIDEO_ID_RE = re.compile(r"(P\d{2}_\d{2,3})")


class UnifiedPipeline:

    def __init__(self, va: VideoAnalyzer, st: StoryTelling,
                 ground_truth: Optional[EpicGroundTruth] = None):
        self.va = va
        self.st = st
        self.graph: Optional[Graph] = None

        # Ground truth de EPIC-KITCHENS (opcional). Si se pasa aquí, se
        # propaga a StoryTelling y se detecta el video_id automáticamente
        # a partir del nombre del archivo de video (ej. ".../P01_11.MP4").
        if ground_truth is not None:
            self.st.gt = ground_truth
            self.st.video_id = self._detect_epic_video_id(self.va.path)
            if self.st.video_id and not ground_truth.has_video(self.st.video_id):
                print(f"⚠ video_id '{self.st.video_id}' detectado en el nombre "
                      f"del archivo pero no está en las anotaciones cargadas "
                      f"(¿cargaste train.csv y validation.csv?). Se usará CLIP.")
                self.st.video_id = None
            elif self.st.video_id:
                print(f"✓ Ground truth activo para video_id='{self.st.video_id}'")

    @staticmethod
    def _detect_epic_video_id(video_path: str) -> Optional[str]:
        """Extrae 'P01_11' de un path tipo '.../videos/P01_11.MP4'.
        Devuelve None si el video no parece ser de EPIC-KITCHENS
        (ej. tus propios videos grabados con el robot)."""
        match = EPIC_VIDEO_ID_RE.search(Path(video_path).stem)
        return match.group(1) if match else None

    def run(self) -> None:
        self.va.get_video_info()
        self.va.load_info_canonical()

        # StoryTelling necesita fps/width/height para MediaPipe
        self.st.fps    = self.va.fps
        self.st.width  = self.va.width
        self.st.height = self.va.height

        framecounter = 0

        with tqdm(total=self.va.total_frames, desc="UnifiedPipeline", unit="frame") as pbar:
            while self.va.cap.isOpened():
                ret, frame = self.va.cap.read()
                if not ret:
                    break

                # 1. YOLO + Depth
                yolo_result, depth_np = self.va.process_frame(frame, framecounter)

                # 2. MediaPipe + (ground truth si existe, si no CLIP)
                frame_data = self.st.process_frame_hands(frame, yolo_result, framecounter)
                if frame_data is not None:
                    self.st.scene.append(frame_data)

                self.va.vectors.append(depth_np)

                annotated = yolo_result.plot(img=frame.copy())
                self.va.out.write(annotated)

                pbar.update(1)
                framecounter += 1

        self.va.cap.release()
        self.va.out.release()
        cv2.destroyAllWindows()

        print(f"\nFrames procesados  : {framecounter}")
        print(f"Frames con manos   : {len(self.st.scene)}")
        print(f"Objetos confirmados: {list(self.va.confirmed.keys())}")
        print(f"Eventos semánticos : {len(self.st.semantic_line)}")
        self._print_gt_vs_clip_stats()
        self.st.print_semantic_line()

    def _print_gt_vs_clip_stats(self) -> None:
        """Cuántos eventos de semantic_line vinieron de ground truth (conf=1.0
        exacto y self.st.gt activo) vs de CLIP. Útil para saber cuánto te
        estás apoyando en anotaciones reales vs en inferencia visual."""
        if self.st.gt is None or self.st.video_id is None:
            print("Ground truth : no usado (video sin anotaciones EPIC-KITCHENS)")
            return

        total = len(self.st.semantic_line)
        from_gt = sum(1 for e in self.st.semantic_line if e["action_conf"] == 1.0)
        from_clip = total - from_gt
        pct = (from_gt / total * 100) if total else 0.0
        print(f"Ground truth vs CLIP: {from_gt}/{total} ({pct:.1f}%) desde EPIC-KITCHENS, "
              f"{from_clip} inferidos con CLIP")

    def generate_scenario(self, goal: Optional[str] = None) -> Dict:

        detected_objects = list(self.va.confirmed.keys())

        object_positions = {
            label: (obj.pos["cx"], obj.pos["cy"], obj.pos["z0"])
            for label, obj in self.va.confirmed.items() if obj.pos
        }

        required_actions = list({e["action"] for e in self.st.semantic_line})
        target_objects   = list({e["object"] for e in self.st.semantic_line})

        if goal is None:
            goal = self._infer_goal(target_objects, required_actions)

        edge_labels = {}

        for event in self.st.semantic_line:
            robot  = f"robot_{event['hand']}"
            action = event["action"]
            obj    = event["object"]

            edge_labels[(robot, action, obj)] = self._weight_from_distance(
                robot, obj, object_positions, base=0.20
            )

            if action == "move_to":
                for dest in target_objects:
                    if dest != obj:
                        edge_labels[(obj, action, dest)] = self._weight_from_distance(
                            obj, dest, object_positions, base=0.25
                        )

        for obj in detected_objects:
            if obj not in target_objects:
                for action in required_actions:
                    edge_labels[("robot_right_hand", action, obj)] = 1.70

        for target in target_objects:
            if target not in detected_objects:
                for action in required_actions:
                    edge_labels[("robot_right_hand", action, target)] = 1.85

        return {
            "goal":             goal,
            "target_objects":   target_objects,
            "required_actions": required_actions,
            "detected_objects": detected_objects,
            "object_positions": object_positions,
            "edge_labels":      edge_labels,
            "source":           "epic_ground_truth" if (self.st.gt and self.st.video_id) else "clip",
        }

    def _weight_from_distance(self, src, tgt, positions, base=0.20) -> float:
        if src in positions and tgt in positions:
            p1 = np.array(positions[src])
            p2 = np.array(positions[tgt])
            dist = float(np.linalg.norm(p1 - p2))
            return round(min(base + dist / 500.0, 1.90), 2)
        return round(base + 0.30, 2)

    def _infer_goal(self, target_objects: List[str],
                    required_actions: List[str]) -> str:
        if self.st.vlm.backend == "llava" and self.st.scene:
            for frame_data in self.st.scene:
                for hand in frame_data.get("hands", []):
                    for inter in hand.get("interacting_with", []):
                        break

        if not self.st.semantic_line:
            return "Perform robot task"

        events = self.st.semantic_line
        first_obj  = events[0]["object"]
        last_obj   = events[-1]["object"]
        first_act  = events[0]["action"]

        templates = {
            "move_to":    f"Move the {first_obj} to the {last_obj}",
            "grasp":      f"Grasp the {first_obj}",
            "pour":       f"Pour from the {first_obj} into the {last_obj}",
            "cut":        f"Cut the {last_obj} with the {first_obj}",
            "open":       f"Open the {first_obj}",
            "close":      f"Close the {first_obj}",
            "push":       f"Push the {first_obj}",
            "pull":       f"Pull the {first_obj}",
            "press":      f"Press the {first_obj}",
            "rotate":     f"Rotate the {first_obj}",
            "inspect":    f"Inspect the {first_obj}",
            "remove_from":f"Remove the {first_obj} from the {last_obj}",
        }
        return templates.get(first_act, f"Perform {first_act} on {first_obj}")

    def _compute_edge_weight(
        self,
        source: str,
        target: str,
        action: str,
        target_objects: List[str],
        required_actions: List[str],
        detected_objects: List[str],
        object_positions: Dict[str, Tuple],
    ) -> float:
        src_concept = source.split("_")[0] if "_" in source else source
        tgt_concept = target.split("_")[0] if "_" in target else target

        target_set   = set(target_objects)
        required_set = set(required_actions)
        detected_set = set(detected_objects)

        observed = any(
            e["action"] == action and e["object"] == tgt_concept
            for e in self.st.semantic_line
        )

        distance: Optional[float] = None
        if src_concept in object_positions and tgt_concept in object_positions:
            p1 = np.array(object_positions[src_concept])
            p2 = np.array(object_positions[tgt_concept])
            distance = float(np.linalg.norm(p1 - p2))

        near = distance is not None and distance < 150

        if observed and near:
            w = 0.20
        elif observed:
            w = 0.28
        elif action in required_set and tgt_concept in target_set and tgt_concept in detected_set and near:
            w = 0.25
        elif action in required_set and tgt_concept in target_set and tgt_concept in detected_set:
            w = 0.35
        elif action in required_set and tgt_concept in target_set:
            w = 0.50
        elif action in required_set and tgt_concept in detected_set and near:
            w = 0.55
        elif action in required_set and tgt_concept in detected_set:
            w = 0.65
        elif action in required_set:
            w = 0.80
        elif tgt_concept in target_set and tgt_concept in detected_set and near:
            w = 0.75
        elif tgt_concept in target_set and tgt_concept in detected_set:
            w = 0.90
        elif tgt_concept in target_set:
            w = 1.10
        elif tgt_concept in detected_set and near:
            w = 1.20
        elif tgt_concept in detected_set:
            w = 1.50
        else:
            w = 1.85

        return round(w, 2)

    def save(self, scenario: Optional[Dict] = None) -> None:
        run = self.va.run_name
        out_dir = Path(f"output/{run}")
        out_dir.mkdir(parents=True, exist_ok=True)

        self.st.save_scene()
        self.va.save(str(out_dir / "depth_vectors"))

        confirmed_data = {
            label: obj.pos
            for label, obj in self.va.confirmed.items()
            if obj.pos
        }
        with open(out_dir / "confirmed_objects.json", "w") as f:
            json.dump(confirmed_data, f, indent=2)

        if scenario is not None:
            serializable = dict(scenario)
            serializable["edge_labels"] = {
                str(k): v for k, v in scenario["edge_labels"].items()
            }
            with open(out_dir / "scenario.json", "w") as f:
                json.dump(serializable, f, indent=2)
            print(f"  scenario.json  (listo para train_supervised())")

        print(f"\nGuardado en output/{run}/")
        print(f"  scene.json             ({len(self.st.scene)} frames)")
        print(f"  semantic_line.json     ({len(self.st.semantic_line)} eventos)")
        print(f"  confirmed_objects.json ({len(confirmed_data)} objetos)")


if __name__ == "__main__":
    from pathlib import Path

    base = Path(__file__).resolve().parent
    ws   = base.parent.parent
    RUN  = "P01_11"   # usa el video_id real si es un video de EPIC-KITCHENS

    va = VideoAnalyzer(
        video       = f"{RUN}.mp4",
        path        = str(base / f"{RUN}.mp4"),
        output_path = str(base / f"output_{RUN}.mp4"),
        model_path  = str(ws  / "yoloe-11l-seg-pf.pt"),
        confidence  = 0.5,
        info_path   = str(ws  / "src/ar_perception/models3d/canonical.json"),
        alpha       = 0.5,
        N           = 5,
        run_name    = RUN,
    )

    st = StoryTelling(
        video        = f"{RUN}.mp4",
        path         = str(base / f"{RUN}.mp4"),
        output_path  = str(base / f"output_{RUN}.mp4"),
        model_path   = str(ws  / "yoloe-11l-seg-pf.pt"),
        confidence   = 0.5,
        N            = 5,
        run_name     = RUN,
        vlm_backend  = "clip",
    )

    # Carga el ground truth UNA sola vez y pásalo al pipeline; él se encarga
    # de detectar el video_id (P01_11) desde el nombre del archivo y de
    # conectarlo con StoryTelling.
    gt = EpicGroundTruth(["EPIC_100_train.csv", "EPIC_100_validation.csv"])

    pipeline = UnifiedPipeline(va, st, ground_truth=gt)
    pipeline.run()

    scenario = pipeline.generate_scenario(goal=None)
    pipeline.save(scenario)