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
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import networkx as nx
from tqdm import tqdm

from videoAnalyzer import VideoAnalyzer
from storyTelling import StoryTelling, Graph


class UnifiedPipeline:

    def __init__(self, va: VideoAnalyzer, st: StoryTelling):
        self.va = va
        self.st = st
        self.graph: Optional[Graph] = None

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

                # 2. MediaPipe + CLIP  (usa yolo_result de arriba)
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
        self.st.print_semantic_line()

    # def generate_scenario(
    #     self,
    #     semantic_graph: nx.MultiDiGraph,
    #     goal: Optional[str] = None,
    # ) -> Dict:
    #     """
    #     Parámetro:
    #       semantic_graph : el nx.MultiDiGraph de SemanticActionGraph.graph
    #                        (necesario para iterar todas las aristas posibles
    #                         y asignarles un peso, igual que edge_labels)
    #       goal           : string del objetivo. Si es None, se infiere de
    #                        la línea semántica.

    #     Devuelve un dict listo para GNNTrainer.train_supervised().
    #     """

    #     detected_objects = list(self.va.confirmed.keys())
    #     object_positions: Dict[str, Tuple] = {
    #         label: (
    #             float(obj.pos.get("cx", 0)),
    #             float(obj.pos.get("cy", 0)),
    #             float(obj.pos.get("z0", 0)),
    #         )
    #         for label, obj in self.va.confirmed.items()
    #         if obj.pos
    #     }

    #     required_actions = list({
    #         e["action"] for e in self.st.semantic_line
    #     })
    #     target_objects = list({
    #         e["object"] for e in self.st.semantic_line
    #     })

    #     if goal is None:
    #         goal = self._infer_goal(target_objects, required_actions)

    #     # Para cada arista del grafo semántico completo (SemanticActionGraph),
    #     # calcula el peso basándose en lo que se observó en el video.
    #     # Misma lógica que _compute_heuristic_factor del trainer,
    #     # pero usando distancias 3D reales en lugar de posiciones sintéticas.
    #     edge_labels: Dict[Tuple, float] = {}
    #     for u, v, _, data in semantic_graph.edges(keys=True, data=True):
    #         action = data.get("action", "unknown")
    #         weight = self._compute_edge_weight(
    #             source          = u,
    #             target          = v,
    #             action          = action,
    #             target_objects  = target_objects,
    #             required_actions= required_actions,
    #             detected_objects= detected_objects,
    #             object_positions= object_positions,
    #         )
    #         edge_labels[(u, action, v)] = weight

    #     scenario = {
    #         "goal":             goal,
    #         "target_objects":   target_objects,
    #         "required_actions": required_actions,
    #         "detected_objects": detected_objects,
    #         "object_positions": object_positions,
    #         "edge_labels":      edge_labels,
    #     }

    #     print(f"\nEscenario generado")
    #     print(f"  goal             : {goal}")
    #     print(f"  target_objects   : {target_objects}")
    #     print(f"  required_actions : {required_actions}")
    #     print(f"  detected_objects : {detected_objects}")
    #     print(f"  edge_labels      : {len(edge_labels)} aristas")

    #     return scenario

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
        """
        Construye una descripción del goal a partir de la línea semántica.
        Ejemplos:
          grasp(cup) + move_to(table)  - 'Move the cup to the table'
          open(cabinet)                - 'Open the cabinet'
          grasp(knife) + cut(avocado)  - 'Cut the avocado with the knife'
        Si hay LLaVA disponible (self.st.vlm.backend == 'llava'), usa una
        llamada al VLM sobre el primer crop con interacción.
        """

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
        """

        Escala de pesos:
          0.10 – 0.30 : acción correcta, objetos cerca y detectados
          0.30 – 0.80 : acción correcta pero con penalizaciones (distancia, no detectado)
          0.80 – 1.20 : acción plausible pero no requerida
          1.20 – 1.95 : acción incorrecta o inviable
        """
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

    # def build_perceptual_graph(self):
    #     """
    #     Convierte VideoAnalyzer.confirmed → PerceptualGraph (InstanceNode).
    #     Necesario para pasarle al GNN en inferencia:
    #       gnn_optimizer.optimize_costs(semantic_graph, perceptual_graph, goal, ...)
    #     """
    #     # Importa la clase real que el GNN usa (definida en el notebook)
    #     from perceptual_knowledge import PerceptualGraph as PG

    #     pg = PG()
    #     for label, obj in self.va.confirmed.items():
    #         if not obj.pos:
    #             continue
    #         pos = (obj.pos["cx"], obj.pos["cy"], obj.pos["z0"])
    #         conf = obj.pos.get("confidence", 0.9)
    #         bbox_w, bbox_h = 50, 50   # aproximación si no tienes bbox exacto
    #         pg.add_instance(
    #             concept    = label,
    #             position   = pos,
    #             confidence = conf,
    #             bbox       = (pos[0] - bbox_w/2, pos[1] - bbox_h/2, bbox_w, bbox_h),
    #         )
    #     pg.compute_spatial_relations()
    #     return pg

    # ──────────────────────────────────────────────────────────────────
    # Guardado
    # ──────────────────────────────────────────────────────────────────

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
            # Serializa las claves tuple de edge_labels a strings
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
    RUN  = "prueba_unificada"

    va = VideoAnalyzer(
        video       = "test.mp4",
        path        = str(base / "test.mp4"),
        output_path = str(base / f"output_{RUN}.mp4"),
        model_path  = str(ws  / "yoloe-11l-seg-pf.pt"),
        confidence  = 0.5,
        info_path   = str(ws  / "src/ar_perception/models3d/canonical.json"),
        alpha       = 0.5,
        N           = 5,
        run_name    = RUN,
    )

    st = StoryTelling(
        video        = "test.mp4",
        path         = str(base / "test.mp4"),
        output_path  = str(base / f"output_{RUN}.mp4"),
        model_path   = str(ws  / "yoloe-11l-seg-pf.pt"),
        confidence   = 0.5,
        N            = 5,
        run_name     = RUN,
        vlm_backend  = "clip",
    )

    pipeline = UnifiedPipeline(va, st)
    pipeline.run()

    scenario = pipeline.generate_scenario(goal=None)

    # El SemanticActionGraph se inicializa igual que en el notebook
    # from robot_physical_capacities import RobotCapabilities
    # from knowledge_base import KnowledgeBase
    # from semantic_graph import SemanticActionGraph

    # robot_capabilities = RobotCapabilities()
    # knowledge_base     = KnowledgeBase()
    # sem_graph          = SemanticActionGraph(robot_capabilities, knowledge_base)
    # sem_graph.build_full_action_graph(
    #     knowledge_base.concepts.keys(),
    #     robot_capabilities.actions,
    # )

    # # Genera el escenario automáticamente desde el video
    # # goal=None → se infiere de la línea semántica de CLIP
    # scenario = pipeline.generate_scenario(
    #     semantic_graph = sem_graph.graph,
    #     goal           = None,
    # )

    pipeline.save(scenario)

    # Para acumular varios videos y entrenar:
    # all_scenarios = [scenario1, scenario2, ...]
    # trainer.train_supervised(labeled_examples=all_scenarios, epochs=300)