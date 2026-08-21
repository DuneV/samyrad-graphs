#!/usr/bin/env python3
"""
benchmark_comparison.py
==========================
Compara, sobre el MISMO video, tres formas de generar/evaluar un plan:

  A) Tu pipeline completo: clasificador de acciones (CLIP/fine-tuned) +
     GNNCostOptimizer -- lo que realmente vas a correr en producción.
  B) Ground truth de EPIC-KITCHENS: el grafo se arma directo del verbo
     REAL anotado (sin pasar por ningún clasificador de visión) +
     GNNCostOptimizer -- aísla qué tan bueno es el GNN cuando la
     percepción es perfecta.
  C) VLA (Vision-Language-Action): plantilla lista para que conectes tu
     modelo -- ver la función run_vla_inference() más abajo, es el único
     bloque que necesitas completar.

Para cada uno mide: tiempo de inferencia y accuracy de las acciones
predichas contra el ground truth real (excepto B, que ES el ground
truth, así que su "accuracy" es por definición 100% -- se reporta como
referencia, no como resultado a comparar).

Uso:
    python3 benchmark_comparison.py \
        --video kitchen/EPIC-KITCHENS/P01/videos/P01_11.MP4 \
        --gt-csv annotations/EPIC_100_train.csv annotations/EPIC_100_validation.csv \
        --model-path yoloe-11l-seg-pf.pt \
        --gnn-checkpoint gnn_cost_optimizer.pth \
        --classifier-checkpoint action_classifier_finetuned.pth
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from epic_ground_truth import EpicGroundTruth
from robotAction import EPIC_TO_ROBOT

from neopath.semantic_knowledge import KnowledgeBase
from neopath.robot_physical_capacities import RobotCapabilities
from neopath.semantic_graph import SemanticActionGraph
from neopath.gnn import GNNCostOptimizer
from neopath.perceptual_knowledge import PerceptualGraph

EPIC_VIDEO_ID_RE = re.compile(r"(P\d{2}_\d{2,3})")


def detect_video_id(video_path: str) -> Optional[str]:
    match = EPIC_VIDEO_ID_RE.search(Path(video_path).stem)
    return match.group(1) if match else None


def build_perceptual_graph(object_positions: Dict[str, Tuple]) -> PerceptualGraph:
    pg = PerceptualGraph()
    for obj_name, position in object_positions.items():
        pg.add_instance(
            concept=obj_name, position=tuple(position),
            confidence=0.9, bbox=(position[0] - 25, position[1] - 25, 50, 50),
        )
    pg.compute_spatial_relations()
    return pg


def run_gnn_inference(gnn_optimizer: GNNCostOptimizer, semantic_graph_obj: SemanticActionGraph,
                      scenario: Dict) -> Tuple[float, Dict]:
    """Corre optimize_costs() y devuelve (tiempo_segundos, resultado)."""
    perceptual_graph = build_perceptual_graph(scenario["object_positions"])
    t0 = time.perf_counter()
    optimized_graph = gnn_optimizer.optimize_costs(
        semantic_graph_obj.graph, perceptual_graph, scenario["goal"],
        target_objects=scenario.get("target_objects"),
        required_actions=scenario.get("required_actions"),
    )
    elapsed = time.perf_counter() - t0
    return elapsed, optimized_graph


def accuracy_vs_ground_truth(predicted_actions: List[str], gt_actions: List[str]) -> float:
    """Accuracy simple: qué fracción de las acciones predichas coincide
    con las reales, comparando como conjuntos ordenados por frecuencia
    (no depende de un alineamiento frame-a-frame exacto)."""
    if not gt_actions:
        return float("nan")
    from collections import Counter
    pred_counts = Counter(predicted_actions)
    gt_counts = Counter(gt_actions)
    correct = sum(min(pred_counts[a], gt_counts[a]) for a in gt_counts)
    return correct / sum(gt_counts.values())


def build_scenario_from_classifier(video_path: str, model_path: str,
                                   confidence: float = 0.5) -> Tuple[Dict, float]:
    """
    Corre tu pipeline de percepción (YOLO + MediaPipe + clasificador) y
    arma el scenario dict. Devuelve (scenario, tiempo_total_segundos).

    NOTA: usa la ruta "no-chunked" (UnifiedPipeline) para tener un tiempo
    de principio a fin sobre un solo video, en vez de la ruta de
    subprocesos (que aísla memoria pero no es representativa del tiempo
    real de inferencia en producción).
    """
    import torch
    from videoAnalyzer import VideoAnalyzer
    from storyTelling import StoryTelling
    from unifiedPipeline2 import UnifiedPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_name = Path(video_path).stem

    t0 = time.perf_counter()

    va = VideoAnalyzer(
        video=video_path, path=video_path,
        output_path=f"/tmp/bench_{run_name}.mp4",
        model_path=model_path, confidence=confidence,
        info_path="canonical.json", alpha=0.5, N=5,
        run_name=run_name, device=device, store_depth_vectors=False,
    )
    st = StoryTelling(
        video=video_path, path=video_path,
        output_path=f"/tmp/bench_{run_name}.mp4",
        model_path=model_path, confidence=confidence, N=5,
        run_name=run_name, vlm_backend="clip", device=device,
    )

    pipeline = UnifiedPipeline(va, st, checkpoint_every=None, memory_log_every=None)
    pipeline.run()
    scenario = pipeline.generate_scenario(goal=None)
    scenario["semantic_line"] = st.semantic_line

    elapsed = time.perf_counter() - t0
    return scenario, elapsed


def build_scenario_from_ground_truth(video_id: str, gt: EpicGroundTruth,
                                     object_positions: Dict[str, Tuple],
                                     goal: Optional[str] = None) -> Tuple[Dict, float]:
    """
    Arma el scenario directo del verbo REAL anotado en EPIC-KITCHENS, sin
    pasar por ningún clasificador de visión. object_positions debe venir
    de una corrida previa de YOLO+Depth (la percepción espacial no cambia,
    solo la fuente de la ACCIÓN cambia).
    """
    t0 = time.perf_counter()

    segments = gt.segments_for_video(video_id)
    edge_labels = {}
    required_actions = set()
    target_objects = set()
    detected_objects = set(object_positions.keys())

    for seg in segments:
        action = EPIC_TO_ROBOT.get(seg.verb)
        if action is None:
            continue
        obj = seg.noun
        required_actions.add(action)
        target_objects.add(obj)
        detected_objects.add(obj)

        key = ("robot_right_hand", action, obj)
        if obj in object_positions and "robot_right_hand" in object_positions:
            p1 = np.array(object_positions["robot_right_hand"])
            p2 = np.array(object_positions[obj])
            dist = float(np.linalg.norm(p1 - p2))
            weight = round(min(0.2 + dist / 500.0, 1.9), 2)
        else:
            weight = 0.5
        edge_labels[key] = weight

    if goal is None:
        goal = (f"Perform {list(required_actions)[0]} on {list(target_objects)[0]}"
               if required_actions and target_objects else "Perform robot task")

    scenario = {
        "goal": goal,
        "target_objects": list(target_objects),
        "required_actions": list(required_actions),
        "detected_objects": list(detected_objects),
        "object_positions": object_positions,
        "edge_labels": edge_labels,
        "semantic_line": [{"action": EPIC_TO_ROBOT.get(s.verb), "object": s.noun}
                          for s in segments if EPIC_TO_ROBOT.get(s.verb) is not None],
    }
    elapsed = time.perf_counter() - t0
    return scenario, elapsed


_VLA_STATE = {"model": None, "processor": None}

# ── Umbrales de discretización (ver calibrate_vla_thresholds.py) ──────
# OpenVLA predice una acción continua de 7-DoF: [dx, dy, dz, droll,
# dpitch, dyaw, gripper]. Como NEOPATH usa un vocabulario discreto de 8
# clases, esto discretiza la acción continua con umbrales heurísticos.
# *** ESTOS UMBRALES SON UN PUNTO DE PARTIDA, NO ESTÁN CALIBRADOS ***
# Corre calibrate_vla_thresholds.py sobre un puñado de ventanas
# etiquetadas antes de confiar en los resultados finales -- reportar
# números de una heurística sin calibrar sería engañoso para el paper.
VLA_THRESHOLDS = {
    "gripper_close": -0.3,     # gripper delta por debajo de esto = cerrando fuerte
    "gripper_open": 0.3,       # gripper delta por encima de esto = abriendo fuerte
    "position_still": 0.01,    # magnitud de desplazamiento por debajo = "quieto"
    "rotation_dominant": 0.15, # magnitud de rotación por encima = domina sobre posición
}


def discretize_vla_action(action_7dof: np.ndarray,
                          thresholds: dict = VLA_THRESHOLDS) -> str:
    """
    Mapea la acción continua de 7-DoF de OpenVLA a una de las 8 clases
    del vocabulario de NEOPATH. Heurística de primer intento -- ver el
    aviso arriba sobre calibración.
    """
    dx, dy, dz, droll, dpitch, dyaw, gripper = action_7dof
    pos_mag = float(np.linalg.norm([dx, dy, dz]))
    rot_mag = float(np.linalg.norm([droll, dpitch, dyaw]))

    if gripper < thresholds["gripper_close"] and pos_mag < thresholds["position_still"]:
        return "grasp"
    if gripper > thresholds["gripper_open"] and pos_mag < thresholds["position_still"]:
        return "move_to"   # sin clase "release" propia; el más cercano
    if rot_mag > thresholds["rotation_dominant"] and rot_mag > pos_mag:
        return "rotate"
    if pos_mag < thresholds["position_still"]:
        return "press"
    # desplazamiento dominante: sin más contexto no se puede distinguir
    # push/pull/cut/pour/actuate de forma confiable con este heurístico
    return "displace"


def run_vla_inference(video_path: str, goal: str,
                      model_id: str = "openvla/openvla-7b",
                      n_steps: int = 8) -> Tuple[Optional[List[str]], float]:
    """
    Corre OpenVLA sobre frames muestreados del video, con el goal como
    instrucción en lenguaje natural. Cada paso produce una acción
    continua de 7-DoF que se discretiza a una clase del vocabulario de
    NEOPATH vía discretize_vla_action().

    Requiere ~16GB+ de VRAM en bf16 (cabe cómodo en una RTX 5090 de 32GB).
    """
    import cv2
    import torch
    from PIL import Image

    if _VLA_STATE["model"] is None:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        print(f"  Cargando {model_id}...")
        _VLA_STATE["processor"] = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True)
        _VLA_STATE["model"] = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True, trust_remote_code=True,
        ).to("cuda")

    processor = _VLA_STATE["processor"]
    model = _VLA_STATE["model"]

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_idxs = np.linspace(0, max(total_frames - 1, 0), n_steps, dtype=int)

    predicted_actions = []
    t0 = time.perf_counter()

    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        prompt = f"In: What action should the robot take to {goal.lower()}?\nOut:"
        inputs = processor(prompt, image).to("cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            action_7dof = model.predict_action(
                **inputs, unnorm_key="bridge_orig", do_sample=False)

        predicted_actions.append(discretize_vla_action(np.asarray(action_7dof)))

    elapsed = time.perf_counter() - t0
    cap.release()

    if not predicted_actions:
        return None, elapsed
    return predicted_actions, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--gt-csv", nargs="+", required=True)
    parser.add_argument("--model-path", default="yoloe-11l-seg-pf.pt")
    parser.add_argument("--gnn-checkpoint", required=True,
                        help="Checkpoint del GNN ya entrenado "
                             "(train_gnn_from_scenarios.py)")
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--skip-classifier-pipeline", action="store_true",
                        help="Si ya tienes el scenario de A) guardado en JSON, "
                             "pásalo con --classifier-scenario-json en vez de "
                             "recorrer el video de nuevo")
    parser.add_argument("--classifier-scenario-json", default=None)
    parser.add_argument("--output", default="benchmark_results.json")
    parser.add_argument("--skip-vla", action="store_true",
                        help="No corre la rama C) VLA -- útil cuando el "
                             "VLA vive en otra máquina (ej. servidor con "
                             "GPU distinta) y esta corrida solo genera A) "
                             "y B) para combinar después.")
    args = parser.parse_args()

    video_id = detect_video_id(args.video)
    if video_id is None:
        print(f"⚠ No se pudo detectar el video_id de EPIC-KITCHENS en {args.video}")

    gt = EpicGroundTruth(args.gt_csv)
    has_gt = video_id is not None and gt.has_video(video_id)
    if not has_gt:
        print(f"⚠ {video_id}: sin ground truth disponible -- solo se podrá "
              f"correr la rama A) y C), sin comparación de accuracy real.")

    results = {}

    print("\n" + "=" * 60)
    print("A) PIPELINE CON CLASIFICADOR (percepción real)")
    print("=" * 60)
    if args.skip_classifier_pipeline and args.classifier_scenario_json:
        scenario_a = json.loads(Path(args.classifier_scenario_json).read_text())
        time_perception_a = float("nan")
        print("  (usando scenario ya guardado, sin recronometrar percepción)")
    else:
        scenario_a, time_perception_a = build_scenario_from_classifier(
            args.video, args.model_path, args.confidence)
        print(f"  Tiempo de percepción (video -> scenario): {time_perception_a:.2f}s")

    kb = KnowledgeBase()
    rc = RobotCapabilities()
    sg = SemanticActionGraph(rc, kb)
    sg.build_full_action_graph(kb.concepts.keys(), rc.actions)
    new_concepts = set(scenario_a.get("detected_objects", [])) - set(kb.concepts.keys())
    if new_concepts:
        for c in new_concepts:
            kb.learn_concept(c)
        sg.build_full_action_graph(new_concepts, rc.actions)

    gnn = GNNCostOptimizer(kb, pretrained_path=args.gnn_checkpoint)
    perceptual_graph_a = build_perceptual_graph(scenario_a["object_positions"])
    gnn.initialize_model(sg.graph, perceptual_graph_a, scenario_a["goal"])

    time_gnn_a, _ = run_gnn_inference(gnn, sg, scenario_a)
    print(f"  Tiempo de inferencia GNN: {time_gnn_a:.4f}s")
    pred_actions_a = [e["action"] for e in scenario_a.get("semantic_line", [])
                      if e.get("action")]

    results["A_pipeline_clasificador"] = {
        "tiempo_percepcion_seg": time_perception_a,
        "tiempo_gnn_seg": time_gnn_a,
        "tiempo_total_seg": (time_perception_a if not np.isnan(time_perception_a) else 0) + time_gnn_a,
        "n_acciones_predichas": len(pred_actions_a),
    }

    if has_gt:
        print("\n" + "=" * 60)
        print("B) GROUND TRUTH (oracle, sin clasificador de visión)")
        print("=" * 60)
        scenario_b, time_scenario_b = build_scenario_from_ground_truth(
            video_id, gt, scenario_a["object_positions"], goal=scenario_a["goal"])
        print(f"  Tiempo de armado del scenario (lectura de CSV): {time_scenario_b:.4f}s")

        time_gnn_b, _ = run_gnn_inference(gnn, sg, scenario_b)
        print(f"  Tiempo de inferencia GNN: {time_gnn_b:.4f}s")
        gt_actions = [e["action"] for e in scenario_b["semantic_line"]]

        results["B_ground_truth_oracle"] = {
            "tiempo_scenario_seg": time_scenario_b,
            "tiempo_gnn_seg": time_gnn_b,
            "tiempo_total_seg": time_scenario_b + time_gnn_b,
            "n_acciones_reales": len(gt_actions),
        }

        acc_a = accuracy_vs_ground_truth(pred_actions_a, gt_actions)
        results["A_pipeline_clasificador"]["accuracy_vs_ground_truth"] = acc_a
        print(f"\n  Accuracy de A) vs ground truth real: {acc_a*100:.1f}%")
    else:
        gt_actions = []

    print("\n" + "=" * 60)
    print("C) VLA")
    print("=" * 60)
    if args.skip_vla:
        print("  Saltado (--skip-vla) -- se corre en otro servidor. "
              "Combina este resultado con el de la rama C) usando "
              "run_paper_comparison.py --merge-with cuando tengas ambos.")
        results["C_vla"] = {"status": "saltado (--skip-vla), correr aparte en otro servidor"}
    else:
        vla_actions, time_vla = run_vla_inference(args.video, scenario_a["goal"])
        if vla_actions is not None:
            acc_vla = accuracy_vs_ground_truth(vla_actions, gt_actions) if has_gt else float("nan")
            results["C_vla"] = {
                "tiempo_total_seg": time_vla,
                "n_acciones_predichas": len(vla_actions),
                "accuracy_vs_ground_truth": acc_vla,
            }
            print(f"  Tiempo VLA: {time_vla:.2f}s | Accuracy vs ground truth: {acc_vla*100:.1f}%")
        else:
            results["C_vla"] = {"status": "no implementado (ver run_vla_inference())"}

    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    print(json.dumps(results, indent=2, default=str))

    Path(args.output).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nGuardado en {args.output}")


if __name__ == "__main__":
    main()