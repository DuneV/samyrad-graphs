#!/usr/bin/env python3
"""
run_full_pipeline.py
=======================
Orquesta los 6 pasos completos. Cada uno se puede saltar con
--skip-stepN si ya corriste ese paso antes.

STEP 1: TRAINING    -- entrena el clasificador de acciones (fine-tuning)
STEP 2: USING        -- clasifica un video de prueba con el modelo Y con ground truth
STEP 3: GENERATION   -- construye la KnowledgeBase desde observaciones reales
STEP 4: TRAINING     -- entrena el GNN con esos escenarios reales
STEP 5: OBTAINING    -- corre el GNN entrenado sobre un video egocéntrico, muestra el grafo
STEP 6: COMPARING    -- VLA vs grafo(clasificador) vs grafo(ground truth), contra ground truth

Uso (todo de una vez):
    python3 run_full_pipeline.py \
        --crops-dirs crops_cache/P01 crops_cache/P02 \
        --gt-csv annotations/EPIC_100_train.csv annotations/EPIC_100_validation.csv \
        --model-path yoloe-11l-seg-pf.pt \
        --test-video kitchen/EPIC-KITCHENS/P37/videos/P37_01.MP4 \
        --classifier-checkpoint action_classifier_finetuned.pth

Uso (saltando pasos ya hechos):
    python3 run_full_pipeline.py --skip-step1 --skip-step4 ...
"""

import argparse
import json
from pathlib import Path


def step1_training(args):
    print("\n" + "=" * 70)
    print("STEP 1: TRAINING (clasificador de acciones)")
    print("=" * 70)
    if args.skip_step1:
        print(f"  Saltado -- usando checkpoint existente: {args.classifier_checkpoint}")
        return args.classifier_checkpoint

    import subprocess
    cmd = ["python3", "train_finetune.py",
           "--cache-dirs", *args.crops_dirs,
           "--output", args.classifier_checkpoint,
           "--epochs", str(args.epochs),
           "--unfrozen-layers", str(args.unfrozen_layers)]
    if args.use_lora:
        cmd += ["--use-lora", "--clip-model", "openai/clip-vit-large-patch14"]
    print(f"  Corriendo: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return args.classifier_checkpoint


def step2_using(args, classifier_checkpoint):
    print("\n" + "=" * 70)
    print("STEP 2: USING (clasificar con el modelo entrenado Y con ground truth)")
    print("=" * 70)

    import torch
    from ultralytics import YOLO
    import mediapipe as mp
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode

    from epic_ground_truth import EpicGroundTruth
    from trained_classifier_inference import TrainedActionClassifier
    from pipeline_step2_using import build_scenarios_dual

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gt = EpicGroundTruth(args.gt_csv)
    classifier = TrainedActionClassifier.load(classifier_checkpoint, device=device)
    yolo = YOLO(args.model_path).to(device)
    hand_options = HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=args.confidence,
        min_tracking_confidence=args.confidence,
    )
    hands = HandLandmarker.create_from_options(hand_options)

    import re
    video_id_match = re.search(r"(P\d{2}_\d{2,3})", Path(args.test_video).stem)
    video_id = video_id_match.group(1) if video_id_match else None
    if video_id is None or not gt.has_video(video_id):
        raise ValueError(f"El video de prueba no tiene ground truth: {args.test_video}")

    scenario_classifier, scenario_ground_truth = build_scenarios_dual(
        args.test_video, video_id, gt, classifier, yolo, hands, args.confidence)

    Path("scenario_classifier.json").write_text(json.dumps({
        **scenario_classifier,
        "edge_labels": {str(k): v for k, v in scenario_classifier["edge_labels"].items()},
    }, indent=2))
    Path("scenario_ground_truth.json").write_text(json.dumps({
        **scenario_ground_truth,
        "edge_labels": {str(k): v for k, v in scenario_ground_truth["edge_labels"].items()},
    }, indent=2))

    print(f"  {len(scenario_classifier['semantic_line'])} eventos (clasificador)")
    print(f"  {len(scenario_ground_truth['semantic_line'])} eventos (ground truth)")
    print(f"  Guardados: scenario_classifier.json, scenario_ground_truth.json")
    return scenario_classifier, scenario_ground_truth


def step3_generation(args, scenario_files):
    print("\n" + "=" * 70)
    print("STEP 3: GENERATION (KnowledgeBase desde observaciones reales)")
    print("=" * 70)
    if args.skip_step3:
        print(f"  Saltado -- usando {args.knowledge_base_path} existente")
        return args.knowledge_base_path

    import subprocess
    cmd = ["python3", "build_knowledge_base_from_scenarios.py",
           "--scenarios", *scenario_files,
           "--knowledge-base-path", args.knowledge_base_path,
           "--robot-actions-path", args.robot_actions_path]
    print(f"  Corriendo: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return args.knowledge_base_path


def step4_training(args, scenario_files, knowledge_base_path):
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING (GNN)")
    print("=" * 70)
    if args.skip_step4:
        print(f"  Saltado -- usando checkpoint existente: {args.gnn_checkpoint}")
        return args.gnn_checkpoint

    import subprocess
    cmd = ["python3", "train_gnn_from_scenarios.py",
           "--scenarios", *scenario_files,
           "--save-path", args.gnn_checkpoint,
           "--epochs", str(args.gnn_epochs)]
    print(f"  Corriendo: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    return args.gnn_checkpoint


def step5_obtaining(args, gnn_checkpoint, scenario_classifier):
    print("\n" + "=" * 70)
    print("STEP 5: OBTAINING (grafo entrenado sobre un video egocéntrico)")
    print("=" * 70)

    from neopath.semantic_knowledge import KnowledgeBase
    from neopath.robot_physical_capacities import RobotCapabilities
    from neopath.semantic_graph import SemanticActionGraph
    from neopath.gnn import GNNCostOptimizer
    from benchmark_comparison import build_perceptual_graph, run_gnn_inference

    kb = KnowledgeBase(storage_path=args.knowledge_base_path)
    rc = RobotCapabilities(actions_file=args.robot_actions_path)
    sg = SemanticActionGraph(rc, kb)
    sg.build_full_action_graph(kb.concepts.keys(), rc.actions)
    gnn = GNNCostOptimizer(kb, pretrained_path=gnn_checkpoint)

    perceptual_graph = build_perceptual_graph(scenario_classifier["object_positions"])
    gnn.initialize_model(sg.graph, perceptual_graph, scenario_classifier["goal"])
    elapsed, optimized_graph = run_gnn_inference(gnn, sg, scenario_classifier)

    print(f"  Grafo optimizado: {optimized_graph.number_of_nodes()} nodos, "
          f"{optimized_graph.number_of_edges()} aristas")
    print(f"  Tiempo de inferencia: {elapsed:.4f}s")
    print("  Aristas de menor costo (más probables en el plan):")
    edges_sorted = sorted(
        optimized_graph.edges(data=True), key=lambda e: e[2].get("adjusted_cost", 999))
    for u, v, data in edges_sorted[:5]:
        print(f"    {u} --({data.get('action','?')})--> {v}  "
              f"costo={data.get('adjusted_cost', '?'):.3f}")
    return optimized_graph


def step6_comparing(args, gnn_checkpoint):
    print("\n" + "=" * 70)
    print("STEP 6: COMPARING (VLA vs grafo-clasificador vs grafo-ground-truth)")
    print("=" * 70)
    if args.skip_step6:
        print("  Saltado (--skip-step6)")
        return

    import subprocess
    cmd = ["python3", "benchmark_comparison.py",
           "--video", args.test_video,
           "--gt-csv", *args.gt_csv,
           "--model-path", args.model_path,
           "--gnn-checkpoint", gnn_checkpoint,
           "--output", "benchmark_results.json",
           # Reutiliza el scenario ya generado en STEP 2 (pipeline liviano,
           # TrainedActionClassifier) en vez de reprocesar el video entero
           # con StoryTelling -- ese es el que se comía toda la RAM.
           "--skip-classifier-pipeline",
           "--classifier-scenario-json", "scenario_classifier.json"]
    if args.skip_vla:
        cmd.append("--skip-vla")
        print("  VLA desactivado (--skip-vla) -- se corre aparte en otro servidor")
    print("  Reutilizando scenario_classifier.json de STEP 2 "
          "(no reprocesa el video con StoryTelling)")
    print(f"  Corriendo: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("  Resultados en benchmark_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops-dirs", nargs="+", required=True)
    parser.add_argument("--gt-csv", nargs="+", required=True)
    parser.add_argument("--model-path", default="yoloe-11l-seg-pf.pt")
    parser.add_argument("--robot-actions-path", default="robot_actions.json")
    parser.add_argument("--knowledge-base-path", default="knowledge_base.json")
    parser.add_argument("--classifier-checkpoint", default="action_classifier_finetuned.pth")
    parser.add_argument("--gnn-checkpoint", default="gnn_cost_optimizer.pth")
    parser.add_argument("--test-video", required=True)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--gnn-epochs", type=int, default=200)
    parser.add_argument("--unfrozen-layers", type=int, default=8)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--skip-step1", action="store_true")
    parser.add_argument("--skip-step3", action="store_true")
    parser.add_argument("--skip-step4", action="store_true")
    parser.add_argument("--skip-step6", action="store_true")
    parser.add_argument("--skip-vla", action="store_true",
                        help="No corre la rama VLA en STEP 6 -- se corre "
                             "aparte en otro servidor con GPU distinta.")
    args = parser.parse_args()

    classifier_checkpoint = step1_training(args)
    scenario_classifier, scenario_ground_truth = step2_using(args, classifier_checkpoint)
    scenario_files = ["scenario_classifier.json", "scenario_ground_truth.json"]
    knowledge_base_path = step3_generation(args, scenario_files)
    gnn_checkpoint = step4_training(args, scenario_files, knowledge_base_path)
    step5_obtaining(args, gnn_checkpoint, scenario_classifier)
    step6_comparing(args, gnn_checkpoint)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETO")
    print("=" * 70)


if __name__ == "__main__":
    main()