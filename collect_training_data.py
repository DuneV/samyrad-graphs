#!/usr/bin/env python3
"""
    python3 collect_training_data.py \
      --videos-dir ./videos/kitchen/ \
      --info-path ./canonical.json \
      --output output/mis_escenarios.json

  # Solo procesar videos - JSON
  python3 collect_training_data.py \
    --videos vid1.mp4 vid2.mp4 \
    --model-path yoloe-11l-seg-pf.pt \
    --info-path canonical.json \
    --output mis_escenarios.json

  # Cargar JSON ya procesados + entrenar
  python3 collect_training_data.py \
    --load-json mis_escenarios.json \
    --train \
    --save-path gnn_cost_optimizer.pth

  # Todo en uno
  python3 collect_training_data.py \
    --videos vid1.mp4 \
    --train \
    --save-path gnn_cost_optimizer.pth
"""

import argparse
import json
import numpy as np
from pathlib import Path

VALID_ROBOT_ACTIONS = {
    "grasp", "move_to", "cut", "inspect", "search_for",
    "open", "close", "push", "pull", "press", "rotate",
    "pour", "remove_from", "greet",
}


def validate_scenario(scenario: dict) -> dict:
    """
    Valida y limpia un escenario del UnifiedPipeline:
    - Filtra edge_labels con acciones no reconocidas
    - Garantiza que required_actions solo contenga acciones válidas
    - Asegura que object_positions tenga al menos los target_objects
    """
    valid_edge_labels = {}
    for k, v in scenario.get("edge_labels", {}).items():
        src, action, tgt = k
        if action in VALID_ROBOT_ACTIONS:
            valid_edge_labels[k] = v

    scenario["edge_labels"] = valid_edge_labels

    scenario["required_actions"] = [
        a for a in scenario.get("required_actions", [])
        if a in VALID_ROBOT_ACTIONS
    ]

    if not scenario["required_actions"] and valid_edge_labels:
        from collections import Counter
        action_counts = Counter(k[1] for k in valid_edge_labels)
        scenario["required_actions"] = [action_counts.most_common(1)[0][0]]

    return scenario


def scenario_from_video(video_path: str, run_name: str,
                         model_path: str, info_path: str,
                         confidence: float = 0.5,
                         N: int = 5) -> dict:
    """
    Procesa un video y devuelve un dict compatible con train_supervised().
    """
    from videoAnalyzer import VideoAnalyzer
    from storyTelling import StoryTelling
    from unifiedPipeline import UnifiedPipeline

    va = VideoAnalyzer(
        video       = video_path,
        path        = video_path,
        output_path = f"output_{run_name}.mp4",
        model_path  = model_path,
        confidence  = confidence,
        info_path   = info_path,
        alpha       = 0.5,
        N           = N,
        run_name    = run_name,
    )
    st = StoryTelling(
        video       = video_path,
        path        = video_path,
        output_path = f"output_{run_name}.mp4",
        model_path  = model_path,
        confidence  = confidence,
        N           = N,
        run_name    = run_name,
        vlm_backend = "clip",
    )

    pipeline = UnifiedPipeline(va, st)
    pipeline.run()

    scenario = pipeline.generate_scenario(goal=None)
    scenario = validate_scenario(scenario)
    pipeline.save(scenario)
    return scenario


def load_scenarios_from_json(path: str) -> list:
    """Carga escenarios guardados previamente y restaura claves tuple."""
    data = json.loads(Path(path).read_text())
    for s in data:
        if "edge_labels" in s:
            fixed = {}
            for k, v in s["edge_labels"].items():
                inner = k.strip("()").split(", ")
                if len(inner) == 3:
                    fixed[tuple(inner)] = v
                else:
                    fixed[k] = v
            s["edge_labels"] = fixed
        validate_scenario(s)
    return data


def save_scenarios_to_json(scenarios: list, path: str) -> None:
    """Guarda escenarios serializando claves tuple a strings."""
    serializable = []
    for s in scenarios:
        s2 = dict(s)
        s2["edge_labels"] = {str(k): v for k, v in s["edge_labels"].items()}
        serializable.append(s2)
    Path(path).write_text(json.dumps(serializable, indent=2))
    print(f"Guardado {len(serializable)} escenarios en {path}")


def print_scenario_summary(scenario: dict) -> None:
    print(f"  goal       : {scenario['goal']}")
    print(f"  objects    : {scenario['detected_objects']}")
    print(f"  actions    : {scenario['required_actions']}")
    print(f"  edge_labels: {len(scenario['edge_labels'])} aristas")
    if scenario['edge_labels']:
        sample = list(scenario['edge_labels'].items())[:3]
        for k, v in sample:
            print(f"    {k} → {v:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Recolectar datos de videos y entrenar GNN')
    parser.add_argument('--videos',      nargs='+', default=[])
    parser.add_argument('--videos-dir',  default=None,
                        help='Carpeta con videos (.mp4/.avi/.mov/.mkv)')
    parser.add_argument('--load-json',   default=None)
    parser.add_argument('--output',      default='extra_scenarios.json')
    parser.add_argument('--model-path',  default='yoloe-11l-seg-pf.pt')
    parser.add_argument('--info-path',   default='canonical.json')
    parser.add_argument('--confidence',  type=float, default=0.5)
    parser.add_argument('--N',           type=int,   default=5)
    parser.add_argument('--train',       action='store_true')
    parser.add_argument('--save-path',   default='gnn_cost_optimizer.pth')
    parser.add_argument('--epochs',      type=int,   default=300)
    parser.add_argument('--no-synthetic',action='store_true',
                        help='No mezclar con training_data.py sintético')
    args = parser.parse_args()

    VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    video_list = list(args.videos)

    if args.videos_dir:
        folder = Path(args.videos_dir)
        if not folder.is_dir():
            print(f"No existe la carpeta: {args.videos_dir}")
        else:
            found = sorted(
                p for p in folder.iterdir()
                if p.suffix.lower() in VIDEO_EXTS
            )
            print(f"Carpeta {args.videos_dir}: {len(found)} videos encontrados")
            for p in found:
                print(f"  {p.name}")
            video_list.extend(str(p) for p in found)

    real_scenarios = []

    if args.load_json:
        print(f"Cargando escenarios desde {args.load_json}...")
        real_scenarios = load_scenarios_from_json(args.load_json)
        print(f"  {len(real_scenarios)} escenarios cargados")

    for i, video_path in enumerate(video_list):
        print(f"\nProcesando video {i+1}/{len(video_list)}: {video_path}")
        run_name = Path(video_path).stem
        try:
            scenario = scenario_from_video(
                video_path = video_path,
                run_name   = run_name,
                model_path = args.model_path,
                info_path  = args.info_path,
                confidence = args.confidence,
                N          = args.N,
            )
            real_scenarios.append(scenario)
            print(f"  ✓ Escenario extraído:")
            print_scenario_summary(scenario)
        except Exception as e:
            print(f"  ✗ Error: {e}")

    if real_scenarios:
        save_scenarios_to_json(real_scenarios, args.output)

    if args.train:
        from neopath.semantic_knowledge import KnowledgeBase
        from neopath.robot_physical_capacities import RobotCapabilities
        from neopath.semantic_graph import SemanticActionGraph
        from neopath.gnn import GNNCostOptimizer, GNNTrainer

        print("\n" + "="*60)
        print("ENTRENAMIENTO DEL GNN")
        print("="*60)

        synthetic = []
        if not args.no_synthetic:
            from training_data import get_all_training_data
            synthetic = get_all_training_data()

        print(f"\nDatos sintéticos : {len(synthetic)}")
        print(f"Datos reales     : {len(real_scenarios)}")

        all_data = synthetic + real_scenarios
        print(f"Total            : {len(all_data)}")

        if not all_data:
            print("Sin datos para entrenar")
            return

        robot_capabilities = RobotCapabilities()
        knowledge_base     = KnowledgeBase()
        semantic_graph     = SemanticActionGraph(robot_capabilities, knowledge_base)
        semantic_graph.build_full_action_graph(
            knowledge_base.concepts.keys(),
            robot_capabilities.actions
        )

        known_concepts = set(knowledge_base.concepts.keys())
        real_concepts  = set()
        for s in real_scenarios:
            real_concepts.update(s.get('detected_objects', []))
        new_concepts = real_concepts - known_concepts
        if new_concepts:
            print(f"\nConceptos nuevos en videos: {new_concepts}")
            for concept in new_concepts:
                knowledge_base.learn_concept(concept)
            semantic_graph.build_full_action_graph(
                new_concepts, robot_capabilities.actions)

        # Entrenar
        gnn_optimizer = GNNCostOptimizer(knowledge_base)
        trainer       = GNNTrainer(gnn_optimizer, semantic_graph, knowledge_base)

        trainer.train_supervised(
            labeled_examples        = all_data,
            epochs                  = args.epochs,
            validation_split        = 0.2,
            early_stopping_patience = 30,
            save_path               = args.save_path,
        )
        print(f"\n✓ Modelo guardado en {args.save_path}")


if __name__ == '__main__':
    main()