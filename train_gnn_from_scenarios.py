#!/usr/bin/env python3
"""
train_gnn_from_scenarios.py
==============================
Entrena el GNNCostOptimizer (neopath/gnn.py) con escenarios reales —
los mismos que produce tu pipeline (UnifiedPipeline.generate_scenario()
o process_chunk.py), en formato:

    {
        "goal": str,
        "target_objects": [...],
        "required_actions": [...],
        "detected_objects": [...],
        "object_positions": {obj: [x,y,z], ...},
        "edge_labels": {"(src, action, tgt)": peso, ...}
    }

Uso:
    python3 train_gnn_from_scenarios.py \
        --scenarios output/p01_scenarios.json output/p02_scenarios.json \
        --save-path gnn_cost_optimizer.pth \
        --epochs 200
"""

import argparse
import json
from pathlib import Path

from neopath.semantic_knowledge import KnowledgeBase
from neopath.robot_physical_capacities import RobotCapabilities
from neopath.semantic_graph import SemanticActionGraph
from neopath.gnn import GNNCostOptimizer, GNNTrainer


def load_scenarios(paths):
    """Carga escenarios desde uno o varios .json, reconstruyendo las
    claves tuple de edge_labels (igual que load_scenarios_from_json en
    collect_training_data.py)."""
    all_scenarios = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        scenarios = data if isinstance(data, list) else [data]
        for s in scenarios:
            if "edge_labels" in s:
                fixed = {}
                for k, v in s["edge_labels"].items():
                    if isinstance(k, str):
                        inner = k.strip("()").replace("'", "").split(", ")
                        if len(inner) == 3:
                            fixed[tuple(inner)] = v
                            continue
                    fixed[k] = v
                s["edge_labels"] = fixed
            all_scenarios.append(s)
    return all_scenarios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", required=True,
                        help="Uno o varios .json de escenarios (salida de "
                             "tu pipeline: generate_scenario / chunked_pipeline)")
    parser.add_argument("--save-path", default="gnn_cost_optimizer.pth")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    args = parser.parse_args()

    print("Cargando escenarios...")
    scenarios = load_scenarios(args.scenarios)
    print(f"  {len(scenarios)} escenarios cargados")

    if not scenarios:
        print("No hay escenarios para entrenar.")
        return

    print("\nInicializando componentes de neopath...")
    robot_capabilities = RobotCapabilities()
    knowledge_base = KnowledgeBase()
    semantic_graph = SemanticActionGraph(robot_capabilities, knowledge_base)
    semantic_graph.build_full_action_graph(
        knowledge_base.concepts.keys(), robot_capabilities.actions)

    # Aprende conceptos nuevos que aparezcan en los escenarios pero no
    # estén en la knowledge base (igual que hacía collect_training_data.py)
    known_concepts = set(knowledge_base.concepts.keys())
    real_concepts = set()
    for s in scenarios:
        real_concepts.update(s.get("detected_objects", []))
    new_concepts = real_concepts - known_concepts
    if new_concepts:
        print(f"Conceptos nuevos detectados en los escenarios: {sorted(new_concepts)}")
        for concept in new_concepts:
            knowledge_base.learn_concept(concept)
        semantic_graph.build_full_action_graph(new_concepts, robot_capabilities.actions)

    gnn_optimizer = GNNCostOptimizer(knowledge_base)
    trainer = GNNTrainer(gnn_optimizer, semantic_graph, knowledge_base)

    trainer.train_supervised(
        labeled_examples=scenarios,
        epochs=args.epochs,
        validation_split=args.validation_split,
        early_stopping_patience=args.early_stopping_patience,
        save_path=args.save_path,
    )

    print(f"\n✓ GNN entrenado y guardado en {args.save_path}")


if __name__ == "__main__":
    main()