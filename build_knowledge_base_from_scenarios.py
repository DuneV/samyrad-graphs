#!/usr/bin/env python3
"""
build_knowledge_base_from_scenarios.py
=========================================
Recorre tus escenarios REALES (los que produce tu pipeline de
clasificación de acciones, no scripts/training_data.py sintético),
cuenta con qué acciones se observó cada objeto, y llena la
KnowledgeBase usando learn_concept_from_observations() -- afordancias
derivadas de datos reales, no el placeholder genérico {"inspectable"}
que usaba learn_concept() sin Groq configurado.

Uso:
    python3 build_knowledge_base_from_scenarios.py \
        --scenarios output/*.json \
        --knowledge-base-path knowledge_base.json \
        --robot-actions-path robot_actions.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from neopath.semantic_knowledge import KnowledgeBase
from neopath.robot_physical_capacities import RobotCapabilities


def load_scenarios(paths):
    all_scenarios = []
    for path in paths:
        data = json.loads(Path(path).read_text())
        scenarios = data if isinstance(data, list) else [data]
        all_scenarios.extend(scenarios)
    return all_scenarios


def count_observed_actions(scenarios):
    """
    Cuenta, por objeto, cuántas veces se observó cada acción con él --
    a partir de semantic_line (si está presente, el detalle frame a
    frame real) o si no, de edge_labels como respaldo.
    """
    counts = defaultdict(lambda: defaultdict(int))

    for scenario in scenarios:
        semantic_line = scenario.get("semantic_line", [])
        if semantic_line:
            for event in semantic_line:
                obj = event.get("object")
                action = event.get("action")
                if obj and action:
                    counts[obj][action] += 1
        else:
            # Respaldo: usa edge_labels si no hay semantic_line
            for key in scenario.get("edge_labels", {}):
                if isinstance(key, str):
                    inner = key.strip("()").replace("'", "").split(", ")
                    if len(inner) != 3:
                        continue
                    _, action, obj = inner
                else:
                    _, action, obj = key
                counts[obj][action] += 1

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", required=True)
    parser.add_argument("--knowledge-base-path", default="knowledge_base.json")
    parser.add_argument("--robot-actions-path", default="robot_actions.json")
    parser.add_argument("--min-observations", type=int, default=1,
                        help="Mínimo de observaciones para aprender un "
                             "concepto (objetos vistos muy pocas veces "
                             "se saltan)")
    args = parser.parse_args()

    print("Cargando escenarios...")
    scenarios = load_scenarios(args.scenarios)
    print(f"  {len(scenarios)} escenarios cargados")

    action_counts = count_observed_actions(scenarios)
    print(f"  {len(action_counts)} objetos distintos con acciones observadas")

    kb = KnowledgeBase(storage_path=args.knowledge_base_path)
    rc = RobotCapabilities(actions_file=args.robot_actions_path)
    print(f"  KnowledgeBase con {len(kb.concepts)} conceptos existentes")
    print(f"  RobotCapabilities con {len(rc.actions)} acciones definidas\n")

    n_learned, n_skipped_existing, n_skipped_few = 0, 0, 0
    for obj, actions in sorted(action_counts.items(), key=lambda x: -sum(x[1].values())):
        total_obs = sum(actions.values())
        if obj in kb.concepts:
            n_skipped_existing += 1
            continue
        if total_obs < args.min_observations:
            n_skipped_few += 1
            continue

        kb.learn_concept_from_observations(obj, dict(actions), rc)
        n_learned += 1

    print(f"\n{'='*60}")
    print(f"Conceptos aprendidos de observaciones reales: {n_learned}")
    print(f"Ya existían en la base: {n_skipped_existing}")
    print(f"Con muy pocas observaciones (<{args.min_observations}): {n_skipped_few}")
    print(f"Total de conceptos en la base ahora: {len(kb.concepts)}")
    print(f"Guardado en: {args.knowledge_base_path}")


if __name__ == "__main__":
    main()