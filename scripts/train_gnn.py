#!/usr/bin/env python3

import os
import sys
from NeoPath.semantic_knowledge import KnowledgeBase
from NeoPath.semantic_graph import SemanticActionGraph
from NeoPath.robot_physical_capacities import RobotCapabilities


def main():
    """
    Construye y guarda el grafo semántico (sin entrenar el GNN).
    Ejecutar antes de usar el robot para verificar que el grafo es correcto.
    """

    print("\n" + "="*60)
    print("CONSTRUCCIÓN DEL GRAFO SEMÁNTICO")
    print("="*60 + "\n")

    package_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(package_path, 'models')
    os.makedirs(models_dir, exist_ok=True)
    graph_json_path = os.path.join(models_dir, 'semantic_graph.json')
    graph_png_path  = os.path.join(models_dir, 'semantic_graph.png')

    # 1. Inicializar componentes
    print("Inicializando componentes...")
    robot_capabilities = RobotCapabilities()
    knowledge_base = KnowledgeBase()
    semantic_graph = SemanticActionGraph(robot_capabilities, knowledge_base)
    semantic_graph.build_full_action_graph(knowledge_base.concepts.keys(), robot_capabilities.actions)
    print("✓ Knowledge base y grafo semántico construidos\n")

    # 2. Estadísticas básicas
    stats = semantic_graph.export_to_json()["statistics"]
    print(f"  Conceptos   : {stats['total_concepts']}")
    print(f"  Acciones    : {stats['total_actions']}")
    print(f"  Permitidas  : {stats['safe_actions']}")
    print(f"  Prohibidas  : {stats['prohibited_actions']}\n")

    # 3. Guardar JSON
    semantic_graph.save_graph(graph_json_path)

    # 4. Visualizar PNG
    semantic_graph.visualize(graph_png_path)

    print("\n" + "="*60)
    print("GRAFO LISTO")
    print("="*60)
    print(f"  JSON : {graph_json_path}")
    print(f"  PNG  : {graph_png_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrumpido")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
