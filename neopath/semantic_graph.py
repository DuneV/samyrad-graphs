# semantic_graph.py

import json
import os
from typing import Dict, Optional, Set, List
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
import numpy as np
import matplotlib
matplotlib.use('Agg')
import imageio
import matplotlib.pyplot as plt
from neopath.semantic_knowledge import KnowledgeBase
from neopath.robot_physical_capacities import RobotCapabilities, RobotAction

@dataclass
class ConceptNode:
    """Nodo de CONCEPTO en el grafo (no instancia)"""
    concept_id: str
    concept_type: str
    tool: bool
    affordances: Set[str]
    contextual_info: str
    safety_level: str
    
    def to_dict(self):
        return {
            "concept_id": self.concept_id,
            "concept_type": self.concept_type,
            "tool": self.tool,
            "affordances": list(self.affordances),
            "contextual_info": self.contextual_info,
            "safety_level": self.safety_level
        }

@dataclass
class ActionEdge:
    """Arista de acción entre CONCEPTOS"""
    action: str
    source_concept: str
    target_concept: str
    preconditions: Dict
    effects: Dict
    cost: float = 1.0
    ethical_weight: float = 1.0
    
    def is_feasible(self, current_state: Dict) -> bool:
        if self.ethical_weight == 0.0:
            return False
        
        for key, value in self.preconditions.items():
            if key == "robot_hand" and value == "holding":
                if current_state.get("robot_hand") == "free":
                    return False
            elif current_state.get(key) != value:
                return False
        return True

class SemanticActionGraph:
    """
    Grafo OBJETO-ACCIÓN-OBJETO
    
    - Nodos = OBJETO/CONCEPTOS (keyboard, no keyboard_0, keyboard_1)
    - Se construye con lista de acciones predeterminadas
    - Affordances se aprenden con Groq
    """
    
    def __init__(self, robot_capabilities: RobotCapabilities, 
                 knowledge_base: KnowledgeBase):
        self.graph = nx.MultiDiGraph()
        self.concept_nodes: Dict[str, ConceptNode] = {}
        self.robot_capabilities = robot_capabilities
        self.knowledge_base = knowledge_base
    
    def add_concept_node(self, concept_type: str):
        """Añade nodo de CONCEPTO al grafo (solo si no existe)"""
        
        if concept_type in self.concept_nodes:
            return  # Ya existe
        
        # Obtener o aprender conocimiento
        if not self.knowledge_base.has_concept(concept_type):
            self.knowledge_base.learn_concept(concept_type)
        
        concept_knowledge = self.knowledge_base.get_concept(concept_type)
        
        # Crear nodo de concepto
        concept_node = ConceptNode(
            concept_id=concept_type,
            concept_type=concept_type,
            tool=concept_knowledge.tool,
            affordances=concept_knowledge.affordances,
            contextual_info=concept_knowledge.contextual_info,
            safety_level=concept_knowledge.safety_level
        )
        
        self.concept_nodes[concept_type] = concept_node
        
        # Añadir al grafo networkx
        self.graph.add_node(
            concept_type,
            type="concept",
            affordances=list(concept_knowledge.affordances),
            contextual_info=concept_knowledge.contextual_info,
            safety_level=concept_knowledge.safety_level
        )
        
        self.knowledge_base.update_usage(concept_type)
    
    def add_action_edge(self, action_edge: ActionEdge):
        """Añade arista de acción entre conceptos"""
        self.graph.add_edge(
            action_edge.source_concept,
            action_edge.target_concept,
            action=action_edge.action,
            cost=action_edge.cost,
            ethical_weight=action_edge.ethical_weight,
            preconditions=action_edge.preconditions,
            effects=action_edge.effects
        )
    
    def _add_special_nodes(self):
        """Añade nodos especiales del robot"""
        special_concepts = ["robot_camera", "robot_left_hand", "robot_right_hand", "robot_hands"]
        
        for concept in special_concepts:
            if concept not in self.concept_nodes:
                self.concept_nodes[concept] = ConceptNode(
                    concept_id=concept,
                    concept_type=concept,
                    tool=False,
                    affordances=set(),
                    contextual_info=f"{concept}",
                    safety_level="safe"
                )
                
                self.graph.add_node(
                    concept,
                    type="robot_component",
                    affordances=[],
                    contextual_info=f"{concept}",
                    safety_level="safe"
                )
    
    def _calculate_ethical_weight(self,
                                source_concept: ConceptNode,
                                target_concept: ConceptNode,
                                action: RobotAction) -> float:
        """
        Strict non-harm ethical rule:
        If the target is a living entity, only SAFE actions are allowed.
        Based on ASIMOV 1st rule for robots.
        """

        if target_concept:
            context = target_concept.contextual_info.lower()
            source_context = source_concept.contextual_info.lower()

            is_living = getattr(target_concept, "living_organism", False)

            if not is_living:
                context = target_concept.contextual_info.lower()
                is_living = any(
                    k in context for k in ["human", "person", "animal", "pet"]
                )
                
            if is_living and action.risk_level != "low":
                return 0.0
        return 1.0

    
    def build_full_action_graph(self, unique_concepts: Set[str], all_actions: List[RobotAction]):
        """
        Construye un grafo completo basado en las preconditions de cada acción.
        Cada acción solo se conectará desde la parte del robot correspondiente
        y solo a objetos que tengan los affordances requeridos.
        """
        self._add_special_nodes()
        # Asegurar que todos los conceptos existan
        for concept_type in unique_concepts:
            if concept_type.startswith("robot"):
                continue
            self.add_concept_node(concept_type)

        all_concepts = list(unique_concepts) + ["robot_camera", "robot_left_hand", "robot_right_hand", "robot_hands"]

        # Identificar objetos que son herramientas
        tool_objects = [c for c in all_concepts
                        if self.knowledge_base.get_concept(c) and self.knowledge_base.get_concept(c).tool]

        for action in all_actions:
        # Revisar preconditions estándar, p.ej. robot_hand, robot_camera, etc.
            for precond_key in action.preconditions.keys():
                if not precond_key.startswith("robot"):
                    continue
                possible_parts = []
                # Si el precond es genérico ("robot_hand"), expandimos a las manos reales
                if precond_key == "robot_hand":
                    possible_parts.append("robot_right_hand")
                    possible_parts.append("robot_left_hand")
                    possible_parts.append("robot_hands")
                else:
                    possible_parts.append(precond_key)

                for robot_part in possible_parts:
                    src_node = self.concept_nodes.get(robot_part)
                    if src_node is None:
                        continue

                    for target in unique_concepts:
                        if target.startswith("robot"):
                            continue

                        tgt_node = self.concept_nodes.get(target)
                        if not tgt_node:
                            continue

                        concept = self.knowledge_base.get_concept(target)
                        if not concept:
                            continue

                        # Verificar affordances requeridos
                        if not action.required_affordances.intersection(tgt_node.affordances):
                            continue

                        # --- Lógica de selección de mano ---
                        if concept.physical_properties.get("weight") != "light":
                            assigned_part = "robot_hands"
                        elif concept.tool:
                            assigned_part = "robot_left_hand"
                        else:
                            assigned_part = "robot_right_hand"

                        # Solo conectar si coincide con la mano actual
                        if robot_part != assigned_part:
                            continue

                        ethical_weight = self._calculate_ethical_weight(
                            src_node, tgt_node, action
                        )

                        self.add_action_edge(ActionEdge(
                            action=action.name,
                            source_concept=robot_part,
                            target_concept=target,
                            preconditions=action.preconditions,
                            effects=action.effects,
                            cost=action.cost,
                            ethical_weight=ethical_weight
                        ))

            robot_parts = [k for k in action.preconditions.keys() if not k.startswith("robot_hand")]
            for robot_part in robot_parts:
                src_node = self.concept_nodes.get(robot_part)
                if src_node is None:
                    continue

                # Posibles targets: todos los conceptos excepto el robot mismo
                for target in all_concepts:
                    if target == robot_part:
                        continue

                    t_node = self.concept_nodes.get(target)
                    if t_node is None:
                        continue

                    # Solo permitir targets que tengan los affordances requeridos
                    if action.required_affordances.intersection(t_node.affordances):
                        ethical_weight = self._calculate_ethical_weight(src_node, t_node, action)

                        self.add_action_edge(ActionEdge(
                            action=action.name,
                            source_concept=robot_part,
                            target_concept=target,
                            preconditions=action.preconditions,
                            effects=action.effects,
                            cost=action.cost,
                            ethical_weight=ethical_weight
                        ))

        # --- Crear edges entre objetos según acciones de herramienta ---
        tool_actions = [
            action for action in all_actions
            if action.preconditions.get("tool") is True
        ]

        for action in tool_actions:
            for src in tool_objects:
                src_node = self.concept_nodes[src]
                concept_src = self.knowledge_base.get_concept(src)

                # Verificar que la herramienta tenga los affordances adecuados
                if action.tool_affordances and not any(a in src_node.affordances for a in action.tool_affordances):
                    continue

                for tgt in all_concepts:
                    if tgt == src or tgt.startswith("robot"):
                        continue

                    tgt_node = self.concept_nodes.get(tgt)
                    if not tgt_node:
                        continue

                    concept_tgt = self.knowledge_base.get_concept(tgt)
                    if not concept_tgt or concept_tgt.safety_level == "dangerous":
                        continue

                    # Verificar que el target tenga los affordances requeridos
                    if not action.target_affordances.intersection(tgt_node.affordances):
                        continue

                    # Calcular el peso ético o costo
                    ethical_weight = self._calculate_ethical_weight(
                        src_node, tgt_node, action
                    )

                    self.add_action_edge(ActionEdge(
                        action=action.name,  # ej: "cut", "pour"
                        source_concept=src,
                        target_concept=tgt,
                        preconditions=action.preconditions,
                        effects=action.effects,
                        cost=action.cost,
                        ethical_weight=ethical_weight
                    ))
                    
        # --- Acciones objeto → objeto: tool = False ---
        object_actions = [
            action for action in all_actions
            if action.preconditions.get("tool") is False
        ]

        for action in object_actions:
            req_aff = action.required_affordances

            for src in unique_concepts:
                if src.startswith("robot"):
                    continue

                src_node = self.concept_nodes.get(src)
                if not src_node:
                    continue

                # Origen debe tener affordances compatibles
                if not req_aff.intersection(src_node.affordances):
                    continue

                for tgt in unique_concepts:
                    if tgt.startswith("robot") or tgt == src:
                        continue

                    tgt_node = self.concept_nodes.get(tgt)
                    if not tgt_node:
                        continue

                    # Destino también debe calzar
                    if not action.target_affordances.intersection(tgt_node.affordances):
                        continue

                    ethical_weight = self._calculate_ethical_weight(src_node, tgt_node, action)

                    self.add_action_edge(ActionEdge(
                        action=action.name,
                        source_concept=src,
                        target_concept=tgt,
                        preconditions=action.preconditions,
                        effects=action.effects,
                        cost=action.cost,
                        ethical_weight=ethical_weight
                    ))

    def visualize(self, filename="semantic_action_graph.png"):
        """Visualiza el grafo semántico con Robot al centro y Objetos alrededor."""
        try:
            plt.figure(figsize=(20, 15)) # Un poco más grande para evitar amontonamiento

            # 1. CLASIFICAR NODOS
            concept_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'concept']
            robot_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'robot_component']

            # 2. DEFINIR LAYOUT PERSONALIZADO (CÍRCULOS CONCÉNTRICOS)
            pos = {}
            
            # A) Robot en el centro (Círculo pequeño o punto central)
            if robot_nodes:
                if len(robot_nodes) == 1:
                    pos[robot_nodes[0]] = np.array([0.0, 0.0])
                else:
                    # Si hay varios componentes del robot, haz un círculo pequeño
                    radius_inner = 1.5 
                    angle_step = 2 * np.pi / len(robot_nodes)
                    for i, node in enumerate(robot_nodes):
                        theta = i * angle_step
                        pos[node] = np.array([radius_inner * np.cos(theta), radius_inner * np.sin(theta)])
            
            # B) Objetos alrededor (Círculo grande)
            if concept_nodes:
                radius_outer = 6.0 # Radio mucho mayor para dar espacio a las aristas
                angle_step = 2 * np.pi / len(concept_nodes)
                for i, node in enumerate(concept_nodes):
                    theta = i * angle_step
                    pos[node] = np.array([radius_outer * np.cos(theta), radius_outer * np.sin(theta)])

            # 3. COLOREAR NODOS (Tu lógica original)
            node_colors = []
            for node in concept_nodes:
                safety = self.graph.nodes[node].get('safety_level', 'safe')
                if safety == 'dangerous': node_colors.append('#ff4d4d') # Rojo más suave
                elif safety == 'caution': node_colors.append('#ffd700') # Dorado
                else: node_colors.append('#87CEFA') # Lightblue

            # 4. DIBUJAR NODOS
            # Nodos Concepto
            nx.draw_networkx_nodes(self.graph, pos, nodelist=concept_nodes, 
                                node_color=node_colors, node_size=2000, 
                                node_shape='o', edgecolors='gray')
            
            # Nodos Robot
            nx.draw_networkx_nodes(self.graph, pos, nodelist=robot_nodes, 
                                node_color='#F08080', node_size=2500, 
                                node_shape='s', edgecolors='darkred')

            # Etiquetas de los nodos
            nx.draw_networkx_labels(self.graph, pos, font_size=9, font_weight='bold')

            # 5. DIBUJAR ARISTAS Y ETIQUETAS (Lógica mejorada)
            ax = plt.gca()
            for (u, v) in self.graph.edges():
                edges_between = list(self.graph.get_edge_data(u, v).items())
                n_edges = len(edges_between)
                
                # Coordenadas de los nodos
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                
                for i, (key, data) in enumerate(edges_between):
                    # Determinar color
                    ew = data.get('ethical_weight', 1.0)
                    color = 'green' if ew >= 0.7 else 'orange' if ew >= 0.3 else 'red'

                    # Calcular curvatura (rad)
                    # Usamos arc3 para curvas simples
                    rad = 0.2 * (i - (n_edges - 1) / 2) 
                    
                    # Dibujar la arista
                    arrow = list(ax.annotate("",
                                xy=(x2, y2), xycoords='data',
                                xytext=(x1, y1), textcoords='data',
                                arrowprops=dict(arrowstyle="-|>", color=color, 
                                            connectionstyle=f"arc3,rad={rad}", 
                                            linewidth=1.5, alpha=0.8)
                                ).arrow_patch.get_path().iter_segments())

                    # Calcular el punto medio de la curva para poner el texto
                    mid_x = (x1 + x2) / 2
                    mid_y = (y1 + y2) / 2
                    
                    # Vector dirección de la arista
                    dx = x2 - x1
                    dy = y2 - y1
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist == 0: dist = 1 # Evitar división por cero
                    
                    # Vector normal (perpendicular) para desplazar el texto según la curvatura 'rad'
                    # Si rad es positivo, desplazamos hacia un lado, si es negativo hacia el otro
                    normal_x = -dy / dist
                    normal_y = dx / dist
                    
                    # El factor 3.0 es para exagerar el desplazamiento del texto y que no pise la linea
                    offset_scale = dist * rad * 0.5 
                    label_x = mid_x + normal_x * offset_scale
                    label_y = mid_y + normal_y * offset_scale

                    label_text = f"{data.get('action')}"
                    
                    # Dibujar texto con FONDO BLANCO (bbox) para legibilidad
                    plt.text(label_x, label_y, label_text, 
                            fontsize=7, color='black',
                            horizontalalignment='center', 
                            verticalalignment='center',
                            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.9))

            # Leyenda y Títulos
            # Crear proxies manuales para la leyenda para que se vea limpio
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#87CEFA', label='Safe Object', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffd700', label='Caution Object', markersize=10),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4d4d', label='Dangerous Object', markersize=10),
                Line2D([0], [0], marker='s', color='w', markerfacecolor='#F08080', label='Robot Component', markersize=10),
            ]
            
            plt.title("Semantic Graph\nObject -> Action -> Object", fontsize=16)
            plt.legend(handles=legend_elements, loc='upper left')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filename, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"✓ PNG of Semantic graph saved to {filename}")

        except Exception as e:
            print(f"Error visualizing graph: {e}")
            import traceback
            traceback.print_exc()

    def get_fixed_layout(self):
        """Calcula las posiciones UNA sola vez para mantener coherencia en la animación."""
        pos = {}
        
        # 1. CLASIFICAR NODOS
        concept_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'concept']
        robot_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'robot_component']

        # 2. DEFINIR LAYOUT (Tu lógica original intacta)
        if robot_nodes:
            if len(robot_nodes) == 1:
                pos[robot_nodes[0]] = np.array([0.0, 0.0])
            else:
                radius_inner = 1.5 
                angle_step = 2 * np.pi / len(robot_nodes)
                for i, node in enumerate(robot_nodes):
                    theta = i * angle_step
                    pos[node] = np.array([radius_inner * np.cos(theta), radius_inner * np.sin(theta)])
        
        if concept_nodes:
            radius_outer = 6.0
            angle_step = 2 * np.pi / len(concept_nodes)
            for i, node in enumerate(concept_nodes):
                theta = i * angle_step
                pos[node] = np.array([radius_outer * np.cos(theta), radius_outer * np.sin(theta)])
        
        return pos, robot_nodes, concept_nodes

    def generate_reasoning_gif(self, reasoning_path, filename="reasoning_animation.gif"):
        """
        Genera un GIF resaltando la secuencia de pasos.
        
        Args:
            reasoning_path: Lista de pasos. Cada paso es una tupla/dict con:
                            {
                                'active_nodes': ['robot_hand', 'cup'], 
                                'active_edge': ('robot_hand', 'cup', 'grasp'),
                                'description': 'Descripción del paso'
                            }
        """
        frames = []
        pos, robot_nodes, concept_nodes = self.get_fixed_layout()
        
        if not os.path.exists("temp_frames"):
            os.makedirs("temp_frames")

        print(f"Generando {len(reasoning_path)} frames para la animación...")

        try:
            for idx, step in enumerate(reasoning_path):
                active_nodes = set(step.get('active_nodes', []))
                active_edge_tuple = step.get('active_edge', None) 

                plt.figure(figsize=(20, 15))
                ax = plt.gca()

                # --- DIBUJAR NODOS ---
                for node_set, shape in [(concept_nodes, 'o'), (robot_nodes, 's')]:
                    for node in node_set:
                        is_active = node in active_nodes
                        
                        d = self.graph.nodes[node]
                        if d.get('type') == 'robot_component':
                            base_color = '#F08080'
                        else:
                            safety = d.get('safety_level', 'safe')
                            base_color = '#ff4d4d' if safety == 'dangerous' else '#ffd700' if safety == 'caution' else '#87CEFA'
                        
                        alpha = 1.0 if is_active else 0.95
                        linewidth = 3.0 if is_active else 1.0
                        edge_color = 'green' if is_active else 'black'
                        size = 2500 if is_active else 2000

                        nx.draw_networkx_nodes(self.graph, pos, nodelist=[node],
                                            node_color=base_color, node_size=size,
                                            node_shape=shape, edgecolors=edge_color,
                                            linewidths=linewidth, alpha=alpha)

                        font_color = 'black'
                        font_weight = 'bold' if is_active else 'normal'
                        nx.draw_networkx_labels(self.graph, pos, labels={node: node}, 
                                              font_color=font_color, font_weight=font_weight, font_size=9)

                # --- DIBUJAR ARISTAS ---
                for (u, v) in self.graph.edges():
                    edges_between = list(self.graph.get_edge_data(u, v).items())
                    n_edges = len(edges_between)
                    x1, y1 = pos[u]
                    x2, y2 = pos[v]

                    for i, (key, data) in enumerate(edges_between):
                        action_name = data.get('action')
                        
                        is_active_edge = False
                        if active_edge_tuple:
                            au, av, a_action = active_edge_tuple
                            if au == u and av == v and a_action == action_name:
                                is_active_edge = True

                        ew = data.get('ethical_weight', 1.0)
                        base_color = 'green' if ew >= 0.7 else 'orange' if ew >= 0.3 else 'red'
                        
                        color = base_color if is_active_edge else 'black'
                        alpha = 1.0 if is_active_edge else 0.90
                        width = 3.0 if is_active_edge else 1.0
                        zorder = 10 if is_active_edge else 1
                        
                        rad = 0.2 * (i - (n_edges - 1) / 2)

                        list(ax.annotate("",
                                    xy=(x2, y2), xycoords='data',
                                    xytext=(x1, y1), textcoords='data',
                                    arrowprops=dict(arrowstyle="-|>", color=color, 
                                                connectionstyle=f"arc3,rad={rad}", 
                                                linewidth=width, alpha=alpha),
                                    zorder=zorder
                                    ).arrow_patch.get_path().iter_segments())
                        
                        mid_x = (x1 + x2) / 2
                        mid_y = (y1 + y2) / 2
                        dx, dy = x2 - x1, y2 - y1
                        dist = np.sqrt(dx*dx + dy*dy) or 1
                        normal_x, normal_y = -dy / dist, dx / dist
                        offset_scale = dist * rad * 0.5 
                        label_x = mid_x + normal_x * offset_scale
                        label_y = mid_y + normal_y * offset_scale

                        text_alpha = 1.0 if is_active_edge else 0.95
                        text_color = 'black' if is_active_edge else 'gray'
                        box_edge = color if is_active_edge else 'black'
                        
                        plt.text(label_x, label_y, action_name, 
                                fontsize=7, color=text_color,
                                horizontalalignment='center', verticalalignment='center',
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=box_edge, alpha=text_alpha),
                                zorder=zorder)

                plt.title(f"Reasoning Step {idx+1}: {step.get('description', '')}", fontsize=16)
                plt.axis('off')
                plt.tight_layout()
                
                frame_name = f"temp_frames/frame_{idx:03d}.png"
                plt.savefig(frame_name, dpi=100)
                frames.append(frame_name)
                plt.close()

            # --- CONSTRUIR GIF ---
            with imageio.get_writer(filename, mode='I', fps=0.33) as writer: # 1.5 segundos por frame
                for filename_png in frames:
                    image = imageio.imread(filename_png)
                    writer.append_data(image)
            
            # Limpieza: Eliminar los frames individuales y la carpeta temporal
            for filename_png in frames:
                os.remove(filename_png)
            os.rmdir("temp_frames")
            
            print(f"✓ GIF guardado en {filename}")

        except Exception as e:
            print(f"Error animando grafo: {e}")
            import traceback
            traceback.print_exc()

    def export_to_json(self):
        """Exporta grafo a JSON"""
        return {
            "nodes": [
                {"id": node, **self.graph.nodes[node]}
                for node in self.graph.nodes()
            ],
            "edges": [
                {
                    "source": u, "target": v,
                    "action": data.get('action'),
                    "cost": data.get('cost'),
                    "ethical_weight": data.get('ethical_weight', 1.0)
                }
                for u, v, data in self.graph.edges(data=True)
            ],
            "statistics": {
                "total_concepts": len([n for n, d in self.graph.nodes(data=True) 
                                      if d.get('type') == 'concept']),
                "total_actions": self.graph.number_of_edges(),
                "safe_actions": len([1 for u, v, d in self.graph.edges(data=True) 
                                    if d.get('ethical_weight', 1.0) >= 0.7]),
                "prohibited_actions": len([1 for u, v, d in self.graph.edges(data=True) 
                                          if d.get('ethical_weight', 1.0) == 0.0])
            }
        }
    
    def save_graph(self, filename: str = "semantic_graph.json"):
        """Guarda grafo semántico"""
        try:
            graph_data = {
                "timestamp": datetime.now().isoformat(),
                "concepts": {k: v.to_dict() for k, v in self.concept_nodes.items()},
                "graph_export": self.export_to_json()
            }
            
            with open(filename, 'w') as f:
                json.dump(graph_data, f, indent=2)
            
            print(f"✓ Semantic graph saved to {filename}")
        except Exception as e:
            print(f"Error saving graph: {e}")