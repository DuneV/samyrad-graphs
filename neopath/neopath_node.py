# neopath_node.py

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
import json
from typing import Dict, List, Tuple
import requests
import time
import os
import numpy as np
import random
from dotenv import load_dotenv
import networkx as nx
import threading
from neopath.semantic_knowledge import KnowledgeBase
from neopath.perceptual_knowledge import PerceptualGraph
from neopath.robot_physical_capacities import RobotCapabilities
from neopath.semantic_graph import SemanticActionGraph
from neopath.reasoner import GoalGenerator
from neopath.gnn import GNNCostOptimizer


epsilon_object = 0.3   # exploración de objetos
epsilon_action = 0.2   # exploración de acciones
# self._epsilon_decay    = 0.95



_possible_env_paths = [
    os.path.join(os.getcwd(), '.env'),
    os.path.expanduser('~/jetcobot_remote_ws/.env'),
    os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '..', '.env'),
]

for _p in _possible_env_paths:
    _p = os.path.abspath(_p)
    if os.path.exists(_p):
        load_dotenv(_p)         
        print(f"✓ .env loaded from: {_p}")
        break
else:
    print(".env not found in any expected location")

class GroqClient:
    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
    
    def chat_completion(self, messages: List[Dict], temperature: float = 0.3,
                       json_mode: bool = True) -> str:
        url = f"{self.base_url}/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            raise Exception(f"Groq request failed: {e}")

# ============================================================================
# NODO ROS2 PRINCIPAL
# ============================================================================

class GraphAutonomousNode(Node):
    """Nodo ROS2 con grafo semántico de conceptos + percepción en tiempo real"""
    
    def __init__(self, use_gnn: bool = True):
        super().__init__('neopath_node')
        # Groq
        groq_api_key = os.environ.get('GROQ_API_KEY', '')
        self.groq = GroqClient(groq_api_key) if groq_api_key else None

        if self.groq:
            self.get_logger().info("✓ Groq available for learning concepts")
        else:
            self.get_logger().info("⚠ Groq not available - using predefined concepts")

        #Objetivos
        self.goal_generator = GoalGenerator(groq_client=self.groq)

        # Capacidades y conocimiento
        self.robot_capabilities = RobotCapabilities()
        self.knowledge_base = KnowledgeBase(groq_client=self.groq)
        self.depth_image = None  # Última imagen de profundidad
        
        # Grafos
        self.semantic_graph = SemanticActionGraph(
            self.robot_capabilities,
            self.knowledge_base
        )
        self._pending_plan = []
        self._robot_state = "IDLE"  # IDLE, EXECUTING, WAITING_FEEDBACK
        self._executed_actions = []  # historial de acciones ejecutadas
        self._max_plan_steps = 5  
        
        # ── Debug: simular ejecución de acciones ─────────────────────────────
        
        self._action_timeout       = 8.0   # segundos por acción
        self._action_start_time    = 0.0
        self._current_executing_step = None
        self.debug_timer = self.create_timer(1.0, self._debug_action_executor)

        self.perceptual_graph = PerceptualGraph(
            near_threshold = 200,
            far_threshold = 500,
            vertical_threshold = 50,
            horizontal_threshold=100,
            depth_threshold = 0.5,
        )
        
        all_concepts = set(self.knowledge_base.concepts.keys())

        self.semantic_graph.build_full_action_graph(
            all_concepts,
            self.robot_capabilities.actions
        )

        self.use_gnn = use_gnn
        if self.use_gnn:
            try:
                self.gnn_optimizer = GNNCostOptimizer(self.knowledge_base)
                
                # Cargar modelo pre-entrenado si existe
                model_path = os.path.join(
                    os.path.dirname(__file__), 
                    '..', 'models', 'gnn_cost_optimizer.pth'
                )
                if os.path.exists(model_path):
                    self.gnn_optimizer.load_model(model_path)
                    self.get_logger().info(f"✓ GNN model loaded from {model_path}")
                else:
                    self.get_logger().warn(f"⚠ No pre-trained model found at {model_path}")
                    self.get_logger().info("  Run 'python3 train_gnn.py' to train the model first")
                    self.get_logger().info("  GNN will use random initialization (NOT RECOMMENDED)")
                
                self.get_logger().info("✓ GNN Cost Optimizer initialized")
            except Exception as e:
                self.get_logger().error(f"Failed to initialize GNN: {e}")
                self.use_gnn = False
                self.gnn_optimizer = None
        else:
            self.gnn_optimizer = None
        
        # Estado
        self.current_state = {"robot_hand": "free"}
        self.robot_position = (320.0, 240.0, 0.0)  # Centro de la imagen por defecto
        self.plan_logged = False
        self.user_goal = None
        self.current_goal = None
        self.conversation_started = False
        self.full_plan = None
        
        # Declarar parametros ROS
        # self.declare_parameter('detection_topic', '/yolo/detection_results')
        # Cambio de yolo + 3d
        self.declare_parameter('detection_depth', '/pose_estimator/scene_graph')

        self.declare_parameter('depth_topic', '/camera/depth/image_raw')
        self.declare_parameter('action_feedback_topic', '/robot/action_feedback')
        self.declare_parameter('results_topic', '/robot/autonomous_action')
        self.declare_parameter('user_topic', '/user_input')
        self.declare_parameter('mode', 'auto')

        # Obtener parametros ROS
        # detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        # Obtener informacioń 3d
        detectiond_topic = self.get_parameter('detection_depth').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        action_feedback_topic = self.get_parameter('action_feedback_topic').get_parameter_value().string_value
        results_topic = self.get_parameter('results_topic').get_parameter_value().string_value
        user_topic = self.get_parameter('user_topic').get_parameter_value().string_value
        self.mode = self.get_parameter('mode').get_parameter_value().string_value
        
        # Subscribers y Publishers
        # self.yolo_sub = self.create_subscription(
        #     String, 
        #     detection_topic,
        #     self.yolo_callback, 
        #     100)
        self.scene_sub = self.create_subscription(
            String,
            '/pose_estimator/scene_graph',
            self.scene_callback,
            10
        )
        self.perceptual_pub = self.create_publisher(String, '/yolo/detection_results', 10)
        # self.depth_sub = self.create_subscription(
        #     Image,
        #     depth_topic,
        #     self.depth_callback,
        #     100)
        self.feedback_sub = self.create_subscription(
            String,
            action_feedback_topic,
            self.action_feedback_callback,
            100
        )
        self.results_pub = self.create_publisher(
            String,
            results_topic,
            100
        )
        self.user_sub = self.create_subscription(
            String,
            user_topic,
            self.user_goal_callback,
            100
        )
        
        # Timers
        self.save_timer = self.create_timer(30.0, self.periodic_save)
        
        # Log
        self.get_logger().info("=" * 70)
        self.get_logger().info("GRAPH SYSTEM INITIALIZED")
        self.get_logger().info("=" * 70)
        self.get_logger().info(f"  Known concepts: {len(self.knowledge_base.concepts)}")
        self.get_logger().info(f"  Robot actions: {len(self.robot_capabilities.actions)}")
        self.get_logger().info(f"  GNN Optimization: {'ENABLED' if self.use_gnn else 'DISABLED'}")
        self.get_logger().info("=" * 70)

    def user_goal_callback(self, msg):
        """Recibe objetivos del usuario via topic"""
        self.user_goal = msg.data
        self.current_goal = None  # Resetear para procesar nuevo objetivo
        self.plan_logged = False
        self.conversation_started = False
        self.get_logger().info(f"Objetivo del usuario: {self.user_goal}")
    
    def action_feedback_callback(self, msg):
        try:
            feedback = json.loads(msg.data)
            status = feedback.get('status')
            
            if status == "completed":
                self._executed_actions.append(feedback)
                self.get_logger().info(f"✓ Acción completada: {feedback}")
                # ── Penalizar arista ejecutada ────────────────────────────────
                action_name = feedback.get('action')
                source = feedback.get('source', '').rsplit('_', 1)[0] if '_' in feedback.get('source', '') else feedback.get('source', '')
                target = feedback.get('target', '').rsplit('_', 1)[0] if '_' in feedback.get('target', '') else feedback.get('target', '')

                if action_name and source and target:
                    G = self.semantic_graph.graph
                    if G.has_edge(source, target):
                        for key, data in G[source][target].items():
                            if data.get('action') == action_name:
                                old_cost = data.get('cost', 1.0)
                                new_cost = old_cost + 5.0
                                G[source][target][key]['cost'] = new_cost
                                G[source][target][key]['adjusted_cost'] = new_cost
                                self.get_logger().info(
                                    f"[policy] Penalizando {source}→{action_name}→{target}: "
                                    f"{old_cost:.1f} → {new_cost:.1f}")
                                
                if hasattr(self, '_pending_plan') and self._pending_plan:
                    # Ejecutar siguiente paso del plan
                    next_step = self._pending_plan.pop(0)
                    self._current_executing_step = next_step 
                    self._action_start_time      = time.time() 
                    
                    action_msg = {
                        "action": next_step[1],
                        "source": next_step[0],
                        "target": next_step[2],
                        "remaining_steps": len(self._pending_plan)
                    }
                    plan_msg = String()
                    plan_msg.data = json.dumps(action_msg)
                    self.results_pub.publish(plan_msg)
                    
                    self.get_logger().info(
                        f"[policy] Siguiente paso: "
                        f"{next_step[0]} → {next_step[1]} → {next_step[2]}"
                    )
                else:
                    # Plan completo → replanificar con nueva percepción
                    self.get_logger().info("[policy] Plan completo → replanificando")
                    self._robot_state = "IDLE"
                    self.current_goal = None
                    self.plan_logged = False
                    self._executed_actions = []
                    self._current_executing_step = None           # ← agregar al completar
                    self._robot_state = "IDLE"

            elif status == "failed":
                self.get_logger().warn(f"✗ Acción fallida: {feedback}")
                # Replanificar desde el estado actual
                self._robot_state = "IDLE"
                self.current_goal = None
                self.plan_logged = False
                self._pending_plan = []
        
        except Exception as e:
            self.get_logger().error(f"Feedback error: {e}")


    def _debug_action_executor(self):
        """Simula ejecución con tiempo real — solo para debugging."""
        if self._robot_state != "EXECUTING":
            return
        if self._current_executing_step is None:
            return

        elapsed   = time.time() - self._action_start_time
        remaining = self._action_timeout - elapsed

        self.get_logger().info(
            f"[EXECUTING] {self._current_executing_step[0]} → "
            f"{self._current_executing_step[1]} → "
            f"{self._current_executing_step[2]} | "
            f"{elapsed:.1f}s / {self._action_timeout:.1f}s",
            throttle_duration_sec=2.0
        )

        if elapsed < self._action_timeout:
            return

        # ── Acción completada ─────────────────────────────────────────────
        self.get_logger().info(
            f"[EXECUTING]  Completado: {self._current_executing_step}"
        )
        feedback = {
            "status": "completed",
            "action": self._current_executing_step[1],
            "source": self._current_executing_step[0],
            "target": self._current_executing_step[2],
        }
        msg = String()
        msg.data = json.dumps(feedback)
        self.action_feedback_callback(msg)
    def _handle_success(self, feedback: dict):
        self.get_logger().info(f"✓ Acción completada: {feedback}")
        self._executed_actions.append(feedback)

        # Aumentar costo de la arista ejecutada (ya se hizo)
        action_name = feedback.get('action')
        source      = feedback.get('source', '').split('_')[0]  # robot_left_hand → robot_left_hand
        target      = feedback.get('target', '').split('_')[0]  # apple_1 → apple

        if action_name and source and target:
            G = self.semantic_graph.graph
            if G.has_edge(source, target):
                for key, data in G[source][target].items():
                    if data.get('action') == action_name:
                        old_cost = data.get('cost', 1.0)
                        new_cost = old_cost + 5.0  # penalizar re-ejecución
                        G[source][target][key]['cost']          = new_cost
                        G[source][target][key]['adjusted_cost'] = new_cost
                        self.get_logger().info(
                            f"[policy] Costo actualizado: {source}→{action_name}→{target} "
                            f"{old_cost:.1f} → {new_cost:.1f}"
                        )

        # Ejecutar siguiente paso si quedan
        if hasattr(self, '_pending_plan') and self._pending_plan:
            next_step = self._pending_plan.pop(0)
            action_msg = {
                "action":          next_step[1],
                "source":          next_step[0],
                "target":          next_step[2],
                "remaining_steps": len(self._pending_plan)
            }
            self.results_pub.publish(String(data=json.dumps(action_msg)))
            self.get_logger().info(
                f"[policy] Siguiente: {next_step[0]} → {next_step[1]} → {next_step[2]}"
            )
        else:
            # Plan completo → replanificar
            self.get_logger().info("[policy] Plan completo → replanificando con nueva percepción")
            self._robot_state = "IDLE"
            self.current_goal  = None
            self.plan_logged   = False

    def _handle_failure(self, feedback: dict):
        self.get_logger().warn(f"✗ Acción fallida: {feedback}")
        self.plan_logged = False
        self.current_goal = None

    def execute_goal_with_fallback(self, candidate_goals: List, start_nodes: List[str]) -> Tuple[bool, List[Tuple[str, str, str]], str]:
        """
        Intenta ejecutar objetivos en orden de prioridad, con fallback de búsqueda.
        
        Returns:
            (success, full_plan, selected_goal_text)
        """
        # Ordenar por prioridad
        sorted_goals = sorted(candidate_goals, key=lambda g: g.priority, reverse=True)
        
        for goal in sorted_goals:
            # Evitar alucinaciones
            target_objects = list(dict.fromkeys(goal.target_objects))
            required_actions = list(dict.fromkeys(goal.required_actions))
            self.get_logger().info(f"═══════════════════════════════════════════")
            self.get_logger().info(f"Intentando objetivo: {goal.goal_text} (priority={goal.priority})")
            self.get_logger().info(f"Target objects: {target_objects}")
            self.get_logger().info(f"Required actions: {required_actions}")
            self.get_logger().info(f"═══════════════════════════════════════════")
             
            # Intentar generar plan
            success, plan_edges = self.plan_with_m_etd(
                required_actions=required_actions,
                target_objects=target_objects,
                start_nodes=start_nodes
            )
            
            if success:
                # Plan exitoso
                self.get_logger().info(f"✓ Plan generado exitosamente para: {goal.goal_text}")
                return True, plan_edges, goal.goal_text
            else:
                self.get_logger().warn(f"✗ No se pudo generar plan para: {goal.goal_text}, intentando siguiente objetivo...")
                self.current_goal = None 
                self.plan_logged = False 
                
        
        # No quedaron objetivos
        self.get_logger().error("No se pudo generar plan para ningún objetivo")
        self.current_goal = None
        return False, [], ""
    
    def _log_spatial_relations(self):
        if not self.perceptual_graph.instances:
            return

        relations_log = []
        for inst_id, inst in self.perceptual_graph.instances.items():
            if inst.relations:
                u, v, z = inst.position
                pos_str = f"({u:.0f}, {v:.0f}, {z:.2f}m)" if z > 0 else f"({u:.0f}, {v:.0f})"
                relations_log.append(f"  {inst_id} at {pos_str}:")
                for rel_type, targets in inst.relations.items():
                    if targets:
                        distances = []
                        for target_id in targets[:3]:
                            if target_id in self.perceptual_graph.instances:
                                target = self.perceptual_graph.instances[target_id]
                                dist_3d = inst.distance_to(target)
                                distances.append(f"{target_id} ({dist_3d:.2f}m)")
                        relations_log.append(f"    {rel_type}: {', '.join(distances)}")

        # ← FALTABA ESTO
        if relations_log:
            self.get_logger().info(
                "[spatial]\n" + "\n".join(relations_log),
                throttle_duration_sec=5.0
            )
    def run_planning_pipeline(self):
        if not self.perceptual_graph.instances:
            return

        # ── Si está ejecutando, esperar feedback ──────────────────────────────
        if self._robot_state == "EXECUTING":
            return

        # ── Generar goal si no hay uno ────────────────────────────────────────
        if not hasattr(self, "current_goal") or self.current_goal is None:
            if self.mode == 'auto':
                self.get_logger().info("Generando objetivo con LLM...")
                self.candidate_goals = self.goal_generator.generate_goals(
                    perceptual_graph=self.perceptual_graph,
                    semantic_graph=self.semantic_graph.graph,
                    use_llm=True,
                    mode='auto',
                    user_text=None
                )
                if self.candidate_goals:
                    detected_concepts = list(set(
                        inst.concept for inst in self.perceptual_graph.instances.values()
                    ))
                    if random.random() < epsilon_object and detected_concepts:
                        random_concept = random.choice(detected_concepts)
                        matching = [g for g in self.candidate_goals
                                    if random_concept in g.target_objects]
                        chosen_goal = random.choice(matching) if matching \
                                    else random.choice(self.candidate_goals)
                        self.get_logger().info(
                            f"[epsilon-greedy] Explorando objeto '{random_concept}' "
                            f"-> {chosen_goal.goal_text}")
                    else:
                        chosen_goal = max(self.candidate_goals, key=lambda g: g.priority)
                        self.get_logger().info(f"[epsilon-greedy] Explotando prioridad")
                    self.current_goal = chosen_goal

                    self.get_logger().info("Objetivos sugeridos:")
                    for g in self.candidate_goals:
                        self.get_logger().info(f" - {g.goal_text} (priority={g.priority})")
                    self.get_logger().info(f"Objetivo seleccionado: {self.current_goal.goal_text}")
                else:
                    self.get_logger().error("No se generaron objetivos.")
                    return

            elif self.mode == 'manual':
                if self.user_goal is None:
                    return
                if self.conversation_started:
                    return
                self.conversation_started = True
                self.candidate_goals = self.goal_generator.generate_goals(
                    perceptual_graph=self.perceptual_graph,
                    semantic_graph=self.semantic_graph.graph,
                    use_llm=True,
                    mode='manual',
                    user_text=self.user_goal
                )
                self.get_logger().info(
                    f"Objetivos generados: {len(self.candidate_goals) if self.candidate_goals else 0}"
                )
                if self.candidate_goals:
                    chosen_goal = max(self.candidate_goals, key=lambda g: g.priority)
                    self.current_goal = chosen_goal
                    self.get_logger().info("Objetivos sugeridos:")
                    for g in self.candidate_goals:
                        self.get_logger().info(f" - {g.goal_text} (priority={g.priority})")
                    self.get_logger().info(f"Objetivo seleccionado: {self.current_goal.goal_text}")
                else:
                    self.get_logger().error("No se generaron objetivos.")
                    return

        # ── Planificar ────────────────────────────────────────────────────────
        if self.current_goal and not self.plan_logged:
            target_objects  = self.current_goal.target_objects
            required_actions = self.current_goal.required_actions

            # Extraer targets si están vacíos
            if not target_objects:
                self.get_logger().warn("No target objects — extrayendo del texto")
                target_objects = [
                    cid for cid in self.semantic_graph.concept_nodes.keys()
                    if cid.split("_")[0] in self.current_goal.goal_text.lower()
                ]
            if not target_objects:
                self.get_logger().warn("No se encontraron target objects.")
                self.current_goal = None
                return

            # Extraer acciones si están vacías
            if not required_actions:
                self.get_logger().warn("No required actions — extrayendo del texto")
                required_actions = [
                    aid for aid in self.robot_capabilities.actions
                    if aid.name in self.current_goal.goal_text.lower()
                ]
                required_actions = [a.name for a in required_actions]
            if not required_actions:
                self.get_logger().warn("No se encontraron required actions.")
                self.current_goal = None
                return

            self.get_logger().info(f"Goal:            {self.current_goal.goal_text}")
            self.get_logger().info(f"Target objects:  {target_objects}")
            self.get_logger().info(f"Required actions:{required_actions}")

            # Nodos de inicio del robot
            possible_starts = [
                nid for nid in self.semantic_graph.concept_nodes.keys()
                if nid.startswith("robot_")
            ]
            if not possible_starts:
                self.get_logger().warn("No hay nodos de inicio válidos.")
                return

            # Generar plan con fallback
            success, full_plan, selected_goal_text = self.execute_goal_with_fallback(
                candidate_goals=self.candidate_goals,
                start_nodes=possible_starts
            )

            if not success:
                self.get_logger().error("NO OBJECTIVES LEFT — no se pudo generar plan")
                return

            self.full_plan     = full_plan
            self._pending_plan = list(full_plan[1:])  # pasos restantes
            self.plan_logged   = True

            # Log del plan completo
            plan_str = " → ".join(f"{u}·{a}·{v}" for u, a, v in full_plan)
            self.get_logger().info(f"═══════════════════════════════════════════")
            self.get_logger().info(f"PLAN GENERADO — Goal: {selected_goal_text}")
            self.get_logger().info(f"  {plan_str}")
            self.get_logger().info(f"═══════════════════════════════════════════")

            first = full_plan[0]
            self._current_executing_step = first  
            self._action_start_time      = time.time() 
            action_msg = {
                "action":          first[1],
                "source":          first[0],
                "target":          first[2],
                "remaining_steps": len(self._pending_plan)
            }
            plan_msg = String()
            plan_msg.data = json.dumps(action_msg)
            self.results_pub.publish(plan_msg)

            self._robot_state = "EXECUTING"
            self.get_logger().info(
                f"[policy] Ejecutando paso 1/{len(full_plan)}: "
                f"{first[0]} → {first[1]} → {first[2]}"
            )

            if self.mode == 'manual':
                self.current_goal        = None
                self.user_goal           = None
                self.conversation_started = False
                self.get_logger().info("Esperando nuevo objetivo.")

        if not hasattr(self, "_last_scene_snapshot"):
            self._last_scene_snapshot = (0, 0, 0, 0)

        perceptual_stats = self.perceptual_graph.to_json()["statistics"]
        semantic_stats   = self.semantic_graph.export_to_json()["statistics"]
        current_snapshot = (
            perceptual_stats["total_instances"],
            perceptual_stats["total_relations"],
            semantic_stats["total_concepts"],
            semantic_stats["total_actions"]
        )
        if current_snapshot != self._last_scene_snapshot:
            self.get_logger().info(
                f"Scene Update:\n"
                f"  Perceptual: {perceptual_stats['total_instances']} instances, "
                f"{perceptual_stats['total_relations']} spatial relations\n"
                f"  Semantic: {semantic_stats['total_concepts']} concepts, "
                f"{semantic_stats['total_actions']} possible actions"
            )
            self._last_scene_snapshot = current_snapshot

        self._log_spatial_relations()

    def scene_callback(self, msg):
        try:
            data = json.loads(msg.data)
            objects = data.get('objects', [])
            if not objects:
                return

            # ── 1. SIEMPRE actualizar el grafo perceptual ─────────────────────
            self.perceptual_graph.clear()
            new_concepts = set()

            for obj in objects:
                object_type = obj['label']
                pos = obj['position']
                position = (pos['x'], pos['y'], pos['z'])
                confidence = obj.get('conf', 1.0)

                # ── 2. ¿Existe en la knowledge base? ─────────────────────────
                if object_type not in self.knowledge_base.concepts:
                    self.get_logger().info(
                        f'[knowledge] Concepto nuevo detectado: "{object_type}" → '
                        f'consultando {"Groq" if self.groq else "default"}...'
                    )
                    self.knowledge_base.learn_concept(object_type)
                    new_concepts.add(object_type)
                
                concept = self.knowledge_base.get_concept(object_type)
                if concept and object_type in new_concepts:
                    self.get_logger().info(
                        f'[knowledge] "{object_type}" aprendido → '
                        f'affordances: {concept.affordances} | tool: {concept.tool}'
                    )

                # Agregar al grafo perceptual independientemente
                self.perceptual_graph.add_instance(
                    concept=object_type,
                    position=position,
                    confidence=confidence,
                    bbox=None,
                    affordances=list(concept.affordances) if concept else [],
                    tool=concept.tool if concept else False,
                    safety_level=concept.safety_level if concept else "unknown",
                    contextual_info=concept.contextual_info if concept else "",
                    learned_from=concept.learned_from if concept else "unknown",
                    visual_properties=concept.visual_properties if concept else {},  # ← nuevo
                )

            self.perceptual_graph.compute_spatial_relations()

            # ── Sincronizar instancias perceptuales al grafo semántico ───────────
            for inst_id, inst in self.perceptual_graph.instances.items():
                concept = self.knowledge_base.get_concept(inst.concept)
                if not concept:
                    continue

                concept_name = inst.concept  # ← usar "apple" no "apple_1"

                # Actualizar posición en el nodo de concepto existente
                if concept_name in self.semantic_graph.graph:
                    self.semantic_graph.graph.nodes[concept_name]['position']   = inst.position
                    self.semantic_graph.graph.nodes[concept_name]['confidence'] = inst.confidence
                    self.semantic_graph.graph.nodes[concept_name]['detected']   = True

                safety_cost = {
                    "safe":      0.1,
                    "caution":   0.5,
                    "dangerous": 2.0,
                    "unknown":   1.0,
                }.get(inst.safety_level, 1.0)

                # Aristas robot → concepto (no instancia)
                for action in self.robot_capabilities.actions:
                    has_target_aff = concept.affordances & set(action.target_affordances)
                    has_tool_aff   = concept.affordances & set(action.tool_affordances)

                    if has_target_aff:
                        for robot_node in [n for n in self.semantic_graph.graph.nodes
                                        if str(n).startswith("robot_")]:
                            if not self.semantic_graph.graph.has_edge(robot_node, concept_name):
                                self.semantic_graph.graph.add_edge(
                                    robot_node, concept_name,
                                    action=action.name,
                                    cost=safety_cost,
                                    adjusted_cost=safety_cost,
                                    ethical_weight=1.0,
                                    preconditions=action.preconditions,
                                )

                    # Knife → cut → apple (herramienta sobre otro concepto)
                    if has_tool_aff:
                        for other_id, other_inst in self.perceptual_graph.instances.items():
                            if other_id == inst_id:
                                continue
                            other_concept_name = other_inst.concept
                            other_concept = self.knowledge_base.get_concept(other_concept_name)
                            if not other_concept:
                                continue
                            has_other_target = other_concept.affordances & set(action.target_affordances)
                            if has_other_target:
                                if not self.semantic_graph.graph.has_edge(concept_name, other_concept_name):
                                    self.semantic_graph.graph.add_edge(
                                        concept_name, other_concept_name,
                                        action=action.name,
                                        cost=safety_cost,
                                        adjusted_cost=safety_cost,
                                        ethical_weight=1.0,
                                        preconditions=action.preconditions,
                                    )
            # ── 3. Si hay conceptos nuevos → actualizar grafo semántico ──────
            if new_concepts:
                self.semantic_graph.build_full_action_graph(
                    new_concepts,
                    self.robot_capabilities.actions
                )
                for concept in new_concepts:
                    if concept in self.semantic_graph.graph:
                        edges = list(self.semantic_graph.graph.in_edges(concept, data=True))
                        self.get_logger().info(
                            f'[semantic] Aristas hacia "{concept}": {len(edges)} → '
                            f'{[(u, d.get("action")) for u, v, d in edges[:3]]}'
                        )
                    else:
                        self.get_logger().warn(
                            f'[semantic] "{concept}" no tiene nodo en el grafo semántico'
                        )

                # ── Solo replanificar si NO estamos ejecutando ────────────────────
                if self._robot_state != "EXECUTING":
                    self.current_goal = None
                    self.plan_logged  = False
                    self.get_logger().info(
                        f'[scene] Nuevos conceptos {new_concepts} → replanificando'
                    )
                else:
                    self.get_logger().info(
                        f'[scene] Nuevos conceptos {new_concepts} detectados → '
                        f'plan en ejecución, NO se interrumpe'
                    )

            # Publicar para visualizador
            detections = []
            for inst_id, inst in self.perceptual_graph.instances.items():
                x, y, z = inst.position
                detections.append({
                    "class_name":     inst.concept,
                    "confidence":     inst.confidence,
                    "x": x, "y": y, "z": z,
                    "center": [x, y],
                    "bbox": [x - 25, y - 25, 50, 100],
                    "affordances":    inst.affordances,
                    "tool":           inst.tool,
                    "safety_level":   inst.safety_level,
                    "visual_properties": inst.visual_properties,
                })
            perc_msg = String()
            perc_msg.data = json.dumps({"detections": detections})
            self.perceptual_pub.publish(perc_msg)

            # ── 4. Planning con cooldown ──────────────────────────────────────
            self.run_planning_pipeline()

        except Exception as e:
            self.get_logger().error(f'Scene callback error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    
    def wait_for_detection(self, obj_name, timeout=10.0):
        import time
        start = time.time()
        while time.time() - start < timeout:
            detected = self.perceptual_graph.instances.keys()

            for inst in detected:
                if inst.startswith(f"{obj_name}_"):
                    return True
                
            rclpy.spin_once(self, timeout_sec=0.1)
        return False
        
    def find_meta_edges(self, action_name: str, targets: List[str]):
        """
        Devuelve lista de aristas (src, tgt, edge_data) que representan
        action_name aplicada sobre uno de los targets.
        Si targets tiene 2 elementos y existe una arista directa src->tgt con action_name,
        la devuelve con preferencia.
        """
        G = self.semantic_graph.graph
        meta_edges = []

        # Si binaria (2 targets) tratamos de encontrar exactamente src->action->tgt
        if len(targets) == 2:
            src_candidate, tgt_candidate = targets[0], targets[1]
            if src_candidate in G and tgt_candidate in G[src_candidate]:
                for key, edata in G[src_candidate][tgt_candidate].items():
                    if edata.get("action") == action_name:
                        meta_edges.append((src_candidate, tgt_candidate, dict(edata)))
                if meta_edges:
                    return meta_edges

        # Si unaria o no se encontro match binario: buscar edges ... -> target
        for tgt in targets:
            if tgt not in G:
                continue
            for u in G.predecessors(tgt):
                for key, edata in G[u][tgt].items():
                    if edata.get("action") == action_name:
                        meta_edges.append((u, tgt, dict(edata)))
        return meta_edges
    
    def find_producer_edges(self, sources:List[str], target: str, action):
        """
        Buscar aristas que 'producen' el objeto/destino, ej:
          robot_camera --search_for--> cloth
        Devuelve (u, v, edata) candidates donde v == produced_target
        """
        # Intentar optimizar con GNN
        if self.use_gnn and self.gnn_optimizer and self.current_goal:
            G = self.gnn_optimizer.optimize_costs(
                semantic_graph=self.semantic_graph.graph,
                perceptual_graph=self.perceptual_graph,
                goal=self.current_goal.goal_text,
                target_objects=self.current_goal.target_objects,
                required_actions=self.current_goal.required_actions
            )
        meta_edges = []

        detected = self.perceptual_graph.instances.keys()

        detected_sources = []
        indetected_sources = []

        # Verificar si existe alguna instancia detectada de este concepto
        for src in sources:
            src_detected = any(det_concept.split("_")[0] == src for det_concept in detected)
            if src_detected and src in self.current_goal.target_objects:
                detected_sources.append(src)
            else:
                indetected_sources.append(src)

        for src in detected_sources:
            if src == target:
                continue
            candidate = G[src][target]
            for key, edata in candidate.items():
                if edata.get("action") == action:
                    meta_edges.append((src, target, dict(edata)))

        if not detected_sources:
            best_src = None
            best_cost = float('inf')
            for src in indetected_sources:
                if src == target:
                    continue
                cost = nx.dijkstra_path_length(G, src, target, weight='adjusted_cost' if self.use_gnn else 'cost')
                if cost < best_cost:
                    best_cost = cost
                    best_src = src
                edata = G[best_src][target][0]
            meta_edges.append((best_src, target, dict(edata)))
        return meta_edges
    
    def preconditions_satisfied(self, edge_data, current_world_state):
        """
        Verifica preconditions simples de la arista (edge_data['preconditions']).
        current_world_state puede ser una estructura simple (ej. manos libres,
        objetos en percepcion).
        """
        preconds = edge_data.get("preconditions", {}) or {}
        seen = current_world_state.get("seen", set())
        hands = current_world_state.get("hands", {"left": "free", "right": "free"})

        # tool availability check
        if preconds.get("tool") is True:
            tool_found = False
            for inst in self.perceptual_graph.instances.values():
                concept = self.knowledge_base.get_concept(inst.concept)
                if concept and getattr(concept, "tool", False):
                    tool_found = True
                    break
            if not tool_found:
                return False

        # robot_hand: if requires free and both hands are occupied -> false
        if preconds.get("robot_hand") == "free":
            if hands.get("left") != "free" and hands.get("right") != "free":
                return False

        # visible precondition: require that the target is seen
        if preconds.get("visible") is True:
            target = edge_data.get("target_concept")
            if target and target not in seen:
                return False

        return True

    def dijkstra_cost_to_node(self, start_nodes: List[str], target_node: str, graph=None):
        """
        Ejecuta Dijkstra desde cualquier start_node hasta target_node y
        retorna (best_start, path, cost) o (None, None, inf) si no es alcanzable.
        Usa GNN si está habilitado para optimizar costos.
        """
        G = graph if graph is not None else self.semantic_graph.graph
        best_cost = float('inf')
        best_path = None
        best_start = None
        weight_key = 'cost'

        # Intentar optimizar con GNN
        if self.use_gnn and self.gnn_optimizer and self.current_goal:
            try:
                G = self.gnn_optimizer.optimize_costs(
                    semantic_graph=self.semantic_graph.graph,
                    perceptual_graph=self.perceptual_graph,
                    goal=self.current_goal.goal_text,
                    target_objects=self.current_goal.target_objects,
                    required_actions=self.current_goal.required_actions
                )
                weight_key = 'adjusted_cost'
            except Exception as e:
                self.get_logger().warn(f"GNN optimization failed during dijkstra_cost_to_node: {e}")
                weight_key = 'cost'

        for s in start_nodes:
            try:
                path = nx.dijkstra_path(G, s, target_node, weight=weight_key)
                cost = nx.dijkstra_path_length(G, s, target_node, weight=weight_key)
                if cost < best_cost:
                    best_cost = cost
                    best_path = path
                    best_start = s
            except nx.NetworkXNoPath:
                continue

        if best_path is None:
            return None, None, float('inf')
        return best_start, best_path, best_cost
    
    def filter_actions(self, required_actions, action_definitions):
        """
        required_actions: lista de acciones generadas por el LLM
        action_definitions: metadata de cada acción, ej:
            {
                "grasp":  {"preconditions": {"robot_hand": True}},
                "cut":    {"preconditions": {"tool": False}},
                "mix":    {"preconditions": {"tool": True}},
                "move_to": {"preconditions": {"tool": False}}
            }
        """
        
        actions_with_tool_key = []
        actions_without_tool_key = []

        for action_name in required_actions:
            info = action_definitions.get(action_name, {})
            pre = info.get("preconditions", {})

            if "tool" in pre:
                actions_with_tool_key.append(action_name)
            else:
                actions_without_tool_key.append(action_name)

        # Si existe AL MENOS UNA acción con la llave "tool"
        if actions_with_tool_key:
            return actions_with_tool_key

        # Si NO existen acciones con "tool", entonces devolver todo
        return required_actions

    def plan_with_m_etd(self, required_actions: List[str], target_objects: List[str], start_nodes: List[str]):
        """
        Implementa el flujo M-ETD: para cada action en required_actions
        busca aristas meta (action->target), dijkstra hasta la fuente; si no hay arista
        o no hay camino, intenta encontrar 'producer edges' (search_for/fetch) como fallback.
        Devuelve (success, plan_edges_list) donde plan_edges_list = [(u, action, v), ...]
        """
        plan = []
        possible_starts = start_nodes[:]  # lista de node ids
        current_world = {
            "seen": set(key.split('_')[0] for key in self.perceptual_graph.instances.keys()),  # instancias detectadas
            "hands": {"left": self.current_state.get("robot_left_hand","free"),
                      "right": self.current_state.get("robot_right_hand","free")}
        }

        # Intentar optimizar con GNN
        if self.use_gnn and self.gnn_optimizer and self.current_goal:
            G = self.gnn_optimizer.optimize_costs(
                semantic_graph=self.semantic_graph.graph,
                perceptual_graph=self.perceptual_graph,
                goal=self.current_goal.goal_text,
                target_objects=self.current_goal.target_objects,
                required_actions=self.current_goal.required_actions
            )
                
        used_path_edges = set()
        used_meta_edges = set()

        action_definitions = {
        a.name: {"preconditions": a.preconditions}
        for a in self.robot_capabilities.actions}   
        actions = self.filter_actions(required_actions, action_definitions)

        for action in actions:
            meta_edges = self.find_meta_edges(action, target_objects)

            # fallback: si no hay meta edges, buscar productores (search_for, fetch)
            if not meta_edges:
                return False, []
            
            completed_edges = []
            sources = list({src for (src, tgt, edata) in meta_edges if not src.startswith("robot_")})
            targets = list({tgt for (src, tgt, edata) in meta_edges})
            if sources:
                for tgt in targets:
                    producer_candidates = self.find_producer_edges(sources, tgt, action)
                    if len(producer_candidates) == 1 and producer_candidates[0][0] not in current_world["seen"]:
                        self.get_logger().warn(f"Searching {producer_candidates[0][0]} to execute {action}...")
                        # Intentar buscar el objeto
                        msg = f"search_for({sources})"
                        self.results_pub.publish(String(data=msg))
                        found = self.wait_for_detection(sources, timeout=10.0)
                        
                        if not found:
                            self.get_logger().warn(f"{sources} not found after search, trying next goal")
                            return False, []
                        
                        # Reintentar buscar productores
                        producer_candidates = self.find_producer_edges(sources, tgt, action)
                    else:
                        completed_edges.extend(producer_candidates)
            else:
                completed_edges.extend(meta_edges)
            # seleccionar la meta_edge mejor reachable
            best_combo = []
            for src, tgt, edata in completed_edges:
                if edata.get("ethical_weight", 1.0) == 0.0:
                    self.get_logger().warn("Ethical violation command. Aborting plan.")
                    return False, []
                start, path, cost_to_src = self.dijkstra_cost_to_node(possible_starts, src)
                if path is None:
                    continue
                best_combo.append((path,(src, edata.get('action'), tgt)))

            if not best_combo:
                return False, []
            
            paths = [p for p, me in best_combo]
            meta_edges = [me for p, me in best_combo]

            for path, meta_edge in zip(paths, meta_edges):
                pairs = list(nx.utils.pairwise(path))

                for (u, v) in pairs:
                    # epilson-greedy para selección de arista en caso de múltiples opciones
                    all_edges = G[u][v] 
                    if random.random() < epsilon_action and len(all_edges) > 1:
                        best_key = random.choice(list(all_edges.keys()))
                        self.get_logger().info(
                            f"[ε-greedy] Explorando acción aleatoria en {u}→{v}")
                    else:
                        best_key = min(all_edges,
                                    key=lambda k: all_edges[k].get('adjusted_cost', 2.0))
                    action_name = all_edges[best_key].get('action')
                    edge_tuple = (u, action_name, v)
                    if edge_tuple in used_path_edges:
                        continue
                    used_path_edges.add(edge_tuple)
                    plan.append(edge_tuple)

                # anexar la arista meta
                if meta_edge in used_meta_edges:
                    continue
                used_meta_edges.add(meta_edge)  
                plan.append(meta_edge)

            return True, plan
    
    def animate_plan(self, full_plan, filename="reasoning_sequence.gif"):
        reasoning_path = []
        
        # Nodos válidos en el grafo semántico base (solo conceptos abstractos)
        valid_nodes = set(self.semantic_graph.concept_nodes.keys())
        
        for i, step in enumerate(full_plan):
            source_node, action, target_node = step
            
            # Mapear instancias (apple_1) a conceptos abstractos (apple)
            def to_concept(node):
                if node in valid_nodes:
                    return node
                # Quitar el sufijo _N
                base = "_".join(node.split("_")[:-1]) if "_" in node else node
                return base if base in valid_nodes else None
            
            display_source = to_concept(source_node)
            display_target = to_concept(target_node)
            
            # Saltar si algún nodo no existe en el grafo semántico
            if not display_source or not display_target:
                self.get_logger().warn(
                    f"[animate] Saltando paso {i+1}: "
                    f"{source_node}→{action}→{target_node} "
                    f"(nodos no encontrados en grafo semántico)"
                )
                continue
            
            reasoning_path.append({
                'description': f"Step {i+1}: {display_source} → {action} → {display_target}",
                'active_nodes': [display_source, display_target],
                'active_edge':  (display_source, display_target, action)
            })
        
        if not reasoning_path:
            self.get_logger().warn("[animate] No hay frames válidos para el GIF")
            return
        
        print(f"Plan recibido con {len(reasoning_path)} pasos. Generando GIF...")
        self.semantic_graph.generate_reasoning_gif(reasoning_path, filename)

    def periodic_save(self):
        try:
            self.knowledge_base.save()
            self.semantic_graph.save_graph("semantic_graph.json")
            self.perceptual_graph.save_to_file("perceptual_graph.json")
            if self.full_plan and self.mode == 'auto':
                self.animate_plan(self.full_plan, filename="reasoning_robot.gif")
        except Exception as e:
            self.get_logger().error(f"Auto-save error: {e}")
    
    def save_all(self):
        self.get_logger().info("Saving all components...")
        
        self.knowledge_base.save()
        self.semantic_graph.save_graph("semantic_graph.json")
        self.perceptual_graph.save_to_file("perceptual_graph.json")
        
        # Visualizar grafo
        try:
            self.semantic_graph.visualize("semantic_action_graph.png")
            self.perceptual_graph.visualize("perceptual_graph.png")
        except Exception as e:
            self.get_logger().warn(f"Visualization error: {e}")
        
        self.get_logger().info("✓ All saved")

# ============================================================================
# MAIN
# ============================================================================

def main(args=None):
    rclpy.init(args=args)
    
    node = GraphAutonomousNode(
        use_gnn=True
    )
    
    # 1. Creamos una bandera para saber si el guardado ha terminado
    save_finished_event = threading.Event()
    
    def background_save():
        """Función que se ejecutará en un hilo secundario para guardar."""
        node.get_logger().info("Starting long save operation...")
        # Llama a tu función original de guardado
        node.save_all() 
        node.get_logger().info("Long save operation finished.")
        # Señalizamos que el guardado ha terminado
        save_finished_event.set() 

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down... Starting asynchronous save.")
        
        # 2. Iniciamos el hilo de guardado inmediatamente
        save_thread = threading.Thread(target=background_save)
        save_thread.start()
        
        # 3. Esperamos activamente por un tiempo limitado (ej: 9 segundos)
        # Esto permite que el nodo principal responda al SIGINT/SIGTERM mientras guarda
        save_thread.join(timeout=14.0) 
        
        # 4. Verificamos si el hilo terminó antes de la fuerza bruta (SIGKILL)
        if not save_finished_event.is_set():
            node.get_logger().warn("Save operation exceeded 9 seconds! Shutting down forcefully.")
        else:
            node.get_logger().info("Save operation successful.")
            
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()