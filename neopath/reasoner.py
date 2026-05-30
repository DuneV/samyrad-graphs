import json
from typing import List, Set
from dataclasses import dataclass, field
from datetime import datetime
from neopath.perceptual_knowledge import PerceptualGraph
from neopath.robot_physical_capacities import RobotCapabilities

@dataclass
class GeneratedGoal:
    """Representa un objetivo estructurado que el robot puede intentar lograr."""
    goal_text: str                   # Descripción concisa del objetivo
    target_objects: List[str]
    required_actions: List[str]
    priority: float = 1.0            # 0.0–1.0, mayor = más importante
    context: str = ""                # Resumen de la escena
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    provenance: str = "groq"         # "groq" si viene del modelo, "heuristic" si fallback

class GoalGenerator:
    """
    Genera objetivos autónomos a partir del estado perceptual y semántico del robot.
    Usa un LLM (Groq) para inferir propósitos contextuales y filtra con semantic_graph opcionalmente.
    """

    def __init__(self, groq_client=None, do_feasibility_check=True, max_candidates=8):
        self.groq = groq_client
        self.do_feasibility_check = do_feasibility_check
        self.max_candidates = max_candidates

    def generate_goals(self, perceptual_graph, semantic_graph=None, use_llm=True, mode='auto', user_text='observe'):
        """
        Genera una lista de posibles objetivos basados en lo que el robot percibe.
        - perceptual_graph: estado actual de la escena
        - semantic_graph: para verificar factibilidad
        - use_llm: si True, usa Groq para razonar sobre el contexto
        """
        if not perceptual_graph.instances:
            return []

        scene_description = self._describe_scene(perceptual_graph)

        # Generar metas desde LLM
        if use_llm and self.groq:
            if mode == 'auto':
                raw_goals = self._generate_goals_from_llm(scene_description, perceptual_graph)

            if mode == 'manual':
                raw_goals = self.generate_goal_from_user_input(user_text, scene_description, perceptual_graph)
        else:
            raw_goals = [GeneratedGoal(
                goal_text="Observe environment", 
                priority=0.1,
                target_objects=[],
                required_actions=[],
                context=scene_description)]
        
        # Filtrar por factibilidad usando semantic_graph (opcional)
        if self.do_feasibility_check and semantic_graph:
            feasible_goals = []
            for g in raw_goals:
                # Validar que los target objects existan en la escena
                if self._validate_goal(g, perceptual_graph):
                    feasible_goals.append(g)
            return feasible_goals or raw_goals  # fallback a todos si ninguno es factible

        return raw_goals

    def _describe_scene(self, perceptual_graph):
        """Crea un resumen textual de la escena para el LLM"""
        descriptions = []
        for inst in perceptual_graph.instances.values():
            x, y, z = inst.position
            descriptions.append(f"{inst.concept} at ({x:.0f}, {y:.0f}, {z:.2f}m)")
        return " | ".join(descriptions)
    
    def available_actions(self, capabilities: RobotCapabilities) -> Set[str]:
        """Sugiere affordances para un concepto dado"""
        available: Set[str] = set()
        for action in capabilities.actions:
            available.add(action.name)
        return available

    def _generate_goals_from_llm(self, scene_description, perceptual_graph):
        """Usa Groq para inferir propósitos y objetivos contextuales"""
        
        # Extraer lista de objetos detectados
        detected_objects = list(set(inst.concept for inst in perceptual_graph.instances.values()))
        available_actions = self.available_actions(RobotCapabilities())
        
        prompt = f"""
You are an autonomous service robot in a household environment.

Current scene:
{scene_description}

Detected objects: {', '.join(detected_objects)}
Available actions: {', '.join(list(available_actions))}

Generate up to {self.max_candidates} realistic goals for the robot.

CRITICAL RULES:
1. Focus on objects that can be physically manipulated: dishes, utensils, food, containers, tools
2. Target objects MUST be from the detected list above
3. Required actions MUST be from the available actions listed above.

Valid goal examples:
✓ "Clean the table by wiping it with a cloth" → targets: [cloth, table], actions: [grasp, wipe]
✓ "Move the cup to the counter" → targets: [cup, counter], actions: [move_to]
✓ "Greet the person politely" → targets: [person], actions: [interact]
✓ "Inspect the food to check if cooked" → targets: [food], actions: [inspect]

INVALID goals (never suggest these):
✗ "Move person to safe location" → Cannot move people!
✗ "Pick up the cat" → Cannot pick up animals!
✗ "Grasp the human" → Cannot grasp people!

Return ONLY valid JSON:
{{
    "goals": [
        {{
            "goal_text": "string",
            "target_objects": ["object1", "object2"],
            "required_actions": ["action1", "action2"],
            "priority": float
        }}
    ]
}}
"""

        try:
            response = self.groq.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                json_mode=True
            )
            data = json.loads(response.strip())
            goals = []
            for g in data.get("goals", []):
                goals.append(GeneratedGoal(
                    goal_text=g["goal_text"],
                    target_objects=g.get("target_objects", []),
                    required_actions=g.get("required_actions", []),
                    priority=float(g.get("priority", 0.8)),
                    context=scene_description,
                    provenance="groq"
                ))
            return goals
        except Exception as e:
            print(f"[GoalGenerator] Error generating goals: {e}")
            return []
        
    def generate_goal_from_user_input(self, user_text:str, scene_description, perceptual_graph):

        detected_objects = list(set(inst.concept for inst in perceptual_graph.instances.values()))
        available_actions = self.available_actions(RobotCapabilities())
        
        prompt = f"""
You are an autonomous robot. A human has issued a command:

USER GOAL: "{user_text}"
Detected objects: {', '.join(detected_objects)}
Available actions: {', '.join(list(available_actions))}
Current Scene: 
{scene_description}

IMPORTANT:
- Normalize the user's command INTERNALLY into English before reasoning (do NOT output the normalized text).
- Interpret the intention the same way regardless of the original language.
- Your reasoning, action planning, and object selection MUST be language-agnostic.

Your task:
1. Carefully interpret the user's intention.
2. Ensure that the structured goal reflects all objects relevant to that intention, not just a subset.
3. Generate up to {self.max_candidates} realistic goals for the robot.
4. Assign a priority score (0-1) for each goal based on how well it matches the user's intention and includes all relevant objects.

CRITICAL RULES:
1. Target objects MUST be selected only if they are clearly relevant to fulfilling the user intention.
2. Do NOT add extra objects just because they appear in the scene.
3. If the user mentions a single object (e.g., "pick up the cup"), use ONLY that object unless context demands additional ones.
4. Required actions MUST be from the available actions.
5. Do NOT suggest actions on humans or animals.


Valid goal examples:
✓ User GOAL: "Organiza la mesa." 
  goal_text: "Move the cup, mouse, and bowl to the office desk" → 
  targets: [cup, mouse, bowl, office desk], actions: [move_to]
✓ User GOAL: Limpia el escritorio. goal_text:"Clean the table by wiping it with a cloth" → targets: [cloth, table], actions: [grasp, wipe]
✓ User GOAL: Organiza la mesa. goal_text:"Move the cup to the counter" → targets: [cup, counter], actions: [move_to]
✓ User GOAL: Saluda a los invitados. goal_text:"Greet the person politely" → targets: [person], actions: [interact]
✓ User GOAL: Revisa si la comida ya esta lista. goal_text:"Inspect the food to check if cooked" → targets: [food], actions: [inspect]

INVALID goals (never suggest these):
✗ "Move person to safe location" → Cannot move people!
✗ "Pick up the cat" → Cannot pick up animals!
✗ "Grasp the human" → Cannot grasp people!
                    
Return ONLY valid JSON. Include ONLY objects that are necessary for the goal.

{{
    "goals": [
        {{
            "goal_text": "string",
            "target_objects": ["object1", "object2"],
            "required_actions": ["action1", "action2"],
            "priority": float
        }}
    ]
}}
"""
        
        try:
            response = self.groq.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                json_mode=True
            )
            data = json.loads(response.strip())
            goals = []
            for g in data.get("goals", []):
                goals.append(GeneratedGoal(
                    goal_text=g["goal_text"],
                    target_objects=g.get("target_objects", []),
                    required_actions=g.get("required_actions", []),
                    priority=float(g.get("priority", 0.8)),
                    context=scene_description,
                    provenance="groq"
                ))
            return goals
        except Exception as e:
            print(f"[GoalGenerator] Error generating goals: {e}")
            return []
        
    def _validate_goal(self, goal: GeneratedGoal, perceptual_graph: PerceptualGraph) -> bool:
        """
        Valida que los target objects del objetivo existan en la escena
        
        Args:
            goal: Objetivo a validar
            perceptual_graph: Grafo perceptual con objetos detectados
            
        Returns:
            True si al menos el 70% de los target objects están detectados
        """
        if not goal.target_objects:
            return True  # Sin objetos específicos, asumir válido
        
        detected_concepts = set(inst.concept for inst in perceptual_graph.instances.values())
        
        # Contar cuántos target objects están detectados
        detected_targets = [obj for obj in goal.target_objects if obj in detected_concepts]
        
        if len(detected_targets) == 0:
            return False  # Ningún target detectado
        
        # Calcular cobertura
        coverage = len(detected_targets) / len(goal.target_objects)
        
        # Requerir al menos 70% de cobertura
        return coverage >= 0.7

    def _is_feasible(self, goal_text, semantic_graph):
        """
        Chequea si el objetivo propuesto tiene caminos posibles en el semantic_graph.
        Simple: verifica si existe un nodo que contenga keywords del goal.
        """
        try:
            for node in semantic_graph.nodes():
                if any(word.lower() in str(node).lower() for word in goal_text.split()):
                    return True
            return False
        except Exception:
            return True  # fallback a True si el grafo falla