# semantic_knowledge.py

import json
from typing import Dict, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
import os
from neopath.robot_physical_capacities import RobotCapabilities
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    _embed_model = None

def _embed_properties(visual_properties: dict, concept: str) -> np.ndarray:
    """Convierte propiedades visuales a vector de embedding."""
    if not _embed_model or not visual_properties:
        return np.zeros(384)
    
    text = f"{concept}: "
    text += ", ".join(f"{k} is {v}" for k, v in visual_properties.items())
    
    return _embed_model.encode(text)

@dataclass
class ConceptKnowledge:
    concept_type: str
    tool: bool
    affordances: Set[str]
    physical_properties: Dict[str, any]
    contextual_info: str        # ← sin default, van primero
    safety_level: str           # ← ídem
    living_organism: bool = False
    usage_count: int = 0
    learned_from: str = "predefined"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    visual_properties: Dict[str, str] = field(default_factory=dict) 

    def to_dict(self):
        return {
            "concept_type": self.concept_type,
            "tool": self.tool,
            "affordances": list(self.affordances),
            "physical_properties": self.physical_properties,
            "contextual_info": self.contextual_info,
            "safety_level": self.safety_level,
            "living_organism": self.living_organism,
            "usage_count": self.usage_count,
            "learned_from": self.learned_from,
            "timestamp": self.timestamp,
            "visual_properties": self.visual_properties
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        data["affordances"] = set(data["affordances"])
        data.setdefault("visual_properties", {})
        return cls(**data)

class KnowledgeBase:
    """Base de conocimiento de conceptos (permanente)"""
    
    def __init__(self, storage_path: str = "knowledge_base.json", groq_client=None):
        self.storage_path = storage_path
        self.concepts: Dict[str, ConceptKnowledge] = {}
        self.groq = groq_client
        self.load()
        self._initialize_base_concepts()
    
    def _initialize_base_concepts(self):
        """Conceptos base predefinidos"""
        base_concepts = {
            "knife": ConceptKnowledge(
                concept_type="knife",
                tool=True,
                affordances={"pickable", "cutting", "movable", "inspectable", "graspable"},
                physical_properties={"weight": "light", "fragility": "sturdy"},
                contextual_info="Sharp cutting tool, dangerous to living beings",
                safety_level="caution",
                learned_from="predefined"
            ),
            "avocado": ConceptKnowledge(
                concept_type="avocado",
                tool=False,
                affordances={"cuttable", "edible", "scoopable"},
                physical_properties={"weight": "varies", "fragility": "delicate"},
                contextual_info="A vegetable that is edible and used for salads; handle with care",
                safety_level="safe",
                learned_from="predefined"
            )
        }
        
        for concept_type, concept in base_concepts.items():
            if concept_type not in self.concepts:
                self.concepts[concept_type] = concept
    
    def has_concept(self, concept_type: str) -> bool:
        return concept_type in self.concepts
    
    def get_concept(self, concept_type: str) -> Optional[ConceptKnowledge]:
        return self.concepts.get(concept_type)
    
    def learn_concept(self, concept_type: str) -> ConceptKnowledge:
        """Aprende un nuevo concepto usando Groq"""
        
        if self.groq:
            concept = self._learn_from_groq(concept_type)
        else:
            concept = ConceptKnowledge(
                concept_type=concept_type,
                tool=False,
                affordances={"inspectable"},
                physical_properties={},
                
                contextual_info=f"Unknown object: {concept_type}",
                safety_level="caution",
                learned_from="default"
            )
        
        self.concepts[concept_type] = concept
        self.save()
        
        print(f"✓ Learned new concept: {concept_type} (affordances: {concept.affordances})")
        return concept
    
    def suggest_affordances(self, capabilities: RobotCapabilities) -> Set[str]:
        """Sugiere affordances para un concepto dado"""
        recommended: Set[str] = set()
        for action in capabilities.actions:
            recommended.update(action.required_affordances)
            recommended.update(action.tool_affordances)
            recommended.update(action.target_affordances)
        return recommended
    
    def _learn_from_groq(self, concept_type: str) -> ConceptKnowledge:
        """Usa Groq para aprender sobre un concepto"""
        recommended = self.suggest_affordances(RobotCapabilities())
        
        prompt = f"""
You are an affordance reasoning system for a robotic agent.

Analyze the object type "{concept_type}" and infer its relevant properties.

AVAILABLE AFFORDANCES: {list(recommended)}

Respond in JSON:
{{
    "tool": true/false, 
    "affordances": [..subset of these: {list(recommended)}],
    "physical_properties": {{
        "weight": "light/medium/heavy",
        "fragility": "fragile/sturdy"
    }},
    "visual_properties": {{
        "color": "primary color(s), e.g. red, green, brown",
        "shape": "geometric shape, e.g. round, cylindrical, flat",
        "size": "small/medium/large",
        "texture": "smooth/rough/soft/hard",
        "transparency": "opaque/transparent/translucent"
    }},
    "contextual_info": "Brief description of purpose and common use",
    "safety_level": "safe/caution/dangerous",
    "living_organism": true/false
}}
...
Guidelines:
- Primarly focus on affordances from the provided list of valid affordances. However, you may include additional relevant affordances if justified.
- Combine both **physical affordances** and **functional affordances** when appropriate.
- Consider the object weight in determining if it's 'pickable' or 'movable', robot can't lift heavy objects.
- Do not include affordances that describe content or meaning (e.g., 'readable', 'consumable', 'viewable').
- Always include 'inspectable' if the object can be visually analyzed by a camera or sensor.
- Aim for 5–8 meaningful affordances that can directly connect with robot actions.
CRITICAL RULES:
- Affordances MUST be from the list of available affordances.     
"""

        try:
            import time
            import numpy as np

            time.sleep(1 + np.random.uniform(0, 0.5))  # evitar rate limits

            response = self.groq.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                json_mode=True
            )
            
            data = json.loads(response.strip())
            
            concept = ConceptKnowledge(
                concept_type=concept_type,
                tool=data.get("tool", False),
                affordances=set(data.get("affordances", ["inspectable"])),
                physical_properties=data.get("physical_properties", {}),
                visual_properties=data.get("visual_properties", {}),  # ← nuevo
                contextual_info=data.get("contextual_info", ""),
                safety_level=data.get("safety_level", "caution"),
                living_organism=data.get("living_organism", False),
                learned_from="groq"
            )
            
            return concept
            
        except Exception as e:
            print(f"Error learning from Groq: {e}")
            return ConceptKnowledge(
                concept_type=concept_type,
                tool=False,
                affordances={"inspectable"},
                physical_properties={},
                visual_properties={},
                contextual_info="",
                safety_level="caution",
                living_organism=False,
                learned_from="error_default"
            )
    
    def update_usage(self, concept_type: str):
        """Actualiza contador de uso"""
        if concept_type in self.concepts:
            self.concepts[concept_type].usage_count += 1
    
    def save(self):
        """Guarda base de conocimiento"""
        try:
            data = {
                "concepts": {k: v.to_dict() for k, v in self.concepts.items()},
                "statistics": {
                    "total_concepts": len(self.concepts),
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"✓ Saved knowledge base: {len(self.concepts)} concepts")
            
        except Exception as e:
            print(f"Error saving knowledge base: {e}")
    
    def load(self):
        """Carga base de conocimiento"""
        if not os.path.exists(self.storage_path):
            return
        
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)
            
            for concept_type, concept_data in data.get("concepts", {}).items():
                self.concepts[concept_type] = ConceptKnowledge.from_dict(concept_data)
            
            print(f"✓ Loaded knowledge base: {len(self.concepts)} concepts")
            
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
