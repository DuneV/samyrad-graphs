import json
from typing import Dict, Optional, Set, List
from dataclasses import dataclass, field
import os

@dataclass
class RobotAction:
    name: str
    description: str
    required_affordances: Set[str]
    preconditions: Dict[str, str]
    effects: Dict[str, str]
    cost: float = 1.0
    risk_level: str = "low"
    tool_affordances: Set[str] = field(default_factory=set)
    target_affordances: Set[str] = field(default_factory=set)
    
    def can_apply_to_concept(self, affordances: Set[str]) -> bool:
        """Verifica si puede aplicarse a un concepto con estas affordances"""
        return self.required_affordances.issubset(affordances)
    
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "required_affordances": list(self.required_affordances),
            "preconditions": self.preconditions,
            "effects": self.effects,
            "cost": self.cost,
            "risk_level": self.risk_level,
            "tool_affordances": list(self.tool_affordances),
            "target_affordances": list(self.target_affordances),
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        data["required_affordances"] = set(data["required_affordances"])
        data["tool_affordances"] = set(data.get("tool_affordances", []))
        data["target_affordances"] = set(data.get("target_affordances", []))
        return cls(**data)

class RobotCapabilities:
    def __init__(self, actions_file: str = "robot_actions.json"):
        self.actions_file = actions_file
        self.actions: List[RobotAction] = []
        self.load_actions()
    
    def load_actions(self):
        if os.path.exists(self.actions_file):
            try:
                with open(self.actions_file, 'r') as f:
                    data = json.load(f)
                    self.actions = [RobotAction.from_dict(a) for a in data["actions"]]
                print(f"✓ Loaded {len(self.actions)} robot actions")
            except Exception as e:
                print(f"Error loading actions: {e}")
                self._create_default_actions()
        else:
            self._create_default_actions()
    
    def _create_default_actions(self):
        self.actions = [
            RobotAction(
                name="inspect",
                description="Visually inspect an object",
                required_affordances={"inspectable"},
                preconditions={"robot_camera": "free"},
                effects={"state": "inspected"},
                cost=1.0,
                risk_level="low"
            ),
            RobotAction(
                name="greet",
                description="Greet a person.",
                required_affordances={"human"},
                preconditions={"robot_hand": "free"},
                effects={"robot_hand": "holding"},
                cost=0.5,
                risk_level="low"
            ),
            RobotAction(
                name="search_for",
                description="Visually search for an object using robot camera",
                required_affordances={"inspectable"},
                preconditions={"robot_camera": "holding"},
                effects={"robot_camera": "searched"},
                cost=1.5,
                risk_level="low"
            ),
            RobotAction(
                name="move_to",
                description="Move an object to a new location",
                required_affordances={"non_heavy_object", 'movable'},
                preconditions={"tool": False},
                effects={"state": "moved"},
                cost=0.8,
                risk_level="medium",
                target_affordances={"placement_surface"}
            ),
            RobotAction(
                name="cut",
                description="Slice an object using a knife",
                required_affordances={"cuttable_object", "food"},
                preconditions={"tool": True},
                effects={"state": "cutted"},
                cost=2.0,
                risk_level="medium",
                tool_affordances={'sharp', 'cutting_tool'},
                target_affordances={'cuttable_object', 'food'}
            ),
        ]
        self.save_actions()
    
    def save_actions(self):
        try:
            data = {"actions": [a.to_dict() for a in self.actions]}
            with open(self.actions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving actions: {e}")
    
    def get_action(self, name: str) -> Optional[RobotAction]:
        for action in self.actions:
            if action.name == name:
                return action
        return None