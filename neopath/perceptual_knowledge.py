# perceptual_knowledge.py

import json
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import networkx as nx

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# =========================================================
# Instance Node
# =========================================================

@dataclass
class InstanceNode:
    id: str
    concept: str
    position: Tuple[float, float, float]  # (u, v, depth)
    confidence: float
    relations: Dict[str, List[str]] = field(default_factory=dict)
    bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    frame_count: int = 1
    affordances: List[str] = field(default_factory=list)
    tool: bool = False
    safety_level: str = "unknown"
    contextual_info: str = ""
    learned_from: str = "unknown"
    visual_properties: Dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return {
            "id": self.id,
            "concept": self.concept,
            "position": list(self.position),
            "confidence": self.confidence,
            "relations": self.relations,
            "bbox": list(self.bbox),
            "timestamp": self.timestamp,
            "frame_count": self.frame_count,
            "affordances": self.affordances,
            "tool": self.tool,
            "safety_level": self.safety_level,
            "contextual_info": self.contextual_info,
            "learned_from": self.learned_from,
            "visual_properties": self.visual_properties,
        }

    def distance_to(self, other: "InstanceNode") -> float:
        """Compute approximate 3D distance using camera intrinsics."""
        u1, v1, z1 = self.position
        u2, v2, z2 = other.position

        fx, fy = 554.38, 554.38
        cx, cy = 320.0, 240.0

        X1 = (u1 - cx) * z1 / fx
        Y1 = (v1 - cy) * z1 / fy
        X2 = (u2 - cx) * z2 / fx
        Y2 = (v2 - cy) * z2 / fy

        return np.linalg.norm(
            np.array([X1, Y1, z1]) - np.array([X2, Y2, z2])
        )

    def get_3d_position(self) -> np.ndarray:
        """Convert image coordinates (u, v, z) to real-world 3D coordinates."""
        u, v, z = self.position
        fx, fy = 554.38, 554.38
        cx, cy = 320.0, 240.0

        X = (u - cx) * z / fx
        Y = (v - cy) * z / fy
        Z = z

        return np.array([X, Y, Z])


# =========================================================
# Perceptual Graph
# =========================================================

class PerceptualGraph:
    def __init__(
        self,
        near_threshold: float = 200.0,        # pixels for "near"
        far_threshold: float = 500.0,         # pixels for "far"
        vertical_threshold: float = 50.0,     # pixels for above/below
        horizontal_threshold: float = 100.0,  # pixels for left/right
        depth_threshold: float = 0.5           # meters for front/behind
    ):
        self.instances: Dict[str, InstanceNode] = {}
        self.graph = nx.DiGraph()

        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self.vertical_threshold = vertical_threshold
        self.horizontal_threshold = horizontal_threshold
        self.depth_threshold = depth_threshold

        self.instance_counters: Dict[str, int] = {}
        self.total_detections = 0
        self.frame_number = 0

    def clear(self):
        """Clear instances and reset counters for a new frame."""
        self.instances.clear()
        self.graph.clear()
        self.instance_counters = {}
        self.frame_number += 1

    def add_instance(
        self,
        concept: str,
        position: Tuple[float, float, float],
        confidence: float,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        affordances: List[str] = None,
        tool: bool = False,
        safety_level: str = "unknown",
        contextual_info: str = "",
        learned_from: str = "unknown",
        visual_properties: Dict[str, str] = None,
    ) -> str:

        if concept not in self.instance_counters:
            self.instance_counters[concept] = 1

        instance_id = f"{concept}_{self.instance_counters[concept]}"
        self.instance_counters[concept] += 1

        instance = InstanceNode(
            id=instance_id,
            concept=concept,
            position=position,
            confidence=confidence,
            bbox=bbox or (0, 0, 0, 0),
            affordances=affordances or [],
            tool=tool,
            safety_level=safety_level,
            contextual_info=contextual_info,
            learned_from=learned_from,
            visual_properties=visual_properties or {},
        )

        self.instances[instance_id] = instance
        self.graph.add_node(instance_id, concept=concept, position=position, confidence=confidence)
        self.total_detections += 1
        return instance_id

    def add_relation(self, relation: str, source_id: str, target_id: str):
        if source_id not in self.instances or target_id not in self.instances:
            return

        source = self.instances[source_id]
        source.relations.setdefault(relation, [])

        if target_id not in source.relations[relation]:
            source.relations[relation].append(target_id)

        self.graph.add_edge(source_id, target_id, relation=relation)

    # -----------------------------------------------------
    # Spatial relations (2D pixels + optional depth)
    # -----------------------------------------------------

    def compute_spatial_relations(self):
        """
        Compute spatial relations between instances:
        - near / far (2D pixel distance)
        - left_of / right_of (image X axis)
        - above / below (image Y axis)
        - in_front_of / behind (depth Z, if available)
        """
        instances = list(self.instances.values())

        for i, inst1 in enumerate(instances):
            for inst2 in instances[i + 1:]:

                u1, v1, z1 = inst1.position
                u2, v2, z2 = inst2.position

                dist_2d = np.sqrt((u2 - u1) ** 2 + (v2 - v1) ** 2)

                dx = u2 - u1      # + => inst2 is to the right
                dy = v2 - v1      # + => inst2 is below (image coordinates)
                dz = z2 - z1      # + => inst2 is farther away

                # Near / Far
                if dist_2d < self.near_threshold:
                    self.add_relation("near", inst1.id, inst2.id)
                    self.add_relation("near", inst2.id, inst1.id)
                elif dist_2d > self.far_threshold:
                    self.add_relation("far", inst1.id, inst2.id)
                    self.add_relation("far", inst2.id, inst1.id)

                # Left / Right
                if abs(dx) > self.horizontal_threshold:
                    if dx > 0:
                        self.add_relation("right_of", inst1.id, inst2.id)
                        self.add_relation("left_of", inst2.id, inst1.id)
                    else:
                        self.add_relation("left_of", inst1.id, inst2.id)
                        self.add_relation("right_of", inst2.id, inst1.id)

                # Above / Below (image Y+ goes down)
                if abs(dy) > self.vertical_threshold:
                    if dy > 0:
                        self.add_relation("above", inst1.id, inst2.id)
                        self.add_relation("below", inst2.id, inst1.id)
                    else:
                        self.add_relation("below", inst1.id, inst2.id)
                        self.add_relation("above", inst2.id, inst1.id)

                # In front / Behind (only if depth is available)
                if z1 > 0 and z2 > 0 and abs(dz) > self.depth_threshold:
                    if dz > 0:
                        self.add_relation("in_front_of", inst1.id, inst2.id)
                        self.add_relation("behind", inst2.id, inst1.id)
                    else:
                        self.add_relation("behind", inst1.id, inst2.id)
                        self.add_relation("in_front_of", inst2.id, inst1.id)

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    def visualize(self, filename: str = "perceptual_graph.png"):
        """Visualize the perceptual graph."""
        if not plt:
            print("Matplotlib not installed.")
            return

        plt.figure(figsize=(16, 12))
        pos = nx.spring_layout(self.graph, k=3, iterations=100, seed=42)

        concepts = list(set(d["concept"] for _, d in self.graph.nodes(data=True)))
        cmap = plt.cm.get_cmap("Pastel1", len(concepts) + 1)

        node_colors = [
            cmap(concepts.index(d["concept"]))
            for _, d in self.graph.nodes(data=True)
        ]

        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=2500,
            node_color=node_colors,
            edgecolors="black",
            linewidths=1.5
        )

        nx.draw_networkx_labels(self.graph, pos, font_size=10, font_weight="bold")

        relation_colors = {
            "near": "green",
            "far": "red",
            "left_of": "blue",
            "right_of": "cyan",
            "above": "purple",
            "below": "magenta",
            "in_front_of": "orange",
            "behind": "brown"
        }

        ax = plt.gca()

        for u, v, data in self.graph.edges(data=True):
            rel = data["relation"]
            color = relation_colors.get(rel, "black")

            ax.annotate(
                "",
                xy=pos[v], xycoords="data",
                xytext=pos[u], textcoords="data",
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    linewidth=1.5,
                    alpha=0.7
                )
            )

            mid_x = (pos[u][0] + pos[v][0]) / 2
            mid_y = (pos[u][1] + pos[v][1]) / 2

            plt.text(
                mid_x, mid_y, rel,
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color)
            )

        plt.title(
            f"Perceptual Graph – Frame {self.frame_number}\n"
            "Detected Instances and Spatial Relations",
            fontsize=16
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"✓ Perceptual graph visualization saved to {filename}")

    # -----------------------------------------------------
    # Export
    # -----------------------------------------------------

    def to_json(self) -> Dict:
        return {
            "frame_number": self.frame_number,
            "timestamp": datetime.now().isoformat(),
            "instances": {i: n.to_dict() for i, n in self.instances.items()},
            "relations": [
                {"source": u, "target": v, "relation": d["relation"]}
                for u, v, d in self.graph.edges(data=True)
            ],
            "statistics": {
                "total_instances": len(self.instances),
                "unique_concepts": len(set(i.concept for i in self.instances.values())),
                "total_relations": self.graph.number_of_edges(),
                "total_detections": self.total_detections
            }
        }

    def save_to_file(self, filename: str = "perceptual_graph.json"):
        """Guarda el grafo en archivo"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.to_json(), f, indent=2)
                print(f"✓ Perceptual graph saved to {filename}")

        except Exception as e:
            print(f"Error saving perceptual graph: {e}")