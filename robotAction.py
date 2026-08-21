from dataclasses import dataclass
import pandas as pd


@dataclass
class action:
    name: str


ROBOT_ACTIONS = [action(name=n) for n in
    ["grasp", "release", "push", "pull", "lift", "lower",
     "rotate", "twist", "slide", "squeeze"]]
ROBOT_ACTION_NAMES = {a.name for a in ROBOT_ACTIONS}


# ── IMPORTANTE ────────────────────────────────────────────────────────
# Este mapeo apunta al vocabulario de 12 acciones que CLIP realmente
# puede predecir (definido en storyTelling.py: CANDIDATE_ACTIONS /
# CLIP_TO_ACTION), NO al de las 10 primitivas cinemáticas de arriba
# (ROBOT_ACTIONS). Son dos vocabularios distintos en tu proyecto:
#
#   - ROBOT_ACTIONS (10 primitivas): grasp, release, push, pull, lift,
#     lower, rotate, twist, slide, squeeze — pensado para control de
#     bajo nivel del robot.
#   - CLIP_TO_ACTION (12 acciones semánticas): grasp, move_to, pour,
#     cut, open, close, push, pull, press, rotate, inspect, remove_from
#     — es lo que CLIP clasifica y lo que arma edge_labels/semantic_line.
#
# EPIC_TO_ROBOT se usa para comparar el ground truth de EPIC-KITCHENS
# contra lo que predice CLIP (evaluate_detector_vs_gt /
# evaluate_chunks_against_gt.py) — por eso DEBE mapear al vocabulario
# de CLIP. Si en algún momento comparas contra las 10 primitivas
# cinemáticas en cambio, necesitas un segundo diccionario aparte.
CLIP_ACTION_VOCAB = {
    "grasp", "move_to", "pour", "cut", "open", "close",
    "push", "pull", "press", "rotate", "inspect", "remove_from",
}

EPIC_TO_ROBOT = {
    "add":        None,
    "adjust":     "rotate",
    "apply":      "press",
    "attach":     "push",
    "break":      None,
    "brush":      None,
    "carry":      "move_to",
    "check":      "inspect",
    "choose":     None,
    "close":      "close",
    "coat":       None,
    "cook":       None,
    "crush":      "press",
    "cut":        "cut",
    "divide":     "cut",
    "drink":      None,
    "dry":        None,
    "empty":      "pour",
    "fill":       "pour",
    "filter":     None,
    "flatten":    "press",
    "flip":       "rotate",
    "form":       None,
    "gather":     "grasp",
    "hold":       "grasp",
    "increase":   "rotate",
    "insert":     "push",
    "knead":      "press",
    "lift":       "grasp",
    "lower":      "push",
    "mix":        "rotate",
    "move":       "move_to",
    "open":       "open",
    "pat":        "press",
    "peel":       "cut",
    "pour":       "pour",
    "press":      "press",
    "pull":       "pull",
    "put":        "move_to",
    "put-down":   "move_to",
    "put-into":   "move_to",
    "put-onto":   "move_to",
    "remove":     "remove_from",
    "rip":        "cut",
    "roll":       "rotate",
    "rub":        "press",
    "scoop":      "grasp",
    "scrape":     "pull",
    "screw":      "rotate",
    "scrub":      "press",
    "season":     None,
    "serve":      "move_to",
    "set":        "move_to",
    "shake":      "rotate",
    "sharpen":    None,
    "slide":      "push",
    "soak":       None,
    "sort":       None,
    "spray":      "press",
    "sprinkle":   None,
    "squeeze":    "press",
    "stab":       "cut",
    "stretch":    "pull",
    "take":       "grasp",
    "throw":      "move_to",
    "turn":       "rotate",
    "turn-down":  "rotate",
    "turn-off":   "press",
    "turn-on":    "press",
    "uncover":    "open",
    "unroll":     "pull",
    "unscrew":    "rotate",
    "unwrap":     "open",
    "use":        None,
    "wash":       None,
    "wrap":       "close",
}


class RobotActions:
    def __init__(self, file: str):
        self.df = pd.read_csv(file)
        self.epic_verbs = self.df["key"].tolist()

        missing = set(self.epic_verbs) - EPIC_TO_ROBOT.keys()
        if missing:
            print(f"⚠ Verbos EPIC sin entrada en el mapeo: {sorted(missing)}")

        self.mapped = {
            verb: EPIC_TO_ROBOT[verb]
            for verb in self.epic_verbs
            if EPIC_TO_ROBOT.get(verb) is not None
        }

    def robot_action_for(self, epic_verb: str):
        return self.mapped.get(epic_verb)


if __name__ == "__main__":
    print(f"{len(EPIC_TO_ROBOT)} verbos EPIC mapeados")
    print(f"{sum(1 for v in EPIC_TO_ROBOT.values() if v is not None)} con equivalente en el vocabulario de CLIP")