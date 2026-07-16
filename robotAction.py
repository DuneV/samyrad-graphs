# class robot Action

from dataclasses import dataclass
import pandas as pd


@dataclass
class Action:
    name: str


ROBOT_ACTIONS = [
    Action(name="grasp"), Action(name="release"), Action(name="push"),
    Action(name="pull"),  Action(name="lift"),    Action(name="lower"),
    Action(name="rotate"),Action(name="twist"),   Action(name="slide"),
    Action(name="squeeze"),
]
ROBOT_ACTION_NAMES = {a.name for a in ROBOT_ACTIONS}

EPIC_TO_ROBOT = {
    "take": "grasp",     "put": "release",   "wash": None,
    "open": "pull",      "close": "push",    "insert": "slide",
    "turn-on": "rotate", "cut": None,        "turn-off": "rotate",
    "pour": "rotate",    "mix": "rotate",    "move": "slide",
    "remove": "pull",    "throw": "release", "dry": None,
    "shake": None,       "scoop": "lift",    "adjust": "rotate",
    "squeeze": "squeeze","peel": "pull",     "empty": "rotate",
    "press": "push",     "flip": "rotate",   "turn": "rotate",
    "check": None,       "scrape": "slide",  "fill": None,
    "apply": None,       "fold": None,       "scrub": None,
    "break": "pull",     "pull": "pull",     "pat": "push",
    "lift": "lift",      "hold": "grasp",    "eat": None,
    "wrap": None,        "filter": None,     "look": None,
    "unroll": "pull",    "sort": None,       "hang": "lift",
    "sprinkle": None,    "rip": "pull",      "spray": "squeeze",
    "cook": None,        "add": None,        "roll": "rotate",
    "search": None,      "crush": "squeeze", "stretch": "pull",
    "knead": "squeeze",  "divide": "pull",   "set": None,
    "feel": None,        "rub": None,        "soak": None,
    "brush": None,       "sharpen": None,    "drop": "release",
    "drink": None,       "slide": "slide",   "water": None,
    "gather": "grasp",   "attach": "push",   "turn-down": "rotate",
    "coat": None,        "transition": None, "wear": None,
    "measure": None,     "increase": "rotate","unscrew": "twist",
    "wait": None,        "lower": "lower",   "form": "squeeze",
    "smell": None,       "use": None,        "grate": None,
    "screw": "twist",    "let-go": "release","finish": None,
    "stab": "push",      "serve": "release", "uncover": "pull",
    "unwrap": "pull",    "choose": None,     "lock": "rotate",
    "flatten": "push",   "switch": "push",   "carry": "grasp",
    "season": None,      "unlock": "rotate", "prepare": None,
    "bake": None,        "mark": None,       "bend": None,
    "unfreeze": None,
}


class RobotActions:
    def __init__(self, file: str):
        self.df = pd.read_csv(file)
        self.epic_verbs = self.df["key"].tolist()

        missing = set(self.epic_verbs) - EPIC_TO_ROBOT.keys()
        if missing:
            print(f"Verbos EPIC sin entrada en el mapeo: {sorted(missing)}")

        self.mapped = {
            verb: EPIC_TO_ROBOT[verb]
            for verb in self.epic_verbs
            if EPIC_TO_ROBOT.get(verb) in ROBOT_ACTION_NAMES
        }

    def robot_action_for(self, epic_verb: str) -> str | None:
        return self.mapped.get(epic_verb)

    def instances_by_robot_action(self) -> dict[str, list[str]]:
        """Agrupa: primitiva robótica lista de verbos EPIC que la disparan."""
        grouped: dict[str, list[str]] = {a: [] for a in ROBOT_ACTION_NAMES}
        for verb, action in self.mapped.items():
            grouped[action].append(verb)
        return grouped


if __name__ == "__main__":
    ra = RobotActions("verb_classes.csv")
    print(f"{len(ra.mapped)}/{len(ra.epic_verbs)} verbos EPIC mapeados a primitivas robot\n")
    for action, verbs in ra.instances_by_robot_action().items():
        print(f"{action:10s} <- {verbs}")