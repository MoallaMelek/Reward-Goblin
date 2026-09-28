"""Reward functions assembled from a fixed catalogue of components.

There is deliberately no way to inject code: a reward is a JSON document of weights,
two numeric parameters and a handful of termination switches. Everything the RL agent is
trained on is computed here; the human's real objective lives in
``backend/analysis/true_objective.py`` and is never mixed in.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field

# key -> metadata used by the API, the UI editor and the linter.
COMPONENTS: dict[str, dict] = {
    "distance_progress": {
        "label": "Distance progress",
        "group": "shaping",
        "default": 1.0,
        "unit": "per metre",
        "description": "Reward when the box gets closer to the exit, and an equal penalty when it moves away "
                       "(potential-based: the total telescopes to start distance minus end distance).",
    },
    "path_progress": {
        "label": "Path progress",
        "group": "shaping",
        "default": 1.0,
        "unit": "per metre",
        "description": "Like distance progress, but distance is measured along walkable paths, so walls are "
                       "respected (potential-based).",
    },
    "closer_bonus": {
        "label": "Closer bonus",
        "group": "shaping",
        "default": 1.0,
        "unit": "per metre",
        "description": "Reward whenever the box gets closer to the exit. Moving it away costs nothing.",
    },
    "approach_speed": {
        "label": "Hurry bonus",
        "group": "shaping",
        "default": 0.3,
        "unit": "per step at full speed",
        "description": "Reward the goblin for moving quickly toward the exit (meant to discourage dawdling).",
    },
    "agent_box_progress": {
        "label": "Approach the box",
        "group": "shaping",
        "default": 0.3,
        "unit": "per metre",
        "description": "Potential-based reward for the goblin getting closer to the box.",
    },
    "goal_contact": {
        "label": "Exit contact",
        "group": "goal",
        "default": 0.1,
        "unit": "per step",
        "description": "Reward every step the box touches (overlaps) the exit zone.",
    },
    "goal_entry": {
        "label": "Exit entry",
        "group": "goal",
        "default": 5.0,
        "unit": "per event",
        "description": "Reward each time the box starts touching the exit zone.",
    },
    "proximity": {
        "label": "Near the exit",
        "group": "goal",
        "default": 0.1,
        "unit": "per step",
        "description": "Reward every step the box centre is within a radius of the exit centre (straight-line).",
    },
    "stable_goal": {
        "label": "Stable completion",
        "group": "goal",
        "default": 10.0,
        "unit": "once",
        "description": "Reward once when the box has stayed completely inside the exit for the required frames.",
    },
    "survival": {
        "label": "Survival bonus",
        "group": "time",
        "default": 0.1,
        "unit": "per step",
        "description": "Reward every step the episode keeps going.",
    },
    "step_penalty": {
        "label": "Time penalty",
        "group": "time",
        "default": -0.01,
        "unit": "per step",
        "description": "Penalty every simulation step (encourages finishing quickly).",
    },
    "movement_penalty": {
        "label": "Movement penalty",
        "group": "cost",
        "default": -0.005,
        "unit": "per step",
        "description": "Penalty for every step the goblin is actively moving.",
    },
    "collision_penalty": {
        "label": "Collision penalty",
        "group": "cost",
        "default": -0.1,
        "unit": "per hit",
        "description": "Penalty each time the goblin bumps into a wall or obstacle.",
    },
    "hazard_penalty": {
        "label": "Lava penalty",
        "group": "cost",
        "default": -1.0,
        "unit": "once",
        "description": "Penalty when the goblin touches lava.",
    },
}

PARAMS: dict[str, dict] = {
    "stable_goal_frames": {"label": "Required duration", "default": 30, "min": 1, "max": 300,
                           "unit": "frames (15 fps)"},
    "proximity_radius": {"label": "Near-exit radius", "default": 2.5, "min": 0.2, "max": 10.0, "unit": "m"},
}

TERMINATION: dict[str, dict] = {
    "on_contact": {"label": "Terminate when box touches exit", "default": False},
    "on_inside": {"label": "Terminate when box is fully inside exit", "default": False},
    "on_stable": {"label": "Terminate when box stays inside exit (required duration)", "default": True},
    "on_hazard": {"label": "Terminate when goblin touches lava (leaves allowed region)", "default": True},
    "max_steps": {"label": "Terminate after N steps", "default": 300, "min": 50, "max": 1000},
}

WEIGHT_LIMIT = 100.0


@dataclass
class RewardConfig:
    name: str = "Custom reward"
    components: dict[str, float] = field(default_factory=dict)
    params: dict[str, float] = field(default_factory=dict)
    termination: dict = field(default_factory=dict)

    def __post_init__(self):
        self.params = {k: self.params.get(k, v["default"]) for k, v in PARAMS.items()}
        self.termination = {k: self.termination.get(k, v["default"]) for k, v in TERMINATION.items()}
        self.termination["max_steps"] = int(self.termination["max_steps"])
        self.params["stable_goal_frames"] = int(self.params["stable_goal_frames"])
        self.components = {k: float(v) for k, v in self.components.items() if float(v) != 0.0}

    # --- serialisation ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, d: dict) -> "RewardConfig":
        """Accepts the structured form or the flat form from the README
        (``{"goal_contact": 5.0, "stable_goal_frames": 30, ...}``). Raises ValueError on bad input."""
        errors, _ = validate(d)
        if errors:
            raise ValueError("; ".join(errors))
        comps, params, term = _split(d)
        return cls(name=str(d.get("name", "Custom reward")), components=comps, params=params, termination=term)

    def to_dict(self) -> dict:
        return {"name": self.name, "components": dict(self.components),
                "params": dict(self.params), "termination": dict(self.termination)}

    def fingerprint(self) -> str:
        body = {k: v for k, v in self.to_dict().items() if k != "name"}
        return hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:10]

    @property
    def stable_frames(self) -> int:
        return int(self.params["stable_goal_frames"])

    @property
    def max_steps(self) -> int:
        return int(self.termination["max_steps"])


def _split(d: dict) -> tuple[dict, dict, dict]:
    comps = dict(d.get("components", {}))
    params = dict(d.get("params", {}))
    term = dict(d.get("termination", {}))
    for k, v in d.items():  # flat form
        if k in COMPONENTS:
            comps[k] = v
        elif k in PARAMS:
            params[k] = v
        elif k in TERMINATION:
            term[k] = v
    return comps, params, term


def validate(d: dict) -> tuple[list[str], list[dict]]:
    """Returns (errors, lint warnings). Errors make the config unusable."""
    errors: list[str] = []
    if not isinstance(d, dict):
        return ["reward config must be a JSON object"], []
    allowed_top = {"name", "components", "params", "termination", "description", "version"}
    for k in d:
        if k not in allowed_top and k not in COMPONENTS and k not in PARAMS and k not in TERMINATION:
            errors.append(f"unknown field '{k}'")
    comps, params, term = _split(d)
    for k, v in comps.items():
        if k not in COMPONENTS:
            errors.append(f"unknown reward component '{k}'")
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            errors.append(f"component '{k}' must be a finite number")
        elif abs(v) > WEIGHT_LIMIT:
            errors.append(f"component '{k}' weight must be within ±{WEIGHT_LIMIT:g}")
    for k, v in params.items():
        if k not in PARAMS:
            errors.append(f"unknown parameter '{k}'")
            continue
        spec = PARAMS[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not (spec["min"] <= v <= spec["max"]):
            errors.append(f"parameter '{k}' must be a number in [{spec['min']}, {spec['max']}]")
    for k, v in term.items():
        if k not in TERMINATION:
            errors.append(f"unknown termination rule '{k}'")
        elif k == "max_steps":
            spec = TERMINATION[k]
            if (isinstance(v, bool) or not isinstance(v, (int, float)) or not float(v).is_integer()
                    or not (spec["min"] <= v <= spec["max"])):
                errors.append(f"max_steps must be an integer in [{spec['min']}, {spec['max']}]")
        elif not isinstance(v, bool):
            errors.append(f"termination rule '{k}' must be true/false")
    if not errors and not any(float(v) != 0.0 for v in comps.values()):
        errors.append("enable at least one reward component")
    if errors:
        return errors, []
    return [], lint(RewardConfig(components=comps, params=params, termination=term))


def lint(cfg: RewardConfig) -> list[dict]:
    """Static heuristics: which loopholes a goblin is likely to find. Not a proof of anything."""
    c, t = cfg.components, cfg.termination
    w: list[dict] = []

    def add(code, severity, msg):
        w.append({"code": code, "severity": severity, "message": msg})

    success_ends = t["on_inside"] or t["on_stable"] or t["on_contact"]
    if c.get("closer_bonus", 0) > 0:
        add("positive_only_progress", "high",
            "Closer bonus pays for approaching but never charges for retreating: pushing the box back "
            "and forth is pure profit.")
    if c.get("approach_speed", 0) > 0:
        add("rewards_motion", "high",
            "Hurry bonus rewards the goblin's own motion toward the exit, not the box's. Running laps pays.")
    if c.get("goal_entry", 0) > 0 and not t["on_contact"]:
        add("repeatable_event", "high",
            "Exit entry pays again every time the box re-enters. Leaving and coming back is profitable.")
    if c.get("goal_contact", 0) > 0:
        if t["on_inside"] or t["on_stable"]:
            add("finish_ends_income", "high",
                "Exit contact pays every step, but pushing the box fully inside ends the episode. "
                "Parking the box on the edge keeps the income flowing.")
        add("partial_overlap", "medium",
            "'Touching' is satisfied by a sliver of overlap: the box never has to be inside.")
    if c.get("proximity", 0) > 0:
        add("euclidean_through_walls", "high" if cfg.params["proximity_radius"] >= 1.0 else "medium",
            "'Near the exit' is a straight-line distance: it ignores walls, so a box pinned on the wrong "
            "side of a wall still counts.")
    per_step_pos = c.get("survival", 0) + max(0.0, c.get("step_penalty", 0))
    if per_step_pos > 0 and success_ends:
        income = per_step_pos * cfg.max_steps
        add("stalling_pays", "high" if income > c.get("stable_goal", 0) else "medium",
            f"A positive per-step reward (up to {income:.1f} per episode) stops the moment the task is done. "
            f"Completion pays {c.get('stable_goal', 0):.1f}. Procrastination may be optimal.")
    if c.get("step_penalty", 0) < 0 and t["on_hazard"]:
        worst_life = abs(c["step_penalty"]) * cfg.max_steps
        death = abs(min(0.0, c.get("hazard_penalty", 0)))
        if worst_life > death:
            add("early_exit_cheaper", "high" if worst_life > 2 * death + 1 else "medium",
                f"Living a full episode can cost up to {worst_life:.1f}, touching lava costs {death:.1f} and "
                "ends it. Jumping into the lava may be the cheapest option.")
    if c.get("distance_progress", 0) > 0 or c.get("closer_bonus", 0) > 0:
        add("euclidean_shaping", "low",
            "Distance shaping uses straight-line distance. On maps with walls it can point the box into a "
            "dead end; path progress respects walls.")
    if c.get("stable_goal", 0) <= 0:
        add("no_completion_reward", "medium",
            "Nothing rewards the actual objective (box resting inside the exit). Success is incidental.")
    if c.get("stable_goal", 0) > 0 and not success_ends:
        add("no_terminal", "low", "Completion is paid once but the episode keeps running afterwards.")
    if c.get("movement_penalty", 0) < 0 and c.get("stable_goal", 0) <= 0:
        add("lazy_optimum", "medium", "Movement costs reward and completion pays nothing: standing still is safe.")
    return w


# ------------------------------------------------------------------------ per-step rewards
@dataclass
class StepSignals:
    box_d: float                # box centre -> exit centre (m)
    prev_box_d: float
    path_d: float               # walkable box -> exit distance (only computed when used)
    prev_path_d: float
    agent_box_d: float
    prev_agent_box_d: float
    agent_speed_to_goal: float  # goblin velocity projected on direction to exit / max speed
    overlap: bool               # box AABB overlaps exit
    entered: bool               # overlap started this step
    stable_reached: bool        # stable counter reached N this step (first time)
    moving: bool
    wall_hit: bool              # goblin collision onset
    hazard_hit: bool            # goblin touched lava this step (onset)


def compute(cfg: RewardConfig, s: StepSignals) -> dict[str, float]:
    c = cfg.components
    out: dict[str, float] = {}
    for key, w in c.items():
        if key == "distance_progress":
            v = w * (s.prev_box_d - s.box_d)
        elif key == "path_progress":
            v = w * (s.prev_path_d - s.path_d)
        elif key == "closer_bonus":
            v = w * max(0.0, s.prev_box_d - s.box_d)
        elif key == "approach_speed":
            v = w * max(0.0, s.agent_speed_to_goal)
        elif key == "agent_box_progress":
            v = w * (s.prev_agent_box_d - s.agent_box_d)
        elif key == "goal_contact":
            v = w if s.overlap else 0.0
        elif key == "goal_entry":
            v = w if s.entered else 0.0
        elif key == "proximity":
            v = w if s.box_d <= cfg.params["proximity_radius"] else 0.0
        elif key == "stable_goal":
            v = w if s.stable_reached else 0.0
        elif key in ("survival", "step_penalty"):
            v = w
        elif key == "movement_penalty":
            v = w if s.moving else 0.0
        elif key == "collision_penalty":
            v = w if s.wall_hit else 0.0
        elif key == "hazard_penalty":
            v = w if s.hazard_hit else 0.0
        else:  # pragma: no cover - validate() rejects unknown keys
            continue
        out[key] = v
    return out
