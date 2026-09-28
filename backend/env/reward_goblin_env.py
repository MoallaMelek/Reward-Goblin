"""RewardGoblinEnv: push the box into the exit - or whatever the reward actually says.

Observation (37 floats, roughly in [-1, 1]):
    goblin pos (2), goblin vel (2), box pos (2), box vel (2), exit centre (2),
    box - goblin (2), exit - box (2), box-exit distance (1), goblin wall contact (1),
    box touches exit (1), box fully inside (1), stable-frames fraction (1),
    time remaining fraction (1), box wall contact (1), 8 wall rays, 8 lava rays
Actions: Discrete(5) - noop, left, right, up, down.

The environment keeps a per-episode trace. When an episode ends it (1) evaluates the
*true* objective by freezing the goblin and letting physics settle, and (2) runs the
heuristic exploit detectors; both land in ``info["episode_diag"]`` for logging only.
They never touch the reward.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..analysis.exploit_detection import detect_exploits
from ..analysis.true_objective import TRUE_OBJECTIVE, box_inside
from .physics import (ACTION_NAMES, RAY_COUNT, RAY_RANGE, PhysicsParams, World, box_rect,
                      circle_rect_overlap, rect_center, rect_overlap)
from .planning import DistanceField, box_grid
from .randomization import Layout, generate_layout
from .rewards import RewardConfig, StepSignals, compute

ENV_VERSION = "reward-goblin-env/1.0"
OBS_DIM = 21 + 2 * RAY_COUNT

DEFAULT_REWARD = {
    "name": "Default: distance + contact",
    "components": {"distance_progress": 1.0, "goal_contact": 0.1, "step_penalty": -0.01},
    "termination": {"on_stable": True, "max_steps": 300},
}


@dataclass
class EnvConfig:
    layout: str = "arena"        # named layout ("arena", "walled_exit", "lava_room") or difficulty
    randomize: bool = True       # spawn/goal jitter for named layouts
    settle_frames: int = TRUE_OBJECTIVE["settle_frames"]
    diagnostics: bool = True     # true objective + exploit detectors at episode end

    @classmethod
    def from_dict(cls, d: dict | None) -> "EnvConfig":
        d = dict(d or {})
        known = {k: d[k] for k in ("layout", "randomize", "settle_frames", "diagnostics") if k in d}
        return cls(**known)

    def to_dict(self) -> dict:
        return asdict(self)


class RewardGoblinEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, reward: RewardConfig | dict | None = None, env_config: EnvConfig | dict | None = None,
                 physics: PhysicsParams = PhysicsParams()):
        super().__init__()
        self.reward_cfg = reward if isinstance(reward, RewardConfig) else RewardConfig.from_dict(reward or DEFAULT_REWARD)
        self.cfg = env_config if isinstance(env_config, EnvConfig) else EnvConfig.from_dict(env_config)
        self.phys = physics
        self.action_space = spaces.Discrete(len(ACTION_NAMES))
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(OBS_DIM,), dtype=np.float32)
        self.world: World | None = None
        self.layout: Layout | None = None
        self.last_trace: dict | None = None

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        layout = options.get("layout")
        if layout is None:
            layout = generate_layout(self.cfg.layout, self.np_random, self.cfg.randomize)
        self.layout = layout
        self.world = World(layout.width, layout.height, layout.static_rects,
                           layout.agent_spawn, layout.box_spawn, self.phys)
        self.goal_c = rect_center(layout.goal)
        self.t = 0
        self.stable = 0
        self.stable_paid = False
        self.overlap = self._overlap()
        self.inside = self._inside()
        self.in_hazard = self._hazard()
        self.box_d = self._box_d()
        self.agent_box_d = self._agent_box_d()
        self.field = None
        if "path_progress" in self.reward_cfg.components:
            self.field = DistanceField(box_grid(layout, self.phys.box_half, self.phys.agent_radius), self.goal_c)
        self.path_d = self._path_d()
        self.ep_return = 0.0
        self.trace = self._new_trace()
        self._record(action=0, reward=0.0, comps={}, wall_hit=False)
        return self._obs(), {"layout": layout.name}

    def step(self, action):
        action = int(action)
        w = self.world
        was_wall = w.agent_wall_contact
        was_box_wall = w.box_wall_contact
        w.step(action)
        self.t += 1

        prev_box_d, prev_ab, prev_path = self.box_d, self.agent_box_d, self.path_d
        prev_overlap, prev_inside, prev_hazard = self.overlap, self.inside, self.in_hazard
        self.box_d, self.agent_box_d, self.path_d = self._box_d(), self._agent_box_d(), self._path_d()
        self.overlap, self.inside, self.in_hazard = self._overlap(), self._inside(), self._hazard()
        self.stable = self.stable + 1 if self.inside else 0
        stable_now = (not self.stable_paid) and self.stable >= self.reward_cfg.stable_frames
        if stable_now:
            self.stable_paid = True
        wall_hit = w.agent_wall_contact and not was_wall
        entered = self.overlap and not prev_overlap
        hazard_hit = self.in_hazard and not prev_hazard

        ax, ay = w.agent_pos
        gx, gy = self.goal_c
        dist = math.hypot(gx - ax, gy - ay) or 1.0
        vx, vy = w.agent_vel
        speed_to_goal = (vx * (gx - ax) + vy * (gy - ay)) / dist / self.phys.agent_max_speed

        sig = StepSignals(box_d=self.box_d, prev_box_d=prev_box_d, path_d=self.path_d, prev_path_d=prev_path,
                          agent_box_d=self.agent_box_d,
                          prev_agent_box_d=prev_ab, agent_speed_to_goal=speed_to_goal,
                          overlap=self.overlap, entered=entered, stable_reached=stable_now,
                          moving=action != 0, wall_hit=wall_hit, hazard_hit=hazard_hit)
        comps = compute(self.reward_cfg, sig)
        reward = float(sum(comps.values()))
        self.ep_return += reward

        # events for replay markers
        ev = self.trace["events"]
        if wall_hit:
            ev.append((self.t, "collision", "goblin hits a wall"))
        if w.box_wall_contact and not was_box_wall:
            ev.append((self.t, "box_bump", "box hits a wall"))
        if entered:
            ev.append((self.t, "enter", "box touches exit"))
        if prev_overlap and not self.overlap:
            ev.append((self.t, "leave", "box leaves exit"))
        if self.inside and not prev_inside:
            ev.append((self.t, "inside", "box fully inside exit"))
        if prev_inside and not self.inside:
            ev.append((self.t, "slip", "box slips out of exit"))
        if stable_now:
            ev.append((self.t, "stable", f"box stable for {self.reward_cfg.stable_frames} frames"))
        if hazard_hit:
            ev.append((self.t, "hazard", "goblin touches lava"))
        for key in ("goal_entry", "stable_goal"):
            if comps.get(key, 0.0) != 0.0:
                ev.append((self.t, "reward", f"reward {comps[key]:+.2f} ({key})"))

        term = self.reward_cfg.termination
        reason = None
        if term["on_hazard"] and self.in_hazard:
            reason = "lava"
        elif term["on_contact"] and self.overlap:
            reason = "exit_contact"
        elif term["on_inside"] and self.inside:
            reason = "exit_inside"
        elif term["on_stable"] and stable_now:
            reason = "exit_stable"
        terminated = reason is not None
        truncated = (not terminated) and self.t >= self.reward_cfg.max_steps
        if truncated:
            reason = "time_limit"

        self._record(action, reward, comps, wall_hit)
        info = {"components": comps, "true_inside": self.inside}
        if terminated or truncated:
            ev.append((self.t, "end", f"episode ends: {reason}"))
            self.trace["termination_reason"] = reason
            self._finish_episode(info)
        return self._obs(), reward, terminated, truncated, info

    # ------------------------------------------------------------------ episode end
    def _finish_episode(self, info: dict) -> None:
        tr = self.trace
        if self.cfg.diagnostics:
            settle_agent, settle_box, ok = [], [], self.inside
            for _ in range(self.cfg.settle_frames):
                self.world.step(0)  # the goblin stops; does the box stay?
                settle_agent.append(self.world.agent_pos)
                settle_box.append(self.world.box_pos)
                ok = ok and self._inside()
            tr["settle_agent"] = [list(map(_r, p)) for p in settle_agent]
            tr["settle_box"] = [list(map(_r, p)) for p in settle_box]
            tr["true_success"] = bool(ok)
            findings = detect_exploits(tr, self.reward_cfg)
            tr["exploits"] = findings
        else:
            tr["true_success"] = None
            tr["exploits"] = []
        tr["return"] = self.ep_return
        tr["length"] = self.t
        self.last_trace = tr
        info["episode_diag"] = {
            "return": self.ep_return,
            "length": self.t,
            "true_success": tr["true_success"],
            "termination_reason": tr["termination_reason"],
            "exploits": [f["code"] for f in tr["exploits"]],
        }

    # ------------------------------------------------------------------ helpers
    def _new_trace(self) -> dict:
        return {"agent": [], "box": [], "action": [], "reward": [], "cum": [], "components": [],
                "overlap": [], "inside": [], "stable": [], "box_wall": [], "wall_hit": [],
                "box_d": [], "agent_goal_d": [], "hazard": [], "events": [],
                "termination_reason": None, "max_steps": self.reward_cfg.max_steps}

    def _record(self, action: int, reward: float, comps: dict, wall_hit: bool) -> None:
        tr = self.trace
        ax, ay = self.world.agent_pos
        tr["agent"].append((ax, ay))
        tr["box"].append(self.world.box_pos)
        tr["action"].append(action)
        tr["reward"].append(reward)
        tr["cum"].append(self.ep_return)
        tr["components"].append(comps)
        tr["overlap"].append(self.overlap)
        tr["inside"].append(self.inside)
        tr["stable"].append(self.stable)
        tr["box_wall"].append(self.world.box_wall_contact)
        tr["wall_hit"].append(wall_hit)
        tr["box_d"].append(self.box_d)
        tr["agent_goal_d"].append(math.hypot(self.goal_c[0] - ax, self.goal_c[1] - ay))
        tr["hazard"].append(self.in_hazard)

    def _box_rect(self):
        return box_rect(self.world.box_pos, self.phys.box_half)

    def _overlap(self) -> bool:
        return rect_overlap(self._box_rect(), self.layout.goal)

    def _inside(self) -> bool:
        return box_inside(self.world.box_pos, self.phys.box_half, self.layout.goal)

    def _hazard(self) -> bool:
        ax, ay = self.world.agent_pos
        return any(circle_rect_overlap(ax, ay, self.phys.agent_radius, h) for h in self.layout.hazards)

    def _box_d(self) -> float:
        bx, by = self.world.box_pos
        return math.hypot(self.goal_c[0] - bx, self.goal_c[1] - by)

    def _path_d(self) -> float:
        return self.field.query(*self.world.box_pos) if self.field is not None else 0.0

    def _agent_box_d(self) -> float:
        ax, ay = self.world.agent_pos
        bx, by = self.world.box_pos
        return math.hypot(bx - ax, by - ay)

    def _obs(self) -> np.ndarray:
        w, lay = self.world, self.layout
        W, H = lay.width, lay.height
        vmax = self.phys.agent_max_speed
        ax, ay = w.agent_pos
        avx, avy = w.agent_vel
        bx, by = w.box_pos
        bvx, bvy = w.box_vel
        gx, gy = self.goal_c
        half = W / 2
        o = [
            ax / W * 2 - 1, ay / H * 2 - 1, avx / vmax, avy / vmax,
            bx / W * 2 - 1, by / H * 2 - 1, bvx / vmax, bvy / vmax,
            gx / W * 2 - 1, gy / H * 2 - 1,
            (bx - ax) / half, (by - ay) / half, (gx - bx) / half, (gy - by) / half,
            self.box_d / math.hypot(W, H),
            float(w.agent_wall_contact), float(self.overlap), float(self.inside),
            min(1.0, self.stable / self.reward_cfg.stable_frames),
            1.0 - self.t / self.reward_cfg.max_steps,
            float(w.box_wall_contact),
        ]
        o.extend(w.rays())
        o.extend(_hazard_rays(ax, ay, lay.hazards))
        return np.clip(np.asarray(o, dtype=np.float32), -5.0, 5.0)


def _hazard_rays(ox: float, oy: float, hazards) -> list[float]:
    out = []
    for i in range(RAY_COUNT):
        ang = 2 * math.pi * i / RAY_COUNT
        dx, dy = math.cos(ang), math.sin(ang)
        best = 1.0
        for h in hazards:
            a = _ray_rect(ox, oy, dx, dy, h)
            if a is not None:
                best = min(best, a / RAY_RANGE)
        out.append(best)
    return out


def _ray_rect(ox, oy, dx, dy, r):
    tmin, tmax = 0.0, RAY_RANGE
    for o, d, lo, hi in ((ox, dx, r[0], r[2]), (oy, dy, r[1], r[3])):
        if abs(d) < 1e-9:
            if o < lo or o > hi:
                return None
        else:
            t1, t2 = (lo - o) / d, (hi - o) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin, tmax = max(tmin, t1), min(tmax, t2)
            if tmin > tmax:
                return None
    return tmin


def _r(v: float) -> float:
    return round(float(v), 3)
