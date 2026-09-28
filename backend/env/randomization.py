"""Arena layouts: hand-authored gallery maps plus procedural easy / medium / adversarial suites."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .physics import Rect, rect_overlap
from .planning import agent_grid, box_grid

WIDTH, HEIGHT = 10.0, 7.0
GOAL_SIZE = 1.6
BOX_HALF = 0.4
AGENT_R = 0.3
WALL_T = 0.3
# Box needs this much free space around it so the goblin can get behind it on any side.
BOX_CLEAR = BOX_HALF + 2 * AGENT_R + 0.1

DIFFICULTIES = ("easy", "medium", "adversarial")
NAMED_LAYOUTS = ("arena", "exit_pad", "walled_exit", "lava_room")


@dataclass
class Layout:
    name: str
    width: float
    height: float
    walls: list[Rect]          # interior static walls (boundary is implicit)
    obstacles: list[Rect]      # static blocks
    hazards: list[Rect]        # lava: non-physical, touching it can end the episode
    goal: Rect
    agent_spawn: tuple[float, float]
    box_spawn: tuple[float, float]
    meta: dict = field(default_factory=dict)

    @property
    def static_rects(self) -> list[Rect]:
        """All solid geometry, including the arena boundary."""
        w, h, t = self.width, self.height, WALL_T
        boundary = [(-t, -t, w + t, 0.0), (-t, h, w + t, h + t), (-t, 0.0, 0.0, h), (w, 0.0, w + t, h)]
        return boundary + list(self.walls) + list(self.obstacles)

    @property
    def interior_solids(self) -> list[Rect]:
        return list(self.walls) + list(self.obstacles)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["walls"] = [list(map(_r3, r)) for r in self.walls]
        d["obstacles"] = [list(map(_r3, r)) for r in self.obstacles]
        d["hazards"] = [list(map(_r3, r)) for r in self.hazards]
        d["goal"] = list(map(_r3, self.goal))
        d["agent_spawn"] = list(map(_r3, self.agent_spawn))
        d["box_spawn"] = list(map(_r3, self.box_spawn))
        return d


def _r3(v: float) -> float:
    return round(float(v), 3)


def _goal_rect(cx: float, cy: float, size: float = GOAL_SIZE) -> Rect:
    return (cx - size / 2, cy - size / 2, cx + size / 2, cy + size / 2)


# --------------------------------------------------------------------------- named maps
def named_layout(name: str, rng: np.random.Generator | None, jitter: bool) -> Layout:
    j = (lambda a: float(rng.uniform(-a, a))) if (jitter and rng is not None) else (lambda a: 0.0)
    if name == "arena":
        gx, gy = 7.5 + j(0.4), 3.5 + j(1.0)
        return Layout("arena", WIDTH, HEIGHT, [], [], [], _goal_rect(gx, gy),
                      (1.6 + j(0.5), 3.5 + j(2.0)), (4.0 + j(0.8), 3.5 + j(1.3)))
    if name == "exit_pad":
        # Exit in open floor, box spawned close by: pushing through it and back is physically easy.
        gx, gy = 5.6 + j(0.8), 3.5 + j(0.9)
        if rng is not None and jitter:
            ang = float(rng.uniform(0, 2 * math.pi))
            dist = float(rng.uniform(1.9, 2.6))
        else:
            ang, dist = math.pi, 2.2
        bx = min(max(gx + dist * math.cos(ang), BOX_CLEAR), WIDTH - BOX_CLEAR)
        by = min(max(gy + dist * math.sin(ang), BOX_CLEAR), HEIGHT - BOX_CLEAR)
        ax = bx - 1.3 if bx > 2.0 else bx + 1.3
        return Layout("exit_pad", WIDTH, HEIGHT, [], [], [], _goal_rect(gx, gy), (ax, by + j(0.8)), (bx, by))
    if name == "walled_exit":
        # The exit sits inside a room whose only door faces away from the box.
        walls = [(6.8, 2.0, 7.0, 5.0),    # west wall (faces the box)
                 (6.8, 2.0, 10.0, 2.2),   # floor of the room
                 (6.8, 4.8, 8.2, 5.0)]    # roof, leaving a door at x in [8.2, 10]
        goal = (7.7, 2.2, 9.3, 3.8)
        return Layout("walled_exit", WIDTH, HEIGHT, walls, [], [], goal,
                      (1.5 + j(0.5), 3.5 + j(1.5)), (3.6 + j(0.6), 3.6 + j(1.0)))
    if name == "lava_room":
        hazards = [(5.0, 0.0, 6.0, 2.3), (5.0, 4.7, 6.0, HEIGHT)]
        return Layout("lava_room", WIDTH, HEIGHT, [], [], hazards, _goal_rect(7.8 + j(0.3), 3.5 + j(0.6)),
                      (1.5 + j(0.5), 3.5 + j(1.8)), (3.5 + j(0.5), 3.5 + j(1.0)))
    raise ValueError(f"unknown layout {name!r}")


# --------------------------------------------------------------------------- procedural
def generate_layout(spec: str, rng: np.random.Generator, jitter: bool = True) -> Layout:
    if spec in NAMED_LAYOUTS:
        return named_layout(spec, rng, jitter)
    if spec not in DIFFICULTIES:
        raise ValueError(f"unknown layout/difficulty {spec!r}")
    for _ in range(200):
        if spec == "easy":
            lay = _easy(rng)
        elif spec == "medium":
            lay = _medium(rng)
        else:
            lay = _adversarial(rng)
        if lay is not None and is_solvable(lay):
            return lay
    # Extremely unlikely; fall back to the canonical map so reset never fails.
    return named_layout("arena", rng, jitter)


def _spawns(rng, lay_solids: list[Rect], hazards: list[Rect], goal: Rect, box_region, min_goal_dist=2.4):
    gx, gy = (goal[0] + goal[2]) / 2, (goal[1] + goal[3]) / 2
    for _ in range(60):
        bx = float(rng.uniform(*box_region[0]))
        by = float(rng.uniform(*box_region[1]))
        if math.hypot(bx - gx, by - gy) < min_goal_dist:
            continue
        bclear = (bx - BOX_CLEAR, by - BOX_CLEAR, bx + BOX_CLEAR, by + BOX_CLEAR)
        if bx < BOX_CLEAR or by < BOX_CLEAR or bx > WIDTH - BOX_CLEAR or by > HEIGHT - BOX_CLEAR:
            continue
        if any(rect_overlap(bclear, r) for r in lay_solids + hazards):
            continue
        if rect_overlap((bx - BOX_HALF, by - BOX_HALF, bx + BOX_HALF, by + BOX_HALF), goal):
            continue
        for _ in range(40):
            ax = float(rng.uniform(0.5, WIDTH - 0.5))
            ay = float(rng.uniform(0.5, HEIGHT - 0.5))
            if math.hypot(ax - bx, ay - by) < 1.2:
                continue
            ar = (ax - 0.45, ay - 0.45, ax + 0.45, ay + 0.45)
            if any(rect_overlap(ar, r) for r in lay_solids + hazards):
                continue
            if rect_overlap(ar, goal):
                continue
            return (ax, ay), (bx, by)
    return None


def _easy(rng) -> Layout | None:
    gx = float(rng.uniform(5.8, 8.8))
    gy = float(rng.uniform(1.2, HEIGHT - 1.2))
    goal = _goal_rect(gx, gy)
    sp = _spawns(rng, [], [], goal, ((1.2, 6.2), (1.2, HEIGHT - 1.2)))
    if sp is None:
        return None
    return Layout("easy", WIDTH, HEIGHT, [], [], [], goal, sp[0], sp[1], {"difficulty": "easy"})


def _medium(rng) -> Layout | None:
    base = _easy(rng)
    if base is None:
        return None
    obstacles: list[Rect] = []
    for _ in range(int(rng.integers(1, 4))):
        for _ in range(30):
            w, h = float(rng.uniform(0.4, 1.4)), float(rng.uniform(0.4, 2.0))
            x0 = float(rng.uniform(0.8, WIDTH - 0.8 - w))
            y0 = float(rng.uniform(0.0, HEIGHT - h))
            r = (x0, y0, x0 + w, y0 + h)
            g = base.goal
            if rect_overlap(r, (g[0] - 1.0, g[1] - 1.0, g[2] + 1.0, g[3] + 1.0)):
                continue
            obstacles.append(r)
            break
    sp = _spawns(rng, obstacles, [], base.goal, ((1.2, 6.2), (1.2, HEIGHT - 1.2)))
    if sp is None:
        return None
    return Layout("medium", WIDTH, HEIGHT, [], obstacles, [], base.goal, sp[0], sp[1],
                  {"difficulty": "medium"})


def _adversarial(rng) -> Layout | None:
    kind = ("room", "wall_goal", "lava_gap")[int(rng.integers(0, 3))]
    if kind == "room":
        # Exit inside a 3-sided room; the open side faces up or down, never toward the box.
        x0 = float(rng.uniform(6.0, 7.2))
        up = bool(rng.integers(0, 2))
        wall_x = (x0, 1.6, x0 + 0.2, 5.4)
        if up:
            floor = (x0, 1.6, WIDTH, 1.8)
            roof = (x0, 5.2, x0 + 1.4 + float(rng.uniform(0, 0.4)), 5.4)
            goal = (x0 + 1.0, 1.8, x0 + 1.0 + GOAL_SIZE, 1.8 + GOAL_SIZE)
        else:
            floor = (x0, 5.2, WIDTH, 5.4)
            roof = (x0, 1.6, x0 + 1.4 + float(rng.uniform(0, 0.4)), 1.8)
            goal = (x0 + 1.0, 5.2 - GOAL_SIZE, x0 + 1.0 + GOAL_SIZE, 5.2)
        if goal[2] > WIDTH - 0.05:
            return None
        walls = [wall_x, floor, roof]
        sp = _spawns(rng, walls, [], goal, ((1.2, x0 - 1.2), (1.2, HEIGHT - 1.2)))
        if sp is None:
            return None
        return Layout("adversarial", WIDTH, HEIGHT, walls, [], [], goal, sp[0], sp[1],
                      {"difficulty": "adversarial", "kind": kind})
    if kind == "wall_goal":
        # Exit flush against the arena boundary with a pillar partially shielding it.
        side = int(rng.integers(0, 3))
        if side == 0:
            gy0 = 2.8 + float(rng.uniform(-1.4, 1.4))
            goal = (WIDTH - GOAL_SIZE, gy0, WIDTH, gy0 + GOAL_SIZE)
        elif side == 1:
            gx = float(rng.uniform(5.5, 8.5))
            goal = (gx, HEIGHT - GOAL_SIZE, gx + GOAL_SIZE, HEIGHT)
        else:
            gx = float(rng.uniform(5.5, 8.5))
            goal = (gx, 0.0, gx + GOAL_SIZE, GOAL_SIZE)
        gcx, gcy = (goal[0] + goal[2]) / 2, (goal[1] + goal[3]) / 2
        px = gcx - 2.2 + float(rng.uniform(-0.3, 0.3))
        pillar = (px - 0.25, gcy - 0.9, px + 0.25, gcy + 0.9)
        pillar = (pillar[0], max(pillar[1], 0.0), pillar[2], min(pillar[3], HEIGHT))
        sp = _spawns(rng, [pillar], [], goal, ((1.2, 5.5), (1.2, HEIGHT - 1.2)))
        if sp is None:
            return None
        return Layout("adversarial", WIDTH, HEIGHT, [], [pillar], [], goal, sp[0], sp[1],
                      {"difficulty": "adversarial", "kind": kind})
    # lava_gap: a lava river between box and exit with one gap.
    lx = float(rng.uniform(4.6, 5.8))
    gap_y = float(rng.uniform(1.6, HEIGHT - 1.6))
    gap = 1.25 + float(rng.uniform(0, 0.5))
    hazards = [(lx, 0.0, lx + 0.8, gap_y - gap), (lx, gap_y + gap, lx + 0.8, HEIGHT)]
    gx = float(rng.uniform(lx + 2.0, 8.8))
    goal = _goal_rect(gx, float(rng.uniform(1.2, HEIGHT - 1.2)))
    sp = _spawns(rng, [], hazards, goal, ((1.2, lx - 1.2), (1.2, HEIGHT - 1.2)))
    if sp is None:
        return None
    return Layout("adversarial", WIDTH, HEIGHT, [], [], hazards, goal, sp[0], sp[1],
                  {"difficulty": "adversarial", "kind": kind})


def is_solvable(lay: Layout) -> bool:
    """Cheap necessary condition: box can travel to the exit and goblin can reach the box."""
    gcx, gcy = (lay.goal[0] + lay.goal[2]) / 2, (lay.goal[1] + lay.goal[3]) / 2
    if box_grid(lay, BOX_HALF, AGENT_R).astar(lay.box_spawn, (gcx, gcy)) is None:
        return False
    ag = agent_grid(lay, AGENT_R)
    return any(ag.astar(lay.agent_spawn, (lay.box_spawn[0] + dx, lay.box_spawn[1])) is not None
               for dx in (-1.0, 1.0))
