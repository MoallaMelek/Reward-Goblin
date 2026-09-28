"""The 'What you intended' controller.

A hand-written planner with privileged access to the simulator state. It is NOT
learned and exists for two reasons:
  1. to animate the intended behaviour on the left half of the split screen, and
  2. to measure what the *intended* behaviour would score under a given reward, so we
     can say "the goblin's exploit out-earns doing the task properly".

Strategy: plan a 4-connected A* path for the box to the exit (penalising cells hugging
walls, where the goblin could not get behind the box), walk the goblin with its own A*
to the push position for the current straight segment, push while the predicted glide
still fits in the segment, and stop once the box is inside.
"""
from __future__ import annotations

import math

from .planning import agent_grid, box_grid

_DIRS = {(-1, 0): 1, (1, 0): 2, (0, 1): 3, (0, -1): 4}
LATERAL_TOL = 0.2


class ReferencePolicy:
    def __init__(self, env):
        self.env = env
        self._layout_id = None

    def _prepare(self):
        env = self.env
        p = env.phys
        self.box_grid = box_grid(env.layout, p.box_half, p.agent_radius)
        self.agent_grid = agent_grid(env.layout, p.agent_radius)
        self._layout_id = id(env.layout)
        self._path = None
        self._path_from = None

    def act(self, obs=None) -> int:
        env = self.env
        if self._layout_id != id(env.layout):
            self._prepare()
        p = env.phys
        w = env.world
        bx, by = w.box_pos
        ax, ay = w.agent_pos
        bvx, bvy = w.box_vel
        gx, gy = env.goal_c

        if env.inside:
            return 0  # job done: stop and let it settle

        if self._path is None or math.hypot(bx - self._path_from[0], by - self._path_from[1]) > 0.15:
            self._path = self.box_grid.astar((bx, by), (gx, gy), turn_cost=2.0)
            self._path_from = (bx, by)
        if not self._path:
            return self._direct(ax, ay, bx - 1.0, by)

        u, along = self._segment(self._path, bx, by, gx, gy)
        if u is None:
            return 0
        reach = p.box_half + p.agent_radius + 0.04
        px, py = bx - u[0] * reach, by - u[1] * reach
        glide = max(0.0, bvx * u[0] + bvy * u[1]) / p.box_floor_drag

        lateral = abs((ax - px) * u[1] - (ay - py) * u[0])
        behind = (ax - px) * u[0] + (ay - py) * u[1]
        if lateral < 0.14 and -0.35 < behind < 0.12:
            return 0 if along <= glide + 0.05 else _DIRS[u]
        return self._navigate(ax, ay, px, py, bx, by)

    @staticmethod
    def _segment(path, bx, by, gx, gy):
        """Push direction and how far the box should travel before the next turn."""
        k = min(range(len(path)), key=lambda i: (path[i][0] - bx) ** 2 + (path[i][1] - by) ** 2)
        rest = path[k:]
        u = None
        for q in rest[1:]:
            dx, dy = q[0] - bx, q[1] - by
            if abs(dx) > 0.12 or abs(dy) > 0.12:
                u = (1 if dx > 0 else -1, 0) if abs(dx) >= abs(dy) else (0, 1 if dy > 0 else -1)
                break
        if u is None:  # at the final cell: nudge toward the exact exit centre
            dx, dy = gx - bx, gy - by
            if abs(dx) < 0.1 and abs(dy) < 0.1:
                return None, 0.0
            u = (1 if dx > 0 else -1, 0) if abs(dx) >= abs(dy) else (0, 1 if dy > 0 else -1)
            return u, dx * u[0] + dy * u[1]

        def lat(q):
            return (q[1] - by) if u[0] else (q[0] - bx)

        def along(q):
            return (q[0] - bx) * u[0] + (q[1] - by) * u[1]

        best = 0.0
        for q in rest:
            if abs(lat(q)) > LATERAL_TOL:
                return u, best
            best = max(best, along(q))
        # the straight run reaches the end of the path: aim for the exit centre itself
        return u, along((gx, gy))

    def _navigate(self, ax, ay, px, py, bx, by):
        p = self.env.phys
        clear = p.box_half + p.agent_radius + 0.08
        g = self.agent_grid
        extra = set()
        i0, j0 = g.cell(bx - clear, by - clear)
        i1, j1 = g.cell(bx + clear, by + clear)
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                cx, cy = g.center((i, j))
                if abs(cx - bx) < clear and abs(cy - by) < clear:
                    extra.add((i, j))
        if math.hypot(px - ax, py - ay) < 0.45:
            return self._direct(ax, ay, px, py)
        path = g.astar((ax, ay), (px, py), extra=extra)
        if not path or len(path) < 2:
            return self._direct(ax, ay, px, py)
        tx, ty = path[min(2, len(path) - 1)]
        return self._direct(ax, ay, tx, ty)

    @staticmethod
    def _direct(ax, ay, tx, ty) -> int:
        dx, dy = tx - ax, ty - ay
        if abs(dx) < 0.06 and abs(dy) < 0.06:
            return 0
        if abs(dx) >= abs(dy):
            return 2 if dx > 0 else 1
        return 3 if dy > 0 else 4
