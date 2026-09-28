"""Occupancy grids + 4-connected A*.

Used for two things only: checking that procedurally generated layouts are solvable,
and driving the hand-written *reference* controller shown as "what you intended".
The RL agent never sees any of this.
"""
from __future__ import annotations

import heapq
import math

from .physics import Rect

# (rects, clearance, shape) where shape is "square" (box) or "circle" (goblin)
Layer = tuple[list[Rect], float, str]


class Grid:
    def __init__(self, width: float, height: float, layers: list[Layer], boundary: float,
                 soft: tuple[list[Rect], float, float] | None = None, res: float = 0.2):
        """``soft`` = (rects, clearance, penalty): passable cells near these rects (and the
        boundary) cost ``penalty`` extra, which keeps planned box paths away from walls
        where the goblin could not get behind the box."""
        self.res = res
        self.nx = int(round(width / res))
        self.ny = int(round(height / res))
        self.width, self.height = width, height
        self.blocked = [[False] * self.ny for _ in range(self.nx)]
        self.cost = [[0.0] * self.ny for _ in range(self.nx)]
        for i in range(self.nx):
            cx = (i + 0.5) * res
            for j in range(self.ny):
                cy = (j + 0.5) * res
                edge = min(cx, cy, width - cx, height - cy)
                if edge < boundary or any(_inflated_hit(cx, cy, r, c, s) for rects, c, s in layers for r in rects):
                    self.blocked[i][j] = True
                    continue
                if soft is not None:
                    rects, c, pen = soft
                    if edge < c or any(_inflated_hit(cx, cy, r, c, "square") for r in rects):
                        self.cost[i][j] = pen

    def cell(self, x: float, y: float) -> tuple[int, int]:
        i = min(max(int(x / self.res), 0), self.nx - 1)
        j = min(max(int(y / self.res), 0), self.ny - 1)
        return i, j

    def center(self, c: tuple[int, int]) -> tuple[float, float]:
        return ((c[0] + 0.5) * self.res, (c[1] + 0.5) * self.res)

    def free(self, c: tuple[int, int], extra: set | None = None) -> bool:
        i, j = c
        if not (0 <= i < self.nx and 0 <= j < self.ny):
            return False
        if self.blocked[i][j]:
            return False
        return not (extra and c in extra)

    def nearest_free(self, c: tuple[int, int], extra: set | None = None, max_r: int = 6):
        if self.free(c, extra):
            return c
        for r in range(1, max_r + 1):
            best = None
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    n = (c[0] + di, c[1] + dj)
                    if self.free(n, extra):
                        d = di * di + dj * dj
                        if best is None or d < best[0]:
                            best = (d, n)
            if best:
                return best[1]
        return None

    def astar(self, start: tuple[float, float], goal: tuple[float, float],
              extra: set | None = None, turn_cost: float = 0.0):
        """Returns a list of world-space waypoints (cell centres) or None."""
        s = self.nearest_free(self.cell(*start), extra)
        g = self.nearest_free(self.cell(*goal), extra)
        if s is None or g is None:
            return None
        open_heap = [(0.0, 0.0, s, -1)]  # state = (cell, incoming direction) so turns can be penalised
        came: dict = {(s, -1): None}
        cost = {(s, -1): 0.0}
        dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))
        found = None
        while open_heap:
            _, gc, c, d_in = heapq.heappop(open_heap)
            if c == g:
                found = (c, d_in)
                break
            if gc > cost.get((c, d_in), math.inf):
                continue
            for k, (di, dj) in enumerate(dirs):
                n = (c[0] + di, c[1] + dj)
                if not self.free(n, extra):
                    continue
                nc = gc + 1.0 + self.cost[n[0]][n[1]] + (turn_cost if d_in not in (-1, k) else 0.0)
                key = (n, k)
                if nc < cost.get(key, math.inf):
                    cost[key] = nc
                    came[key] = (c, d_in)
                    h = abs(n[0] - g[0]) + abs(n[1] - g[1])
                    heapq.heappush(open_heap, (nc + h, nc, n, k))
        if found is None:
            return None
        path = []
        node = found
        while node is not None:
            path.append(self.center(node[0]))
            node = came[node]
        path.reverse()
        return path


class DistanceField:
    """Walkable distance from every free cell to the exit (Dijkstra, 4-connected).

    ``query`` is continuous in the box position: the minimum over the surrounding 3x3 cells of
    (cell distance + straight-line offset to that cell centre).
    """

    def __init__(self, grid: Grid, goal: tuple[float, float]):
        self.grid = grid
        INF = math.inf
        d = [[INF] * grid.ny for _ in range(grid.nx)]
        g = grid.nearest_free(grid.cell(*goal))
        self.reachable = g is not None
        if g is not None:
            gx, gy = grid.center(g)
            d[g[0]][g[1]] = math.hypot(gx - goal[0], gy - goal[1])
            heap = [(d[g[0]][g[1]], g)]
            while heap:
                dist, c = heapq.heappop(heap)
                if dist > d[c[0]][c[1]]:
                    continue
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    n = (c[0] + di, c[1] + dj)
                    if grid.free(n):
                        nd = dist + grid.res
                        if nd < d[n[0]][n[1]]:
                            d[n[0]][n[1]] = nd
                            heapq.heappush(heap, (nd, n))
        self.d = d
        self.goal = goal

    def query(self, x: float, y: float) -> float:
        g = self.grid
        ci, cj = g.cell(x, y)
        best = math.inf
        for r in (1, 3):  # widen the search if the box is inside an inflated wall margin
            for i in range(ci - r, ci + r + 1):
                for j in range(cj - r, cj + r + 1):
                    if 0 <= i < g.nx and 0 <= j < g.ny and self.d[i][j] < math.inf:
                        cx, cy = g.center((i, j))
                        best = min(best, self.d[i][j] + math.hypot(cx - x, cy - y))
            if best < math.inf:
                return best
        return math.hypot(x - self.goal[0], y - self.goal[1]) + 20.0


def _inflated_hit(cx: float, cy: float, r: Rect, clearance: float, shape: str) -> bool:
    if shape == "square":
        return (r[0] - clearance < cx < r[2] + clearance
                and r[1] - clearance < cy < r[3] + clearance)
    nx = min(max(cx, r[0]), r[2])
    ny = min(max(cy, r[1]), r[3])
    return (cx - nx) ** 2 + (cy - ny) ** 2 < clearance * clearance


def box_grid(layout, box_half: float, agent_r: float) -> Grid:
    solids = layout.interior_solids
    pushable = box_half + 2 * agent_r + 0.1
    return Grid(layout.width, layout.height,
                [(solids, box_half + 0.06, "square"), (layout.hazards, box_half + 0.3, "square")],
                boundary=box_half + 0.06, soft=(solids, pushable, 3.0))


def agent_grid(layout, agent_r: float) -> Grid:
    return Grid(layout.width, layout.height,
                [(layout.interior_solids, agent_r + 0.06, "circle"), (layout.hazards, agent_r + 0.25, "square")],
                boundary=agent_r + 0.06)
