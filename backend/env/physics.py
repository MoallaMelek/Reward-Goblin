"""Thin Pymunk wrapper for the Reward Goblin arena.

Top-down world, no gravity. The goblin is a velocity-servoed circle, the box is a
square with floor drag and bouncy walls. Rotation is locked on both bodies: it keeps
the task learnable in ~100k steps and makes "box fully inside the exit" an exact
axis-aligned test.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pymunk

Rect = tuple[float, float, float, float]  # (x0, y0, x1, y1)

CAT_STATIC = 0b001
CAT_AGENT = 0b010
CAT_BOX = 0b100

# Discrete action space: 0 no-op, 1 left, 2 right, 3 up (+y), 4 down (-y)
ACTION_VECTORS = ((0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
ACTION_NAMES = ("noop", "left", "right", "up", "down")

RAY_COUNT = 8
RAY_RANGE = 5.0


@dataclass(frozen=True)
class PhysicsParams:
    dt: float = 1.0 / 60.0
    substeps: int = 4  # physics steps per env step -> 15 Hz control
    agent_radius: float = 0.3
    agent_mass: float = 1.0
    agent_max_speed: float = 2.5
    agent_response: float = 0.12  # velocity-servo time constant (s)
    agent_max_force: float = 30.0
    box_half: float = 0.4
    box_mass: float = 2.0
    box_floor_drag: float = 2.5  # exponential velocity decay rate (1/s)
    box_elasticity: float = 0.6
    agent_elasticity: float = 0.2
    contact_friction: float = 0.3

    @property
    def step_seconds(self) -> float:
        return self.dt * self.substeps


class _BoxDrag:
    """Picklable velocity_func: exponential floor friction on the box."""

    def __init__(self, rate: float):
        self.rate = rate

    def __call__(self, body, gravity, damping, dt):
        pymunk.Body.update_velocity(body, gravity, math.exp(-self.rate * dt), dt)


class World:
    def __init__(self, width: float, height: float, static_rects: list[Rect],
                 agent_pos: tuple[float, float], box_pos: tuple[float, float],
                 params: PhysicsParams = PhysicsParams()):
        self.p = params
        self.width, self.height = width, height
        space = pymunk.Space()
        space.gravity = (0.0, 0.0)
        space.iterations = 20
        self.space = space

        static_filter = pymunk.ShapeFilter(categories=CAT_STATIC)
        for x0, y0, x1, y1 in static_rects:
            poly = pymunk.Poly(space.static_body, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
            poly.elasticity = 1.0
            poly.friction = 0.4
            poly.filter = static_filter
            space.add(poly)

        self.agent = pymunk.Body(params.agent_mass, float("inf"))
        self.agent.position = agent_pos
        agent_shape = pymunk.Circle(self.agent, params.agent_radius)
        agent_shape.elasticity = params.agent_elasticity
        agent_shape.friction = params.contact_friction
        agent_shape.filter = pymunk.ShapeFilter(categories=CAT_AGENT)
        space.add(self.agent, agent_shape)

        self.box = pymunk.Body(params.box_mass, float("inf"))
        self.box.position = box_pos
        self.box.velocity_func = _BoxDrag(params.box_floor_drag)
        s = params.box_half * 2
        box_shape = pymunk.Poly.create_box(self.box, (s, s))
        box_shape.elasticity = params.box_elasticity
        box_shape.friction = params.contact_friction
        box_shape.filter = pymunk.ShapeFilter(categories=CAT_BOX)
        space.add(self.box, box_shape)

        self._ray_filter = pymunk.ShapeFilter(mask=CAT_STATIC)
        self.agent_wall_contact = False
        self.box_wall_contact = False
        self.agent_box_contact = False

    # ------------------------------------------------------------------ stepping
    def step(self, action: int) -> None:
        ax, ay = ACTION_VECTORS[action]
        p = self.p
        tx, ty = ax * p.agent_max_speed, ay * p.agent_max_speed
        agent_wall = box_wall = agent_box = False
        for _ in range(p.substeps):
            vx, vy = self.agent.velocity
            fx = p.agent_mass * (tx - vx) / p.agent_response
            fy = p.agent_mass * (ty - vy) / p.agent_response
            mag = math.hypot(fx, fy)
            if mag > p.agent_max_force:
                fx, fy = fx * p.agent_max_force / mag, fy * p.agent_max_force / mag
            self.agent.force = (fx, fy)
            self.space.step(p.dt)
            a_static, a_box = self._contacts(self.agent)
            b_static, _ = self._contacts(self.box)
            agent_wall |= a_static
            agent_box |= a_box
            box_wall |= b_static
        self.agent_wall_contact = agent_wall
        self.box_wall_contact = box_wall
        self.agent_box_contact = agent_box

    @staticmethod
    def _contacts(body: pymunk.Body) -> tuple[bool, bool]:
        hit = [False, False]  # [static, other dynamic]

        def visit(arbiter):
            a, b = arbiter.shapes
            other = b if a.body is body else a
            if other.body.body_type == pymunk.Body.STATIC:
                hit[0] = True
            else:
                hit[1] = True

        body.each_arbiter(visit)
        return hit[0], hit[1]

    # ------------------------------------------------------------------ sensing
    def rays(self) -> list[float]:
        """Distance (normalised to [0,1]) from the goblin to static geometry in 8 directions."""
        ox, oy = self.agent.position
        out = []
        for i in range(RAY_COUNT):
            ang = 2 * math.pi * i / RAY_COUNT
            end = (ox + RAY_RANGE * math.cos(ang), oy + RAY_RANGE * math.sin(ang))
            hit = self.space.segment_query_first((ox, oy), end, 0.0, self._ray_filter)
            out.append(hit.alpha if hit is not None else 1.0)
        return out

    @property
    def agent_pos(self) -> tuple[float, float]:
        return tuple(self.agent.position)

    @property
    def box_pos(self) -> tuple[float, float]:
        return tuple(self.box.position)

    @property
    def agent_vel(self) -> tuple[float, float]:
        return tuple(self.agent.velocity)

    @property
    def box_vel(self) -> tuple[float, float]:
        return tuple(self.box.velocity)


def rect_overlap(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def rect_contains(outer: Rect, inner: Rect) -> bool:
    return (inner[0] >= outer[0] and inner[1] >= outer[1]
            and inner[2] <= outer[2] and inner[3] <= outer[3])


def circle_rect_overlap(cx: float, cy: float, r: float, rect: Rect) -> bool:
    nx = min(max(cx, rect[0]), rect[2])
    ny = min(max(cy, rect[1]), rect[3])
    return (cx - nx) ** 2 + (cy - ny) ** 2 < r * r


def box_rect(pos: tuple[float, float], half: float) -> Rect:
    return (pos[0] - half, pos[1] - half, pos[0] + half, pos[1] + half)


def rect_center(r: Rect) -> tuple[float, float]:
    return ((r[0] + r[2]) / 2, (r[1] + r[3]) / 2)
