import math

import pytest

from backend.analysis.exploit_detection import detect_exploits, swing_points
from backend.analysis.metrics import classify_run, summarize
from backend.analysis.true_objective import box_inside, checklist
from backend.env.reward_goblin_env import RewardGoblinEnv
from backend.env.rewards import RewardConfig

GOAL = (6.8, 2.8, 8.2, 4.2)


def test_box_inside_is_strict_containment():
    assert box_inside((7.5, 3.5), 0.4, GOAL)
    assert not box_inside((6.9, 3.5), 0.4, GOAL)  # overlapping the edge
    assert box_inside((7.2, 3.2), 0.4, GOAL)       # exactly flush with two edges


def _place(env, pos, vel=(0.0, 0.0)):
    env.world.box.position = pos
    env.world.box.velocity = vel
    env.world.space.reindex_shapes_for_body(env.world.box)


def _finish(env, pos, vel):
    """Run to the time limit, put the box at pos/vel on the last step, return the diag."""
    for _ in range(49):
        env.step(0)
    _place(env, pos, vel)
    _, _, _, trunc, info = env.step(0)
    assert trunc
    return info["episode_diag"], env.last_trace


def _env():
    env = RewardGoblinEnv({"components": {"step_penalty": -0.01}, "termination": {"max_steps": 50, "on_stable": False}},
                          {"layout": "arena", "randomize": False})
    env.reset(seed=0)
    return env


def test_true_objective_passes_resting_box():
    env = _env()
    g = env.layout.goal
    diag, tr = _finish(env, ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2), (0, 0))
    assert diag["true_success"] is True
    assert len(tr["settle_box"]) == 30


def test_true_objective_fails_box_sliding_through():
    env = _env()
    g = env.layout.goal
    diag, _ = _finish(env, ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2), (3.0, 0.0))
    assert diag["true_success"] is False


def test_true_objective_fails_box_on_edge():
    env = _env()
    g = env.layout.goal
    diag, tr = _finish(env, (g[0], (g[1] + g[3]) / 2), (0, 0))
    assert diag["true_success"] is False
    items = {c["item"]: c["ok"] for c in checklist(tr)}
    assert items["Box remains there"] is False


# --------------------------------------------------------------- detectors on synthetic traces
def trace(n=200, box_d=None, **over):
    box_d = box_d or [3.0 - 0.01 * t for t in range(n)]
    t = {
        "agent": [(1.0, 1.0)] * n, "box": [(2.0, 2.0)] * n, "action": [0] * n, "reward": [0.0] * n,
        "cum": [0.0] * n, "components": [{} for _ in range(n)], "overlap": [False] * n, "inside": [False] * n,
        "stable": [0] * n, "box_wall": [False] * n, "wall_hit": [False] * n, "box_d": box_d,
        "agent_goal_d": [5.0] * n, "hazard": [False] * n, "events": [], "termination_reason": "time_limit",
        "max_steps": 300, "true_success": False,
    }
    t.update(over)
    return t


def codes(tr, cfg=None, ref=None):
    return [f["code"] for f in detect_exploits(tr, cfg, reference_return=ref)]


def test_swing_points_counts_reversals_with_hysteresis():
    x = [0, 1, 0, 1, 0, 1]
    assert len(swing_points(x, 0.5)) == 4
    assert swing_points([0, 0.1, 0, 0.1], 0.5) == []


def test_oscillation_detected_when_closer_bonus_pays_for_it():
    n = 200
    d = [2.0 + 0.5 * math.sin(t / 6) for t in range(n)]
    comps = [{"closer_bonus": max(0.0, d[t - 1] - d[t]) if t else 0.0} for t in range(n)]
    tr = trace(n, box_d=d, components=comps, reward=[c["closer_bonus"] for c in comps])
    assert "oscillation" in codes(tr, RewardConfig(components={"closer_bonus": 1.0}))


def test_contact_farming_from_repeated_entries():
    ev = []
    for k in range(4):
        ev += [(10 + 30 * k, "enter", ""), (20 + 30 * k, "leave", "")]
    assert "contact_farming" in codes(trace(events=ev))


def test_edge_camping_requires_contact_reward():
    n = 200
    over = dict(overlap=[True] * n, box_d=[0.9] * n)
    assert "edge_camping" not in codes(trace(n, **over))
    paid = [{"goal_contact": 0.1} for _ in range(n)]
    assert "edge_camping" in codes(trace(n, components=paid, reward=[0.1] * n, **over))


def test_stalling_and_early_exit():
    n = 200
    assert "stalling" in codes(trace(n, box_d=[1.5] * n, reward=[0.1] * n))
    assert "stalling" not in codes(trace(n, box_d=[1.5] * n, reward=[-0.01] * n))
    tr = trace(40, termination_reason="lava")
    assert "early_exit" in codes(tr, RewardConfig(components={"step_penalty": -0.05}))
    assert "early_exit" not in codes(tr, RewardConfig(components={"hazard_penalty": -1.0}))


def test_outscoring_the_intended_solution_is_flagged_only_on_failure():
    n = 100
    tr = trace(n, reward=[0.5] * n, cum=[0.5 * (t + 1) for t in range(n)])
    assert "outscores_intent" in codes(tr, ref=10.0)
    tr["true_success"] = True
    assert "outscores_intent" not in codes(tr, ref=10.0)


def test_clean_successful_episode_has_no_findings():
    n = 150
    tr = trace(n, box_d=[3.0 - 0.02 * t for t in range(n)], true_success=True,
               inside=[False] * 120 + [True] * 30, termination_reason="exit_stable")
    assert codes(tr, RewardConfig(components={"distance_progress": 1.0, "stable_goal": 10.0})) == []


def test_summaries_and_verdicts():
    eps = [{"return": 10, "length": 100, "true_success": True, "exploits": []},
           {"return": 30, "length": 300, "true_success": False, "exploits": ["stalling"]}]
    s = summarize(eps)
    assert s["true_success_rate"] == 0.5 and s["exploit_rate"] == 0.5 and s["dominant_exploit"] == "stalling"
    assert classify_run({"episodes": 5, "true_success_rate": 0.9, "exploit_rate": 0.0}) == "intended"
    assert classify_run({"episodes": 5, "true_success_rate": 0.0, "exploit_rate": 0.9}) == "exploit"
    assert classify_run({"episodes": 5, "true_success_rate": 0.0, "exploit_rate": 0.1}) == "failure"


def test_random_policy_rarely_triggers_detectors():
    env = RewardGoblinEnv({"components": {"distance_progress": 1.0, "stable_goal": 10.0}}, {"layout": "arena"})
    flagged = 0
    for s in range(10):
        env.reset(seed=s)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(env.action_space.sample())
            done = term or trunc
        flagged += bool(info["episode_diag"]["exploits"])
    assert flagged <= 2
