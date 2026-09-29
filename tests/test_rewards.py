import pytest

from backend.env.reward_goblin_env import RewardGoblinEnv
from backend.env.rewards import COMPONENTS, RewardConfig, StepSignals, compute, lint, validate


def sig(**kw):
    base = dict(box_d=2.0, prev_box_d=2.5, path_d=3.0, prev_path_d=3.4, agent_box_d=1.0, prev_agent_box_d=1.5,
                agent_speed_to_goal=0.5, overlap=False, entered=False, stable_reached=False, moving=True,
                wall_hit=False, hazard_hit=False)
    base.update(kw)
    return StepSignals(**base)


def cfg(**components):
    return RewardConfig(components=components)


def test_distance_progress_is_signed():
    assert compute(cfg(distance_progress=2.0), sig())["distance_progress"] == pytest.approx(1.0)
    assert compute(cfg(distance_progress=2.0), sig(box_d=3.0))["distance_progress"] == pytest.approx(-1.0)


def test_closer_bonus_never_penalises_retreat():
    assert compute(cfg(closer_bonus=1.0), sig())["closer_bonus"] == pytest.approx(0.5)
    assert compute(cfg(closer_bonus=1.0), sig(box_d=3.0))["closer_bonus"] == 0.0


def test_path_progress_and_agent_box_progress():
    out = compute(cfg(path_progress=1.0, agent_box_progress=0.5), sig())
    assert out["path_progress"] == pytest.approx(0.4)
    assert out["agent_box_progress"] == pytest.approx(0.25)


def test_event_and_per_step_components():
    c = cfg(goal_contact=0.1, goal_entry=5.0, stable_goal=10.0, survival=0.2, step_penalty=-0.01,
            movement_penalty=-0.005, collision_penalty=-0.1, hazard_penalty=-1.0, approach_speed=0.3)
    quiet = compute(c, sig(moving=False, agent_speed_to_goal=-0.4))
    assert quiet == pytest.approx({"goal_contact": 0.0, "goal_entry": 0.0, "stable_goal": 0.0, "survival": 0.2,
                                   "step_penalty": -0.01, "movement_penalty": 0.0, "collision_penalty": 0.0,
                                   "hazard_penalty": 0.0, "approach_speed": 0.0})
    busy = compute(c, sig(overlap=True, entered=True, stable_reached=True, wall_hit=True, hazard_hit=True))
    assert busy["goal_contact"] == 0.1 and busy["goal_entry"] == 5.0 and busy["stable_goal"] == 10.0
    assert busy["movement_penalty"] == -0.005 and busy["collision_penalty"] == -0.1 and busy["hazard_penalty"] == -1.0
    assert busy["approach_speed"] == pytest.approx(0.15)


def test_proximity_uses_radius_param():
    c = RewardConfig(components={"proximity": 1.0}, params={"proximity_radius": 2.0})
    assert compute(c, sig(box_d=1.9))["proximity"] == 1.0
    assert compute(c, sig(box_d=2.1))["proximity"] == 0.0


def test_potential_based_shaping_telescopes_over_an_episode():
    env = RewardGoblinEnv({"components": {"distance_progress": 1.0}, "termination": {"max_steps": 120, "on_stable": False}},
                          {"layout": "arena"})
    env.reset(seed=4)
    d0 = env.box_d
    total = 0.0
    for t in range(120):
        _, r, term, trunc, _ = env.step([2, 2, 3, 0, 1][t % 5])
        total += r
        if term or trunc:
            break
    assert total == pytest.approx(d0 - env.box_d, abs=1e-9)


def test_validate_accepts_flat_and_structured_forms():
    flat = {"distance_progress": 1.0, "goal_contact": 5.0, "stable_goal": 10.0, "stable_goal_frames": 30,
            "step_penalty": -0.01, "collision_penalty": -0.1}
    errors, _ = validate(flat)
    assert errors == []
    c = RewardConfig.from_dict(flat)
    assert c.components["goal_contact"] == 5.0 and c.stable_frames == 30
    assert RewardConfig.from_dict(c.to_dict()).fingerprint() == c.fingerprint()


@pytest.mark.parametrize("bad, fragment", [
    ({"components": {"python_exec": 1.0}}, "unknown reward component"),
    ({"components": {"goal_contact": "import os"}}, "finite number"),
    ({"components": {"goal_contact": 1e9}}, "within"),
    ({"components": {"goal_contact": True}}, "finite number"),
    ({"components": {"goal_contact": 1.0}, "params": {"stable_goal_frames": 0}}, "stable_goal_frames"),
    ({"components": {"goal_contact": 1.0}, "termination": {"max_steps": 5}}, "max_steps"),
    ({"components": {"goal_contact": 1.0}, "termination": {"on_inside": "yes"}}, "true/false"),
    ({"components": {}}, "at least one"),
    ({"components": {"goal_contact": 1.0}, "eval": "rm -rf"}, "unknown field"),
    ("not a dict", "JSON object"),
])
def test_validate_rejects_bad_configs(bad, fragment):
    errors, _ = validate(bad)
    assert any(fragment in e for e in errors), errors
    if isinstance(bad, dict):
        with pytest.raises(ValueError):
            RewardConfig.from_dict(bad)


def test_lint_flags_known_loopholes():
    codes = lambda **c: {w["code"] for w in lint(RewardConfig(**c))}  # noqa: E731
    assert "positive_only_progress" in codes(components={"closer_bonus": 1.0})
    assert "repeatable_event" in codes(components={"goal_entry": 5.0})
    assert "finish_ends_income" in codes(components={"goal_contact": 0.1}, termination={"on_inside": True})
    assert "stalling_pays" in codes(components={"survival": 0.1, "stable_goal": 5.0})
    assert "early_exit_cheaper" in codes(components={"step_penalty": -0.05, "hazard_penalty": -1.0})
    assert "euclidean_through_walls" in codes(components={"proximity": 0.1})
    honest = codes(components={"path_progress": 1.0, "stable_goal": 10.0, "step_penalty": -0.01, "hazard_penalty": -10.0})
    assert not honest & {"positive_only_progress", "repeatable_event", "finish_ends_income", "stalling_pays",
                         "early_exit_cheaper", "no_completion_reward"}


def test_lava_warning_only_on_maps_with_lava():
    c = RewardConfig(components={"step_penalty": -0.05, "hazard_penalty": -1.0})
    assert "early_exit_cheaper" in {w["code"] for w in lint(c, "lava_room")}
    assert "early_exit_cheaper" not in {w["code"] for w in lint(c, "arena")}


def test_catalogue_metadata_is_complete():
    for k, v in COMPONENTS.items():
        assert {"label", "group", "default", "unit", "description"} <= v.keys(), k
