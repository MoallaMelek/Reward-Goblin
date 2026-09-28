import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env

from backend.env.randomization import DIFFICULTIES, NAMED_LAYOUTS, generate_layout, is_solvable
from backend.env.reference_policy import ReferencePolicy
from backend.env.reward_goblin_env import OBS_DIM, RewardGoblinEnv

REWARD = {"components": {"distance_progress": 1.0, "stable_goal": 10.0, "step_penalty": -0.01}}


@pytest.mark.parametrize("layout", NAMED_LAYOUTS + DIFFICULTIES)
def test_gymnasium_checker(layout):
    env = RewardGoblinEnv(REWARD, {"layout": layout})
    check_env(env)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (OBS_DIM,)
    assert env.observation_space.contains(obs)


def test_path_progress_env_passes_checker():
    env = RewardGoblinEnv({"components": {"path_progress": 1.0}}, {"layout": "walled_exit"})
    check_env(env)


def _rollout(seed, actions, layout="medium"):
    env = RewardGoblinEnv(REWARD, {"layout": layout})
    obs, _ = env.reset(seed=seed)
    out = [obs]
    for a in actions:
        obs, r, term, trunc, _ = env.step(a)
        out.append(obs)
        if term or trunc:
            break
    return np.array(out), env.layout


def test_reset_is_deterministic_for_a_seed():
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 5, size=120).tolist()
    a, la = _rollout(7, actions)
    b, lb = _rollout(7, actions)
    assert la.to_dict() == lb.to_dict()
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_layouts():
    layouts = {tuple(generate_layout("easy", np.random.default_rng(s)).box_spawn) for s in range(10)}
    assert len(layouts) == 10


@pytest.mark.parametrize("spec", DIFFICULTIES)
def test_generated_layouts_are_solvable(spec):
    for s in range(15):
        assert is_solvable(generate_layout(spec, np.random.default_rng(s)))


def test_truncates_at_max_steps():
    env = RewardGoblinEnv({**REWARD, "termination": {"max_steps": 50, "on_stable": True}}, {"layout": "arena"})
    env.reset(seed=0)
    for t in range(50):
        _, _, term, trunc, info = env.step(0)
        if t < 49:
            assert not (term or trunc)
    assert trunc and not term
    assert info["episode_diag"]["termination_reason"] == "time_limit"


def _place_box(env, pos):
    env.world.box.position = pos
    env.world.box.velocity = (0, 0)
    env.world.space.reindex_shapes_for_body(env.world.box)


def test_terminate_on_contact_and_inside():
    cfg = {"components": {"goal_contact": 1.0}, "termination": {"on_contact": True, "on_stable": False}}
    env = RewardGoblinEnv(cfg, {"layout": "arena", "randomize": False})
    env.reset(seed=0)
    g = env.layout.goal
    _place_box(env, (g[0] - 0.2, (g[1] + g[3]) / 2))  # straddling the left edge
    _, r, term, _, info = env.step(0)
    assert term and info["episode_diag"]["termination_reason"] == "exit_contact"
    assert r == pytest.approx(1.0)

    cfg = {"components": {"goal_contact": 1.0}, "termination": {"on_inside": True, "on_stable": False}}
    env = RewardGoblinEnv(cfg, {"layout": "arena", "randomize": False})
    env.reset(seed=0)
    _place_box(env, (g[0] - 0.2, (g[1] + g[3]) / 2))
    _, _, term, _, _ = env.step(0)
    assert not term  # touching is not inside
    _place_box(env, ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2))
    _, _, term, _, info = env.step(0)
    assert term and info["episode_diag"]["termination_reason"] == "exit_inside"
    assert info["episode_diag"]["true_success"] is True


def test_stable_goal_pays_once_after_required_frames():
    cfg = {"components": {"stable_goal": 10.0}, "params": {"stable_goal_frames": 5},
           "termination": {"on_stable": False, "max_steps": 50}}
    env = RewardGoblinEnv(cfg, {"layout": "arena", "randomize": False})
    env.reset(seed=0)
    g = env.layout.goal
    _place_box(env, ((g[0] + g[2]) / 2, (g[1] + g[3]) / 2))
    rewards = [env.step(0)[1] for _ in range(20)]
    assert rewards[4] == pytest.approx(10.0)
    assert sum(rewards) == pytest.approx(10.0)


def test_hazard_terminates():
    cfg = {"components": {"hazard_penalty": -1.0}, "termination": {"on_hazard": True}}
    env = RewardGoblinEnv(cfg, {"layout": "lava_room", "randomize": False})
    env.reset(seed=0)
    h = env.layout.hazards[0]
    env.world.agent.position = ((h[0] + h[2]) / 2, (h[1] + h[3]) / 2)
    _, r, term, _, info = env.step(0)
    assert term and r == pytest.approx(-1.0)
    assert info["episode_diag"]["termination_reason"] == "lava"


def test_reference_policy_solves_the_default_map():
    env = RewardGoblinEnv(REWARD, {"layout": "arena"})
    pol = ReferencePolicy(env)
    wins = 0
    for s in range(5):
        env.reset(seed=100 + s)
        done = False
        while not done:
            _, _, term, trunc, info = env.step(pol.act())
            done = term or trunc
        wins += info["episode_diag"]["true_success"]
    assert wins >= 4
