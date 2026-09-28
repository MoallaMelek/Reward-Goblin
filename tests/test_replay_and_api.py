import importlib
import json

import pytest

from backend.env.reference_policy import ReferencePolicy
from backend.env.reward_goblin_env import RewardGoblinEnv
from backend.env.rewards import RewardConfig
from backend.replay.recorder import build_recording, load_recording, save_recording

REWARD = {"components": {"distance_progress": 1.0, "goal_contact": 0.1, "stable_goal": 10.0}}


def _episode():
    env = RewardGoblinEnv(REWARD, {"layout": "arena"})
    pol = ReferencePolicy(env)
    env.reset(seed=11)
    done = False
    while not done:
        _, _, term, trunc, _ = env.step(pol.act())
        done = term or trunc
    return env


def test_recording_roundtrip(tmp_path):
    env = _episode()
    tr = env.last_trace
    rec = build_recording(tr, env.layout, env.reward_cfg, kind="best_reward", layout_seed=11, reference_trace=tr)
    n = len(tr["reward"])
    f = rec["frames"]
    assert len(f["agent"]) == len(f["box"]) == len(f["action"]) == len(f["cum"]) == n
    assert set(f["components"]) == set(env.reward_cfg.components)
    assert all(len(v) == n for v in f["components"].values())
    assert rec["length"] == n - 1 and rec["true_success"] == tr["true_success"]
    assert len(rec["settle"]["box"]) == 30
    assert rec["checklist"][0]["item"] == "Box enters exit"
    path = tmp_path / "r" / "x.json"
    save_recording(rec, path)
    back = load_recording(path)
    assert back == json.loads(json.dumps(rec))
    # cumulative reward in the recording matches the per-step rewards
    assert back["frames"]["cum"][-1] == pytest.approx(sum(tr["reward"]), abs=0.01)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("REWARD_GOBLIN_ROOT", str(tmp_path))
    import backend.api.main as main
    main = importlib.reload(main)
    return TestClient(main.app)


def test_api_meta_and_validate(client):
    m = client.get("/api/meta").json()
    assert "goal_contact" in m["components"] and "on_stable" in m["termination"]
    ok = client.post("/api/rewards/validate", json={"reward": REWARD}).json()
    assert ok["ok"] and ok["fingerprint"] == RewardConfig.from_dict(REWARD).fingerprint()
    bad = client.post("/api/rewards/validate", json={"reward": {"components": {"__import__": 1}}}).json()
    assert not bad["ok"] and bad["errors"]


def test_api_rejects_bad_training_requests_without_starting(client):
    r = client.post("/api/train", json={"config": {"reward": {"components": {"nope": 1}}}, "seeds": [1]})
    assert r.status_code == 422
    r = client.post("/api/train", json={"config": {"reward": REWARD}, "seeds": list(range(20))})
    assert r.status_code == 422
    r = client.post("/api/train", json={"config": {"reward": REWARD, "experiment": "../etc"}, "seeds": [1]})
    assert r.status_code == 422
    r = client.post("/api/train", json={"config": {"reward": REWARD}, "seeds": [1], "steps": 10_000_000})
    assert r.status_code == 422
    assert client.get("/api/training").json() == []


def test_api_404s(client):
    assert client.get("/api/runs/does_not_exist").status_code == 404
    assert client.get("/api/replays/../x?kind=y").status_code == 404
    assert client.get("/api/experiments/nothing").status_code == 404
    assert client.get("/api/training/nothing").status_code == 404
