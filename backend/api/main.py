"""FastAPI backend for the Reward Goblin UI.

    uvicorn backend.api.main:app --port 8000     (then open http://localhost:8000)

Reward functions are accepted only as structured component configs (validated against
the fixed catalogue in backend/env/rewards.py). Nothing here executes user code.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..analysis.exploit_detection import LABELS
from ..analysis.metrics import summarize
from ..analysis.true_objective import TRUE_OBJECTIVE
from ..env.randomization import DIFFICULTIES, NAMED_LAYOUTS
from ..env.reward_goblin_env import ENV_VERSION, EnvConfig
from ..env.rewards import COMPONENTS, PARAMS, TERMINATION, RewardConfig, validate
from ..training.evaluate import evaluate_policy, layout_seeds, public
from ..training.train import ALGO_DEFAULTS, HPARAM_LIMITS, ROOT
from .jobs import JobManager
from .store import SAFE_ID, Store

DATA_ROOT = Path(os.environ.get("REWARD_GOBLIN_ROOT", ROOT))
FRONTEND = ROOT / "frontend"
MAX_API_STEPS = 500_000
MAX_SEEDS = 8

store = Store(DATA_ROOT)
jobs = JobManager(DATA_ROOT)


@asynccontextmanager
async def lifespan(app):
    yield
    jobs.shutdown()


app = FastAPI(title="Reward Goblin", version="1.0", lifespan=lifespan)


@app.middleware("http")
async def revalidate(request, call_next):
    # Frontend modules change during development; make browsers revalidate instead of caching.
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-cache")
    return response


class RewardBody(BaseModel):
    reward: dict
    layout: str | None = None


class TrainBody(BaseModel):
    config: dict
    seeds: list[int] = Field(default_factory=lambda: [1, 2, 3])
    steps: int | None = None


class EvalBody(BaseModel):
    run_id: str
    layout: str | None = None
    episodes: int = 20


@app.get("/api/meta")
def meta():
    return {
        "env_version": ENV_VERSION,
        "components": COMPONENTS, "params": PARAMS, "termination": TERMINATION,
        "layouts": list(NAMED_LAYOUTS), "difficulties": list(DIFFICULTIES),
        "algorithms": ALGO_DEFAULTS, "hparam_limits": HPARAM_LIMITS,
        "true_objective": TRUE_OBJECTIVE, "exploit_labels": LABELS,
        "limits": {"max_steps": MAX_API_STEPS, "max_seeds": MAX_SEEDS},
    }


@app.post("/api/rewards/validate")
def validate_reward(body: RewardBody):
    errors, warnings = validate(body.reward, body.layout)
    out = {"ok": not errors, "errors": errors, "warnings": warnings}
    if not errors:
        cfg = RewardConfig.from_dict(body.reward)
        out.update(normalized=cfg.to_dict(), fingerprint=cfg.fingerprint())
    return out


# ------------------------------------------------------------------ experiments & runs
@app.get("/api/experiments")
def experiments():
    items = []
    for e in store.gallery():
        hist = store.version_history(e["id"])
        items.append({**e, "results": [
            {"version": h["version"], "title": h["title"], "seeds": len(h["runs"]),
             "in_distribution": {k: v for k, v in h["suites"].get("in_distribution", {}).items() if k != "per_seed"}}
            for h in hist]})
    user = [x for x in store.experiment_ids() if x not in {e["id"] for e in items}]
    return {"gallery": items, "user_experiments": user}


@app.get("/api/experiments/{exp_id}")
def experiment(exp_id: str):
    e = store.experiment(exp_id)
    if e is None:
        raise HTTPException(404, "experiment not found")
    return e


@app.get("/api/runs")
def runs(experiment: str | None = None):
    return store.list_runs(experiment)


@app.get("/api/runs/{run_id}")
def run(run_id: str):
    m = store.meta(run_id)
    if m is None:
        raise HTTPException(404, "run not found")
    return {"meta": m, "eval": store.eval(run_id), "status": store.status(run_id),
            "recordings": store.recording_kinds(run_id)}


@app.get("/api/runs/{run_id}/metrics")
def run_metrics(run_id: str):
    m = store.metrics(run_id)
    if m is None:
        raise HTTPException(404, "metrics not found")
    return m


@app.get("/api/replays/{run_id}")
def replays(run_id: str, kind: str | None = None):
    if kind is None:
        return {"run_id": run_id, "kinds": store.recording_kinds(run_id)}
    rec = store.recording(run_id, kind)
    if rec is None:
        raise HTTPException(404, "replay not found")
    return rec


# ------------------------------------------------------------------ training
@app.post("/api/train")
def train(body: TrainBody):
    cfg = dict(body.config)
    exp = str(cfg.get("experiment") or "playground")
    if not SAFE_ID.match(exp):
        raise HTTPException(422, "experiment id may only contain letters, digits, '_' and '-'")
    seeds = sorted(set(body.seeds))
    if not 1 <= len(seeds) <= MAX_SEEDS or any(not 0 <= s < 1_000_000 for s in seeds):
        raise HTTPException(422, f"choose between 1 and {MAX_SEEDS} seeds in [0, 1e6)")
    if body.steps is not None:
        cfg["steps"] = body.steps
    if int(cfg.get("steps", 150_000)) > MAX_API_STEPS:
        raise HTTPException(422, f"steps must be <= {MAX_API_STEPS} from the UI (use train.py for longer runs)")
    cfg.setdefault("steps", 150_000)
    cfg["experiment"] = exp
    cfg["version"] = store.next_version(exp)
    try:
        return jobs.submit(cfg, seeds)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/training")
def training_jobs():
    return jobs.list()


@app.get("/api/training/{job_id}")
def training(job_id: str):
    j = jobs.get(job_id)
    if j is None:
        raise HTTPException(404, "job not found")
    return j


@app.get("/api/training/{job_id}/metrics")
def training_metrics(job_id: str):
    m = jobs.metrics(job_id)
    if m is None:
        raise HTTPException(404, "job not found")
    return m


@app.post("/api/evaluate")
def evaluate(body: EvalBody):
    from stable_baselines3 import A2C, PPO

    m = store.meta(body.run_id)
    if m is None or m.get("state") != "done":
        raise HTTPException(404, "finished run not found")
    model_path = DATA_ROOT / "models" / f"{body.run_id}.zip"
    if not model_path.exists():
        raise HTTPException(409, "model weights are not present locally (gallery bundles ship recordings "
                                 "and metrics only); run scripts/build_gallery.py to regenerate them")
    layout = body.layout or m["env"]["layout"]
    if layout not in NAMED_LAYOUTS and layout not in DIFFICULTIES:
        raise HTTPException(422, "unknown layout")
    n = max(1, min(body.episodes, 50))
    reward = RewardConfig.from_dict(m["reward"])
    env_cfg = EnvConfig.from_dict({**m["env"], "layout": layout})
    model = {"PPO": PPO, "A2C": A2C}[m["algo"]["name"]].load(str(model_path), device="cpu")
    eps = evaluate_policy(model, reward, env_cfg, layout_seeds(n, offset=20_000))
    return {"run_id": body.run_id, "layout": layout, "summary": summarize([public(e) for e in eps]),
            "episodes": [public(e) for e in eps]}


# ------------------------------------------------------------------ frontend
@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


app.mount("/", StaticFiles(directory=FRONTEND), name="frontend")
