"""Train one goblin (one reward config x one seed), then evaluate it properly.

CLI:
    python train.py --reward configs/touch_goblin_v1.json --seed 42 --steps 200000

Outputs (all keyed by run id):
    runs/<id>/meta.json      config, hyper-parameters, versions, git commit, summary
    runs/<id>/metrics.json   per-rollout training metrics + periodic deterministic evals
    runs/<id>/eval.json      held-out evaluation suites (per episode) + scripted-reference scores
    models/<id>.zip          final policy (+ models/<id>_ckpt/ checkpoints)
    recordings/<id>/*.json   highlighted episode replays
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from ..analysis.metrics import classify_run, summarize
from ..env.reward_goblin_env import ENV_VERSION, EnvConfig, RewardGoblinEnv
from ..env.rewards import RewardConfig
from ..replay.recorder import build_recording, save_recording
from .callbacks import MetricsCallback, write_json_atomic
from .evaluate import evaluate_policy, highlights, layout_seeds, public, reference_summary
from .seeds import seed_everything

ROOT = Path(__file__).resolve().parents[2]

ALGO_DEFAULTS = {
    "PPO": {"learning_rate": 3e-4, "n_steps": 512, "batch_size": 256, "n_epochs": 10, "gamma": 0.99,
            "gae_lambda": 0.95, "ent_coef": 0.01, "clip_range": 0.2, "n_envs": 8, "net_arch": [64, 64]},
    "A2C": {"learning_rate": 7e-4, "n_steps": 16, "gamma": 0.99, "gae_lambda": 0.95, "ent_coef": 0.01,
            "n_envs": 8, "net_arch": [64, 64]},
}
HPARAM_LIMITS = {
    "learning_rate": (1e-6, 1e-2), "n_steps": (8, 4096), "batch_size": (16, 8192), "n_epochs": (1, 50),
    "gamma": (0.8, 0.9999), "gae_lambda": (0.5, 1.0), "ent_coef": (0.0, 0.5), "clip_range": (0.01, 1.0),
    "n_envs": (1, 32),
}
DEFAULT_STEPS = 200_000
EVAL_EPISODES = 20


def normalize_config(cfg: dict) -> dict:
    """Accept a full experiment config or a bare reward config; fill defaults; validate."""
    if "reward" not in cfg:
        cfg = {"reward": cfg}
    reward = RewardConfig.from_dict(cfg["reward"])
    env = EnvConfig.from_dict(cfg.get("env"))
    algo_in = dict(cfg.get("algo") or {})
    name = str(algo_in.pop("name", "PPO")).upper()
    if name not in ALGO_DEFAULTS:
        raise ValueError(f"unsupported algorithm {name!r} (PPO, A2C)")
    algo = {"name": name, **ALGO_DEFAULTS[name]}
    for k, v in algo_in.items():
        if k not in algo:
            raise ValueError(f"unknown hyper-parameter {k!r} for {name}")
        if k in HPARAM_LIMITS:
            lo, hi = HPARAM_LIMITS[k]
            if not isinstance(v, (int, float)) or isinstance(v, bool) or not lo <= v <= hi:
                raise ValueError(f"{k} must be within [{lo}, {hi}]")
        algo[k] = v
    for k in ("n_steps", "batch_size", "n_epochs", "n_envs"):
        if k in algo:
            algo[k] = int(algo[k])
    steps = int(cfg.get("steps", DEFAULT_STEPS))
    if not 1_000 <= steps <= 3_000_000:
        raise ValueError("steps must be within [1000, 3000000]")
    return {
        "experiment": str(cfg.get("experiment", "playground")),
        "version": int(cfg.get("version", 1)),
        "title": str(cfg.get("title", reward.name)),
        "description": str(cfg.get("description", "")),
        "reward": reward.to_dict(),
        "env": env.to_dict(),
        "algo": algo,
        "steps": steps,
        "eval": cfg.get("eval") or default_eval_suites(env.layout),
    }


def default_eval_suites(layout: str) -> list[dict]:
    """Held-out start states on the training map, plus unseen procedural layouts."""
    stress = {"easy": "medium", "medium": "adversarial", "adversarial": "adversarial"}.get(layout, "medium")
    return [{"name": "in_distribution", "layout": layout, "episodes": EVAL_EPISODES, "offset": 0},
            {"name": "shifted", "layout": stress, "episodes": EVAL_EPISODES, "offset": 5_000}]


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def build_model(algo: dict, venv, seed: int):
    from stable_baselines3 import A2C, PPO

    kw = {k: v for k, v in algo.items() if k not in ("name", "n_envs", "net_arch")}
    policy_kwargs = {"net_arch": list(algo.get("net_arch", [64, 64]))}
    cls = PPO if algo["name"] == "PPO" else A2C
    return cls("MlpPolicy", venv, seed=seed, verbose=0, device="cpu", policy_kwargs=policy_kwargs, **kw)


def run_id_for(cfg: dict, seed: int, root: Path) -> str:
    base = f"{cfg['experiment']}_v{cfg['version']}_seed{seed}"
    rid, k = base, 2
    while (root / "runs" / rid).exists():
        rid, k = f"{base}_{k}", k + 1
    return rid


def train_run(cfg: dict, seed: int, *, root: Path = ROOT, run_id: str | None = None,
              status_extra: dict | None = None, log=print) -> dict:
    import stable_baselines3
    import torch
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    from stable_baselines3.common.env_util import make_vec_env

    cfg = normalize_config(cfg)
    run_id = run_id or run_id_for(cfg, seed, root)
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (root / "models").mkdir(exist_ok=True)
    seed_everything(seed)
    reward = RewardConfig.from_dict(cfg["reward"])
    env_cfg = EnvConfig.from_dict(cfg["env"])
    algo = cfg["algo"]

    meta = {
        "run_id": run_id, "experiment": cfg["experiment"], "version": cfg["version"], "title": cfg["title"],
        "description": cfg["description"], "seed": seed, "steps": cfg["steps"],
        "reward": reward.to_dict(), "reward_fingerprint": reward.fingerprint(), "env": env_cfg.to_dict(),
        "algo": algo, "eval_suites": cfg["eval"], "env_version": ENV_VERSION, "git_commit": git_commit(),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "stable_baselines3": stable_baselines3.__version__},
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "state": "training",
    }
    write_json_atomic(run_dir / "meta.json", meta)

    venv = make_vec_env(lambda: RewardGoblinEnv(reward, env_cfg), n_envs=algo["n_envs"], seed=seed)
    model = build_model(algo, venv, seed)

    probe_seeds = layout_seeds(8, offset=50_000)

    def probe(m):
        eps = evaluate_policy(m, reward, env_cfg, probe_seeds, with_reference=False)
        s = summarize([public(e) for e in eps])
        return {k: s[k] for k in ("mean_return", "true_success_rate", "exploit_rate", "mean_length")}

    extra = {"run_id": run_id, **(status_extra or {})}
    metrics_cb = MetricsCallback(run_dir, cfg["steps"], eval_fn=probe, eval_points=5, status_extra=extra)
    ckpt_dir = root / "models" / f"{run_id}_ckpt"
    ckpt_every = max(cfg["steps"] // 4 // algo["n_envs"], 1)
    ckpt_cb = CheckpointCallback(save_freq=ckpt_every, save_path=str(ckpt_dir), name_prefix="ckpt")
    t0 = time.time()
    log(f"[{run_id}] training {algo['name']} for {cfg['steps']} steps")
    model.learn(total_timesteps=cfg["steps"], callback=CallbackList([metrics_cb, ckpt_cb]))
    train_s = time.time() - t0
    model.save(str(root / "models" / f"{run_id}.zip"))
    metrics_cb.flush(state="evaluating")

    log(f"[{run_id}] evaluating")
    eval_out = {"suites": {}, "reference": {}, "highlights": {}}
    for suite in cfg["eval"]:
        s_env = EnvConfig.from_dict({**env_cfg.to_dict(), "layout": suite["layout"]})
        seeds = layout_seeds(suite["episodes"], suite.get("offset", 0))
        eps = evaluate_policy(model, reward, s_env, seeds)
        summary = summarize([public(e) for e in eps])
        eval_out["suites"][suite["name"]] = {"layout": suite["layout"], "seeds": seeds, "summary": summary,
                                             "episodes": [public(e) for e in eps]}
        eval_out["reference"][suite["name"]] = reference_summary(reward, s_env, seeds)
        picks = highlights(eps) if suite["name"] == "in_distribution" else \
            {k: v for k, v in highlights(eps).items() if k in ("worst_exploit", "best_true_success")}
        for kind, ep in picks.items():
            name = kind if suite["name"] == "in_distribution" else f"{suite['name']}_{kind}"
            rec = build_recording(ep["_trace"], ep["_layout"], reward, kind=name, layout_seed=ep["layout_seed"],
                                  reference_trace=ep["_ref"],
                                  extra={"run_id": run_id, "suite": suite["name"], "seed": seed})
            save_recording(rec, root / "recordings" / run_id / f"{name}.json")
            eval_out["highlights"][name] = {"file": f"{name}.json", **public(ep)}
    write_json_atomic(run_dir / "eval.json", eval_out)

    ind = eval_out["suites"]["in_distribution"]["summary"]
    meta.update({
        "state": "done", "train_seconds": round(train_s, 1), "total_seconds": round(time.time() - t0, 1),
        "summary": {name: s["summary"] for name, s in eval_out["suites"].items()},
        "reference": eval_out["reference"],
        "verdict": classify_run(ind),
        "final_train": metrics_cb.series[-1] if metrics_cb.series else None,
    })
    write_json_atomic(run_dir / "meta.json", meta)
    metrics_cb.flush(state="done")
    log(f"[{run_id}] done in {meta['total_seconds']}s: return {ind['mean_return']:.2f}, "
        f"true success {ind['true_success_rate']:.0%}, exploit rate {ind['exploit_rate']:.0%}, "
        f"dominant {ind['dominant_exploit']}")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train a Reward Goblin policy.")
    ap.add_argument("--reward", required=True, help="experiment or reward config JSON")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--algo", default=None, choices=["PPO", "A2C", "ppo", "a2c"])
    ap.add_argument("--layout", default=None, help="override env layout/difficulty")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    cfg = json.loads(Path(args.reward).read_text())
    if "reward" not in cfg:
        cfg = {"reward": cfg, "experiment": Path(args.reward).stem}
    if args.steps:
        cfg["steps"] = args.steps
    if args.algo:
        cfg.setdefault("algo", {})["name"] = args.algo.upper()
    if args.layout:
        cfg.setdefault("env", {})["layout"] = args.layout
    try:
        train_run(cfg, args.seed, run_id=args.run_id)
    except ValueError as e:
        sys.exit(f"invalid config: {e}")


if __name__ == "__main__":
    main()
