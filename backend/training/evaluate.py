"""Deterministic evaluation of trained policies on held-out layouts.

Every evaluation episode is paired with the scripted reference controller on the
*identical* start state, which gives two things: the left-panel animation, and the
return the intended behaviour would have earned under the same reward.

CLI:
    python evaluate.py --model models/<run_id>.zip --episodes 100 [--layout medium]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..analysis.exploit_detection import detect_exploits
from ..analysis.metrics import summarize
from ..env.reference_policy import ReferencePolicy
from ..env.reward_goblin_env import EnvConfig, RewardGoblinEnv
from ..env.rewards import RewardConfig

EVAL_SEED_BASE = 900_000  # far away from training seeds
_REF_CACHE: dict = {}


def layout_seeds(n: int, offset: int = 0) -> list[int]:
    return [EVAL_SEED_BASE + offset + i for i in range(n)]


def rollout(env: RewardGoblinEnv, act, layout_seed: int) -> dict:
    obs, _ = env.reset(seed=layout_seed)
    done = False
    while not done:
        obs, _, term, trunc, _ = env.step(act(obs))
        done = term or trunc
    return env.last_trace


def reference_trace(reward_cfg: RewardConfig, env_cfg: EnvConfig, layout_seed: int) -> tuple[dict, object]:
    key = (reward_cfg.fingerprint(), json.dumps(env_cfg.to_dict(), sort_keys=True), layout_seed)
    if key not in _REF_CACHE:
        env = RewardGoblinEnv(reward_cfg, env_cfg)
        pol = ReferencePolicy(env)
        tr = rollout(env, pol.act, layout_seed)
        _REF_CACHE[key] = (tr, env.layout)
    return _REF_CACHE[key]


def evaluate_policy(model, reward_cfg: RewardConfig, env_cfg: EnvConfig, seeds: list[int],
                    deterministic: bool = True, with_reference: bool = True) -> list[dict]:
    env = RewardGoblinEnv(reward_cfg, env_cfg)
    episodes = []
    for s in seeds:
        tr = rollout(env, lambda o: int(model.predict(o, deterministic=deterministic)[0]), s)
        layout = env.layout
        ref_tr = None
        ref_ret = None
        if with_reference:
            ref_tr, _ = reference_trace(reward_cfg, env_cfg, s)
            ref_ret = ref_tr["return"]
            tr["exploits"] = detect_exploits(tr, reward_cfg, reference_return=ref_ret)
        episodes.append({
            "layout_seed": s,
            "return": tr["return"],
            "length": tr["length"],
            "true_success": tr["true_success"],
            "termination_reason": tr["termination_reason"],
            "exploits": [f["code"] for f in tr["exploits"]],
            "reference_return": ref_ret,
            "reference_success": ref_tr["true_success"] if ref_tr else None,
            "_trace": tr,
            "_ref": ref_tr,
            "_layout": layout,
        })
    return episodes


def public(ep: dict) -> dict:
    return {k: v for k, v in ep.items() if not k.startswith("_")}


def highlights(episodes: list[dict]) -> dict[str, dict]:
    """Pick the replays offered in the UI."""
    if not episodes:
        return {}
    by_ret = sorted(episodes, key=lambda e: e["return"])
    out = {
        "best_reward": by_ret[-1],
        "median": by_ret[len(by_ret) // 2],
        "seed_run": episodes[0],  # canonical first evaluation layout
    }
    succ = [e for e in episodes if e["true_success"]]
    if succ:
        out["best_true_success"] = max(succ, key=lambda e: e["return"])
    expl = [e for e in episodes if e["exploits"] and not e["true_success"]]
    if expl:
        out["worst_exploit"] = max(expl, key=lambda e: e["return"])
    return out


def reference_summary(reward_cfg: RewardConfig, env_cfg: EnvConfig, seeds: list[int]) -> dict:
    eps = []
    for s in seeds:
        tr, _ = reference_trace(reward_cfg, env_cfg, s)
        eps.append({"return": tr["return"], "length": tr["length"], "true_success": tr["true_success"],
                    "exploits": []})
    return summarize(eps)


def main(argv=None):
    from stable_baselines3 import A2C, PPO

    ap = argparse.ArgumentParser(description="Evaluate a trained Reward Goblin policy.")
    ap.add_argument("--model", required=True, help="path to models/<run_id>.zip")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--layout", default=None, help="override layout / difficulty (default: training layout)")
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--out", default=None, help="write per-episode JSON here")
    args = ap.parse_args(argv)

    model_path = Path(args.model)
    meta_path = model_path.parent.parent / "runs" / model_path.stem / "meta.json"
    if not meta_path.exists():
        sys.exit(f"cannot find run metadata at {meta_path}")
    meta = json.loads(meta_path.read_text())
    reward_cfg = RewardConfig.from_dict(meta["reward"])
    env_d = dict(meta["env"])
    if args.layout:
        env_d["layout"] = args.layout
    env_cfg = EnvConfig.from_dict(env_d)
    algo = {"PPO": PPO, "A2C": A2C}[meta["algo"]["name"]]
    model = algo.load(str(model_path), device="cpu")
    eps = evaluate_policy(model, reward_cfg, env_cfg, layout_seeds(args.episodes),
                          deterministic=not args.stochastic)
    s = summarize([public(e) for e in eps])
    print(json.dumps({k: v for k, v in s.items()}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps([public(e) for e in eps], indent=1))


if __name__ == "__main__":
    main()
