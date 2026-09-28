"""SB3 callbacks: genuine training metrics (no smoothing, no invention) streamed to disk."""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from ..analysis.metrics import summarize


def write_json_atomic(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, separators=(",", ":")))
    os.replace(tmp, path)


class MetricsCallback(BaseCallback):
    """After every rollout, aggregates the episodes that finished during it.

    Per-episode data comes from ``info["episode_diag"]``, computed inside the env at episode
    end: training return, true-objective success (settle test) and exploit-detector codes.
    Entropy/KL/loss values are read from SB3's logger (they describe the previous update).
    Optionally runs a small deterministic evaluation at fixed fractions of training.
    """

    def __init__(self, run_dir: Path, total_steps: int, eval_fn=None, eval_points: int = 5,
                 status_extra: dict | None = None):
        super().__init__()
        self.run_dir = run_dir
        self.total = total_steps
        self.eval_fn = eval_fn
        self.eval_at = [int(total_steps * (k + 1) / eval_points) for k in range(eval_points)] if eval_fn else []
        self.buffer: list[dict] = []
        self.series: list[dict] = []
        self.evals: list[dict] = []
        self.started = time.time()
        self.status_extra = status_extra or {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            d = info.get("episode_diag")
            if d is not None:
                self.buffer.append(d)
        return True

    def _on_rollout_end(self) -> None:
        t = int(self.num_timesteps)
        row = {"t": t, "wall_s": round(time.time() - self.started, 2)}
        if self.buffer:
            s = summarize(self.buffer)
            row.update({k: s[k] for k in ("episodes", "mean_return", "true_success_rate", "exploit_rate",
                                          "mean_length")})
            row["exploit_counts"] = dict(Counter(c for e in self.buffer for c in e["exploits"]))
            row["termination"] = dict(Counter(e["termination_reason"] for e in self.buffer))
        vals = self.model.logger.name_to_value
        if "train/entropy_loss" in vals:
            row["entropy"] = -float(vals["train/entropy_loss"])
        for src, dst in (("train/approx_kl", "approx_kl"), ("train/value_loss", "value_loss"),
                         ("train/explained_variance", "explained_variance"), ("train/clip_fraction", "clip_fraction")):
            if src in vals:
                row[dst] = float(vals[src])
        self.series.append(row)
        self.buffer = []
        if self.eval_at and t >= self.eval_at[0]:
            while self.eval_at and t >= self.eval_at[0]:
                self.eval_at.pop(0)
            ev = self.eval_fn(self.model)
            ev["t"] = t
            self.evals.append(ev)
        self.flush()

    def flush(self, state: str = "training") -> None:
        write_json_atomic(self.run_dir / "metrics.json", {"train": self.series, "eval": self.evals})
        write_json_atomic(self.run_dir / "status.json", {
            "state": state, "timesteps": int(self.num_timesteps), "total": self.total,
            "elapsed_s": round(time.time() - self.started, 1), **self.status_extra})
