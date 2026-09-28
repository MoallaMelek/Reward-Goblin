"""Aggregate per-episode diagnostics into the numbers shown in tables and charts."""
from __future__ import annotations

from collections import Counter

import numpy as np


def summarize(episodes: list[dict]) -> dict:
    """``episodes``: dicts with return, length, true_success, exploits (list of codes)."""
    if not episodes:
        return {"episodes": 0}
    rets = np.array([e["return"] for e in episodes], dtype=float)
    succ = np.array([bool(e["true_success"]) for e in episodes], dtype=float)
    expl = np.array([bool(e["exploits"]) for e in episodes], dtype=float)
    labels = Counter(code for e in episodes for code in e["exploits"])
    headline = Counter(e["exploits"][0] for e in episodes if e["exploits"])
    out = {
        "episodes": len(episodes),
        "mean_return": float(rets.mean()),
        "std_return": float(rets.std()),
        "true_success_rate": float(succ.mean()),
        "exploit_rate": float(expl.mean()),
        "mean_length": float(np.mean([e["length"] for e in episodes])),
        "exploit_counts": dict(labels),
        "headline_exploits": dict(headline),
        "dominant_exploit": headline.most_common(1)[0][0] if headline else None,
    }
    refs = [e["reference_return"] for e in episodes if e.get("reference_return") is not None]
    if refs:
        out["mean_reference_return"] = float(np.mean(refs))
        out["outscores_reference_rate"] = float(np.mean(
            [e["return"] > e["reference_return"] for e in episodes if e.get("reference_return") is not None]))
    return out


def classify_run(summary: dict) -> str:
    """One-word verdict for the seed table."""
    if summary.get("episodes", 0) == 0:
        return "unknown"
    s, x = summary["true_success_rate"], summary["exploit_rate"]
    if s >= 0.8 and x <= 0.2:
        return "intended"
    if x >= 0.5:
        return "exploit"
    if s >= 0.4:
        return "unstable"
    return "failure"
