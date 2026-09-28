"""Read-side of the run database (plain JSON files under runs/, recordings/, experiments/)."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ..analysis.exploit_detection import LABELS

SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]{1,120}$")


class Store:
    def __init__(self, root: Path):
        self.root = root
        self.runs_dir = root / "runs"
        self.rec_dir = root / "recordings"
        self.exp_dir = root / "experiments"

    # ------------------------------------------------------------------ runs
    def _read(self, path: Path):
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def run_ids(self) -> list[str]:
        if not self.runs_dir.exists():
            return []
        return sorted(p.name for p in self.runs_dir.iterdir() if p.is_dir() and not p.name.startswith("_"))

    def meta(self, run_id: str) -> dict | None:
        if not SAFE_ID.match(run_id):
            return None
        return self._read(self.runs_dir / run_id / "meta.json")

    def status(self, run_id: str) -> dict | None:
        if not SAFE_ID.match(run_id):
            return None
        return self._read(self.runs_dir / run_id / "status.json")

    def metrics(self, run_id: str) -> dict | None:
        if not SAFE_ID.match(run_id):
            return None
        return self._read(self.runs_dir / run_id / "metrics.json")

    def eval(self, run_id: str) -> dict | None:
        if not SAFE_ID.match(run_id):
            return None
        return self._read(self.runs_dir / run_id / "eval.json")

    def recording(self, run_id: str, kind: str) -> dict | None:
        if not (SAFE_ID.match(run_id) and SAFE_ID.match(kind)):
            return None
        return self._read(self.rec_dir / run_id / f"{kind}.json")

    def recording_kinds(self, run_id: str) -> list[str]:
        d = self.rec_dir / run_id
        return sorted(p.stem for p in d.glob("*.json")) if SAFE_ID.match(run_id) and d.exists() else []

    def list_runs(self, experiment: str | None = None) -> list[dict]:
        out = []
        for rid in self.run_ids():
            m = self.meta(rid)
            if not m or (experiment and m.get("experiment") != experiment):
                continue
            out.append(_run_brief(m))
        return out

    # ------------------------------------------------------------------ experiments
    def gallery(self) -> list[dict]:
        if not self.exp_dir.exists():
            return []
        items = []
        for p in sorted(self.exp_dir.glob("*/experiment.json")):
            e = self._read(p)
            if e:
                e["id"] = p.parent.name
                items.append(e)
        items.sort(key=lambda e: e.get("order", 99))
        return items

    def experiment(self, exp_id: str) -> dict | None:
        if not SAFE_ID.match(exp_id):
            return None
        e = self._read(self.exp_dir / exp_id / "experiment.json") or {"id": exp_id, "title": exp_id, "user": True}
        e["id"] = exp_id
        e["history"] = self.version_history(exp_id)
        if not e["history"] and "versions" not in e:
            return None
        return e

    def experiment_ids(self) -> list[str]:
        ids = {e["id"] for e in self.gallery()}
        for rid in self.run_ids():
            m = self.meta(rid)
            if m:
                ids.add(m["experiment"])
        return sorted(ids)

    def next_version(self, exp_id: str) -> int:
        versions = [m["version"] for m in (self.meta(r) for r in self.run_ids()) if m and m["experiment"] == exp_id]
        g = self._read(self.exp_dir / exp_id / "experiment.json") if SAFE_ID.match(exp_id) else None
        if g:
            versions += [v["version"] for v in g.get("versions", [])]
        return max(versions, default=0) + 1

    def version_history(self, exp_id: str) -> list[dict]:
        """Group runs by reward version and pool their held-out evaluations across seeds."""
        groups: dict[int, list[dict]] = defaultdict(list)
        for rid in self.run_ids():
            m = self.meta(rid)
            if m and m.get("experiment") == exp_id:
                groups[m["version"]].append(m)
        out = []
        for version in sorted(groups):
            metas = sorted(groups[version], key=lambda m: m["seed"])
            first = metas[0]
            done = [m for m in metas if m.get("state") == "done"]
            suites: dict[str, dict] = {}
            for m in done:
                ev = self.eval(m["run_id"]) or {}
                for name, s in ev.get("suites", {}).items():
                    agg = suites.setdefault(name, {"layout": s["layout"], "episodes": [], "per_seed": [],
                                                   "reference": ev.get("reference", {}).get(name)})
                    agg["episodes"].extend(s["episodes"])
                    agg["per_seed"].append({"seed": m["seed"], **_pick(s["summary"])})
            pooled = {name: {"layout": a["layout"], "reference": a["reference"],
                             "seeds_tested": len(a["per_seed"]),
                             "eval_envs": len({e["layout_seed"] for e in a["episodes"]}),
                             **_pool(a["episodes"]), "per_seed": a["per_seed"]}
                      for name, a in suites.items()}
            out.append({
                "experiment": exp_id, "version": version, "title": first.get("title"),
                "description": first.get("description", ""), "reward": first["reward"],
                "reward_fingerprint": first.get("reward_fingerprint"), "env": first["env"],
                "algo": first["algo"], "steps": first["steps"],
                "runs": [_run_brief(m) for m in metas], "suites": pooled,
                "complete": len(done) == len(metas),
            })
        return out


def _pick(s: dict) -> dict:
    return {k: s.get(k) for k in ("mean_return", "true_success_rate", "exploit_rate", "dominant_exploit",
                                  "mean_reference_return", "outscores_reference_rate")}


def _pool(eps: list[dict]) -> dict:
    if not eps:
        return {"episodes": 0}
    heads = Counter(e["exploits"][0] for e in eps if e["exploits"])
    return {
        "episodes": len(eps),
        "true_success_rate": float(np.mean([bool(e["true_success"]) for e in eps])),
        "exploit_rate": float(np.mean([bool(e["exploits"]) for e in eps])),
        "mean_return": float(np.mean([e["return"] for e in eps])),
        "mean_reference_return": float(np.mean([e["reference_return"] for e in eps
                                                if e.get("reference_return") is not None] or [0.0])),
        "reference_success_rate": float(np.mean([bool(e.get("reference_success")) for e in eps])),
        "exploit_breakdown": {LABELS.get(k, k): v for k, v in heads.most_common()},
        "dominant_exploit": heads.most_common(1)[0][0] if heads else None,
    }


def _run_brief(m: dict) -> dict:
    return {k: m.get(k) for k in ("run_id", "experiment", "version", "title", "seed", "steps", "state",
                                  "verdict", "summary", "reference", "created", "train_seconds",
                                  "reward_fingerprint", "algo")}
