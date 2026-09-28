"""Run many (config, seed) trainings in parallel worker processes.

    python -m backend.training.batch configs/touch_goblin_v1.json configs/touch_goblin_v2.json --seeds 1 2 3
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .train import ROOT, train_run


def _worker(cfg: dict, seed: int, root: str, run_id: str | None, status_extra: dict | None):
    try:
        return train_run(cfg, seed, root=Path(root), run_id=run_id, status_extra=status_extra,
                         log=lambda m: print(m, flush=True))
    except Exception as e:  # surfaced to the caller, never swallowed
        import traceback
        return {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(), "seed": seed,
                "run_id": run_id}


def default_workers(n_jobs: int) -> int:
    return max(1, min(n_jobs, (os.cpu_count() or 2) - 2))


def run_batch(jobs: list[tuple[dict, int]], workers: int | None = None, root: Path = ROOT) -> list[dict]:
    workers = workers or default_workers(len(jobs))
    out = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_worker, cfg, seed, str(root), None, None) for cfg, seed in jobs]
        for f in as_completed(futs):
            out.append(f.result())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)
    jobs = []
    for path in args.configs:
        cfg = json.loads(Path(path).read_text())
        if args.steps:
            cfg["steps"] = args.steps
        jobs += [(cfg, s) for s in args.seeds]
    results = run_batch(jobs, args.workers)
    print("\n{:<34} {:>8} {:>8} {:>8}  {}".format("run", "return", "success", "exploit", "dominant"))
    for m in sorted(results, key=lambda m: m.get("run_id") or ""):
        if "error" in m:
            print("ERROR", m["error"])
            print(m["traceback"])
            continue
        s = m["summary"]["in_distribution"]
        sh = m["summary"].get("shifted", {})
        print("{:<34} {:>8.2f} {:>8.0%} {:>8.0%}  {}  | shifted succ {:.0%} expl {:.0%} | ref {:.2f}".format(
            m["run_id"], s["mean_return"], s["true_success_rate"], s["exploit_rate"], s["dominant_exploit"],
            sh.get("true_success_rate", 0), sh.get("exploit_rate", 0),
            m["reference"]["in_distribution"]["mean_return"]))


if __name__ == "__main__":
    main()
