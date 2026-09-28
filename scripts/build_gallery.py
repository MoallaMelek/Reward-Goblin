"""Train every pre-built gallery experiment (all versions x all seeds).

    python scripts/build_gallery.py                 # train whatever is missing
    python scripts/build_gallery.py --only touch_goblin --force
    python scripts/build_gallery.py --workers 6

Each experiments/<id>/experiment.json lists reward versions; runs are stored under the
deterministic id <id>_v<version>_seed<seed>, so re-running is idempotent and results are
reproducible from the committed configs.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.training.batch import _worker, default_workers  # noqa: E402


def jobs_for(exp_dir: Path, force: bool) -> list[tuple[dict, int, str]]:
    spec = json.loads((exp_dir / "experiment.json").read_text())
    out = []
    for v in spec["versions"]:
        cfg = {"experiment": exp_dir.name, "version": v["version"], "title": v["title"],
               "description": v.get("description", ""), "reward": v["reward"], "env": v.get("env", spec.get("env", {})),
               "algo": v.get("algo", spec.get("algo", {})), "steps": v.get("steps", spec.get("steps", 300_000))}
        for seed in v.get("seeds", spec.get("seeds", [1, 2, 3])):
            rid = f"{exp_dir.name}_v{v['version']}_seed{seed}"
            meta = ROOT / "runs" / rid / "meta.json"
            done = meta.exists() and json.loads(meta.read_text()).get("state") == "done"
            if done and not force:
                continue
            for p in (ROOT / "runs" / rid, ROOT / "recordings" / rid, ROOT / "models" / f"{rid}_ckpt"):
                shutil.rmtree(p, ignore_errors=True)
            (ROOT / "models" / f"{rid}.zip").unlink(missing_ok=True)
            out.append((cfg, seed, rid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="experiment ids")
    ap.add_argument("--force", action="store_true", help="retrain even if a finished run exists")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    dirs = sorted(p.parent for p in (ROOT / "experiments").glob("*/experiment.json"))
    if args.only:
        dirs = [d for d in dirs if d.name in args.only]
    jobs = [j for d in dirs for j in jobs_for(d, args.force)]
    # longest first so the pool drains evenly
    jobs.sort(key=lambda j: -j[0]["steps"])
    print(f"{len(jobs)} runs to train")
    if not jobs:
        return
    t0 = time.time()
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers or default_workers(len(jobs))) as pool:
        futs = {pool.submit(_worker, cfg, seed, str(ROOT), rid, None): rid for cfg, seed, rid in jobs}
        for f in as_completed(futs):
            m = f.result()
            if "error" in m:
                failed += 1
                print(f"FAILED {futs[f]}: {m['error']}\n{m['traceback']}", flush=True)
    print(f"finished in {(time.time() - t0) / 60:.1f} min, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
