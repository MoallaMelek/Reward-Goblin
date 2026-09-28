"""Training jobs: one job = one reward version trained on N seeds in worker processes.

Progress is read from each run's status.json / metrics.json, which the training
callback writes after every PPO rollout - so the UI shows genuine, current numbers.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..training.batch import _worker, default_workers
from ..training.callbacks import write_json_atomic
from ..training.train import normalize_config, run_id_for


class JobManager:
    def __init__(self, root: Path, max_workers: int | None = None):
        self.root = root
        self.jobs_dir = root / "runs" / "_jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.max_workers = max_workers or default_workers(64)
        self._pool: ProcessPoolExecutor | None = None
        self._lock = threading.Lock()
        self._futures: dict[str, list] = {}

    def _executor(self) -> ProcessPoolExecutor:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(max_workers=self.max_workers)
        return self._pool

    def submit(self, cfg: dict, seeds: list[int]) -> dict:
        cfg = normalize_config(cfg)  # raises ValueError on bad input
        job_id = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
        run_ids = []
        with self._lock:
            for s in seeds:
                rid = run_id_for(cfg, s, self.root)
                (self.root / "runs" / rid).mkdir(parents=True)
                write_json_atomic(self.root / "runs" / rid / "status.json",
                                  {"state": "queued", "timesteps": 0, "total": cfg["steps"], "run_id": rid})
                run_ids.append(rid)
            job = {"job_id": job_id, "experiment": cfg["experiment"], "version": cfg["version"],
                   "title": cfg["title"], "seeds": seeds, "run_ids": run_ids, "steps": cfg["steps"],
                   "algo": cfg["algo"]["name"], "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
            write_json_atomic(self.jobs_dir / f"{job_id}.json", job)
            ex = self._executor()
            self._futures[job_id] = [
                ex.submit(_worker, cfg, s, str(self.root), rid, {"job_id": job_id})
                for s, rid in zip(seeds, run_ids)]
        return job

    def get(self, job_id: str) -> dict | None:
        path = self.jobs_dir / f"{job_id}.json"
        if not path.exists() or "/" in job_id or "\\" in job_id:
            return None
        job = json.loads(path.read_text())
        runs = []
        for i, rid in enumerate(job["run_ids"]):
            st = _read(self.root / "runs" / rid / "status.json") or {"state": "unknown"}
            fut = self._futures.get(job_id, [None] * len(job["run_ids"]))[i]
            if fut is not None and fut.done():
                res = fut.result()
                if isinstance(res, dict) and "error" in res:
                    st = {**st, "state": "error", "error": res["error"]}
            elif fut is None and st.get("state") in ("queued", "training", "evaluating"):
                st = {**st, "state": "interrupted"}  # server restarted mid-run
            runs.append({"run_id": rid, "seed": job["seeds"][i], **st})
        states = {r["state"] for r in runs}
        if states <= {"done"}:
            job["state"] = "done"
        elif "error" in states and not states & {"queued", "training", "evaluating"}:
            job["state"] = "error"
        elif states & {"training", "evaluating"}:
            job["state"] = "training"
        elif states <= {"interrupted", "done", "error"}:
            job["state"] = "interrupted"
        else:
            job["state"] = "queued"
        job["runs"] = runs
        total = sum(r.get("total", job["steps"]) for r in runs)
        job["progress"] = sum(min(r.get("timesteps", 0), r.get("total", job["steps"])) for r in runs) / max(total, 1)
        return job

    def metrics(self, job_id: str) -> dict | None:
        job = self.get(job_id)
        if job is None:
            return None
        return {"job_id": job_id, "runs": [
            {"run_id": rid, "seed": s, "metrics": _read(self.root / "runs" / rid / "metrics.json") or
             {"train": [], "eval": []}} for rid, s in zip(job["run_ids"], job["seeds"])]}

    def list(self) -> list[dict]:
        return [j for j in (self.get(p.stem) for p in sorted(self.jobs_dir.glob("*.json"))) if j]

    def shutdown(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)


def _read(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
