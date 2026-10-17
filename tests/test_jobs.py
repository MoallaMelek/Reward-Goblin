import json
from concurrent.futures import Future

from backend.api.jobs import JobManager


def test_worker_future_failure_is_visible_as_job_error(tmp_path):
    jobs = JobManager(tmp_path)
    job = {"job_id": "failed", "run_ids": ["run"], "seeds": [1], "steps": 1000}
    (jobs.jobs_dir / "failed.json").write_text(json.dumps(job))
    future = Future()
    future.set_exception(RuntimeError("worker process stopped"))
    jobs._futures["failed"] = [future]
    result = jobs.get("failed")
    assert result["state"] == "error"
    assert result["runs"][0]["error"] == "worker process stopped"
