"""Export a backend-free demo of the gallery to dist/ (e.g. for GitHub Pages).

    python scripts/export_static.py && python -m http.server -d dist 8080

The UI runs from snapshotted API responses. Training, custom rewards and live
evaluation need the FastAPI backend and are disabled in this mode.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.api import main as api  # noqa: E402
from backend.env.rewards import validate  # noqa: E402

DIST = ROOT / "dist"


def dump(rel: str, obj) -> None:
    p = DIST / "api" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, separators=(",", ":")))


def main():
    shutil.rmtree(DIST, ignore_errors=True)
    shutil.copytree(ROOT / "frontend", DIST)
    idx = (DIST / "index.html").read_text(encoding="utf-8")
    idx = idx.replace('<script type="module" src="main.js">',
                      '<script>window.REWARD_GOBLIN_STATIC = true;</script>\n<script type="module" src="main.js">')
    (DIST / "index.html").write_text(idx, encoding="utf-8")

    dump("meta.json", api.meta())
    dump("experiments.json", api.experiments())
    lint = {}
    for e in api.store.gallery():
        exp = api.experiment(e["id"])
        dump(f"experiments/{e['id']}.json", exp)
        for h in exp["history"]:
            _, warnings = validate(h["reward"], h["env"]["layout"])
            lint[h["reward_fingerprint"]] = {"ok": True, "errors": [], "warnings": warnings}
            for r in h["runs"]:
                rid = r["run_id"]
                dump(f"runs/{rid}.json", api.run(rid))
                dump(f"runs/{rid}/metrics.json", api.run_metrics(rid))
                for kind in api.store.recording_kinds(rid):
                    dump(f"replays/{rid}/{kind}.json", api.store.recording(rid, kind))
    dump("lint.json", lint)
    size = sum(p.stat().st_size for p in DIST.rglob("*") if p.is_file())
    print(f"exported to {DIST} ({size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
