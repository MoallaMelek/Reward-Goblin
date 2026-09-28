"""Episode recordings: compact, column-oriented JSON that the frontend replays frame by frame."""
from __future__ import annotations

import json
from pathlib import Path

from ..analysis.true_objective import checklist

RECORDING_VERSION = 1


def _r(v) -> float:
    return round(float(v), 3)


def build_recording(trace: dict, layout, reward_cfg, *, kind: str, layout_seed: int | None,
                    reference_trace: dict | None = None, extra: dict | None = None) -> dict:
    keys = sorted(reward_cfg.components)
    rec = {
        "format": RECORDING_VERSION,
        "kind": kind,
        "layout_seed": layout_seed,
        "layout": layout.to_dict(),
        "reward_name": reward_cfg.name,
        "component_keys": keys,
        "stable_goal_frames": reward_cfg.stable_frames,
        **_frames(trace, keys),
        "exploits": trace.get("exploits", []),
    }
    if reference_trace is not None:
        rec["reference"] = _frames(reference_trace, keys)
    if extra:
        rec.update(extra)
    return rec


def _frames(trace: dict, keys: list[str]) -> dict:
    return {
        "frames": {
            "agent": [[_r(x), _r(y)] for x, y in trace["agent"]],
            "box": [[_r(x), _r(y)] for x, y in trace["box"]],
            "action": list(trace["action"]),
            "reward": [_r(v) for v in trace["reward"]],
            "cum": [_r(v) for v in trace["cum"]],
            "overlap": [int(v) for v in trace["overlap"]],
            "inside": [int(v) for v in trace["inside"]],
            "stable": list(trace["stable"]),
            "components": {k: [_r(c.get(k, 0.0)) for c in trace["components"]] for k in keys},
        },
        "settle": {"agent": trace.get("settle_agent", []), "box": trace.get("settle_box", [])},
        "events": [{"step": s, "type": t, "text": txt} for s, t, txt in trace["events"]],
        "true_success": trace.get("true_success"),
        "return": _r(trace.get("return", 0.0)),
        "length": trace.get("length", len(trace["reward"]) - 1),
        "termination_reason": trace.get("termination_reason"),
        "checklist": checklist(trace),
    }


def save_recording(rec: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, separators=(",", ":")))


def load_recording(path: Path) -> dict:
    return json.loads(path.read_text())
