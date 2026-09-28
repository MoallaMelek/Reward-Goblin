"""Print the README results tables (markdown) from the stored evaluations.

    python scripts/results_table.py > docs/RESULTS.md
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.analysis.exploit_detection import LABELS  # noqa: E402
from backend.api.store import Store  # noqa: E402


def pct(v):
    return "–" if v is None else f"{round(v * 100)}%"


def main():
    st = Store(ROOT)
    print("| Experiment | Reward version | Mean reward (goblin / intended) | True success | Exploit rate | Most common exploit | Unseen layouts: success |")
    print("|---|---|---:|---:|---:|---|---:|")
    for e in st.gallery():
        for h in st.version_history(e["id"]):
            s = h["suites"].get("in_distribution", {})
            sh = h["suites"].get("shifted", {})
            if not s.get("episodes"):
                continue
            dom = LABELS.get(s.get("dominant_exploit"), "none") if s.get("dominant_exploit") else "none"
            print(f"| {e['title']} | v{h['version']} {h['title']} | {s['mean_return']:.1f} / {s['mean_reference_return']:.1f} | "
                  f"{pct(s['true_success_rate'])} | {pct(s['exploit_rate'])} | {dom} | {pct(sh.get('true_success_rate'))} |")
    print()
    print("Per-seed breakdown (held-out start states, 20 episodes each):")
    print()
    print("| Experiment | Version | Seed | Return | True success | Exploit rate | Dominant exploit |")
    print("|---|---|---:|---:|---:|---:|---|")
    for e in st.gallery():
        for h in st.version_history(e["id"]):
            for p in h["suites"].get("in_distribution", {}).get("per_seed", []):
                dom = LABELS.get(p["dominant_exploit"], "–") if p["dominant_exploit"] else "–"
                print(f"| {e['title']} | v{h['version']} | {p['seed']} | {p['mean_return']:.1f} | "
                      f"{pct(p['true_success_rate'])} | {pct(p['exploit_rate'])} | {dom} |")


if __name__ == "__main__":
    main()
