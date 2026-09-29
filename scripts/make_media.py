"""Render README media from real recordings and evaluation results.

    python scripts/make_media.py            # docs/demo_*.gif, docs/results.png, docs/touch_training.png

Everything drawn here is read from runs/ and recordings/ - no hand-made numbers.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DOCS = ROOT / "docs"

from backend.analysis.exploit_detection import LABELS  # noqa: E402
from backend.api.store import Store  # noqa: E402

BG, FLOOR, WALL, EXIT, CRATE = (12, 15, 21), (27, 33, 48), (115, 131, 166), (58, 214, 120), (217, 165, 91)
GOBLIN, SCHEME, IDEAL, TEXT, MUTED, DANGER = (111, 211, 90), (255, 107, 107), (106, 168, 255), (232, 236, 244), (132, 144, 166), (255, 107, 107)
LAVA = (255, 110, 50)


def font(size, bold=False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_arena(d: ImageDraw.ImageDraw, ox, oy, s, layout, agent, box, actor, flagged, trail):
    W, H = layout["width"], layout["height"]
    X = lambda x: ox + x * s  # noqa: E731
    Y = lambda y: oy + (H - y) * s  # noqa: E731
    d.rectangle([X(0), Y(H), X(W), Y(0)], fill=FLOOR, outline=WALL, width=3)
    for h in layout["hazards"]:
        d.rectangle([X(h[0]), Y(h[3]), X(h[2]), Y(h[1])], fill=LAVA)
    g = layout["goal"]
    d.rectangle([X(g[0]), Y(g[3]), X(g[2]), Y(g[1])], fill=(30, 70, 50), outline=EXIT, width=2)
    for r in layout["walls"] + layout["obstacles"]:
        d.rectangle([X(r[0]), Y(r[3]), X(r[2]), Y(r[1])], fill=WALL)
    for i in range(1, len(trail)):
        c = SCHEME if flagged else (IDEAL if actor == "ideal" else GOBLIN)
        a = i / len(trail)
        col = tuple(int(FLOOR[k] * (1 - a * 0.6) + c[k] * a * 0.6) for k in range(3))
        d.line([X(trail[i - 1][0]), Y(trail[i - 1][1]), X(trail[i][0]), Y(trail[i][1])], fill=col, width=2)
    bx, by = box
    d.rectangle([X(bx - 0.4), Y(by + 0.4), X(bx + 0.4), Y(by - 0.4)], fill=CRATE, outline=(168, 118, 58), width=2)
    ax, ay = agent
    r = 0.3 * s
    col = IDEAL if actor == "ideal" else (SCHEME if flagged else GOBLIN)
    if actor != "ideal":
        for side in (-1, 1):
            d.polygon([(X(ax) + side * r * 0.55, Y(ay) - r * 0.55), (X(ax) + side * r * 1.45, Y(ay) - r * 1.05),
                       (X(ax) + side * r * 0.95, Y(ay) - r * 0.1)], fill=tuple(int(v * 0.6) for v in col))
    d.ellipse([X(ax) - r, Y(ay) - r, X(ax) + r, Y(ay) + r], fill=col)
    for side in (-1, 1):
        d.ellipse([X(ax) + side * r * 0.36 - r * 0.2, Y(ay) - r * 0.35, X(ax) + side * r * 0.36 + r * 0.2, Y(ay) + r * 0.05], fill="white")


def state(rec, t):
    f = rec["frames"]
    n = len(f["agent"])
    if t < n:
        return f["agent"][t], f["box"][t], t
    k = min(t - n, len(rec["settle"]["agent"]) - 1)
    return rec["settle"]["agent"][k], rec["settle"]["box"][k], n - 1


def make_gif(store: Store, run_id: str, kind: str, out: Path, title: str, stride: int = 2):
    rec = store.recording(run_id, kind)
    ref = rec["reference"]
    lay = rec["layout"]
    pw, ph = 420, 294
    s = (pw - 20) / lay["width"]
    Wimg, Himg = 2 * pw + 30, ph + 150
    total = max(len(rec["frames"]["agent"]) + len(rec["settle"]["agent"]),
                len(ref["frames"]["agent"]) + len(ref["settle"]["agent"]))
    f_big, f_mid, f_small = font(26, True), font(15, True), font(13)
    frames = []
    for t in list(range(0, total, stride)) + [total - 1] * 12:
        im = Image.new("RGB", (Wimg, Himg), BG)
        d = ImageDraw.Draw(im)
        d.text((10, 8), title, fill=TEXT, font=f_mid)
        d.text((10, 34), "WHAT YOU INTENDED", fill=IDEAL, font=f_mid)
        d.text((pw + 20, 34), "WHAT YOU REWARDED", fill=GOBLIN, font=f_mid)
        a, b, ri = state(ref, t)
        draw_arena(d, 10, 60, s, lay, a, b, "ideal", False, ref["frames"]["agent"][max(0, ri - 40): ri + 1])
        a, b, gi = state(rec, t)
        flagged = any(fd["step"] <= gi for fd in rec["exploits"])
        draw_arena(d, pw + 20, 60, s, lay, a, b, "goblin", flagged, rec["frames"]["agent"][max(0, gi - 40): gi + 1])
        if flagged:
            d.rectangle([pw + 20 + pw / 2 - 120, 66, pw + 20 + pw / 2 + 120, 88], fill=DANGER)
            d.text((pw + 20 + pw / 2 - 110, 69), "POSSIBLE REWARD EXPLOIT", fill=(26, 5, 5), font=f_mid)
        y0 = ph + 70
        cum_g = rec["frames"]["cum"][gi]
        cum_r = ref["frames"]["cum"][ri]
        d.text((10, y0), "reward earned", fill=MUTED, font=f_small)
        d.text((10, y0 + 16), f"{cum_r:6.1f}", fill=TEXT, font=f_big)
        d.text((pw + 20, y0), "reward earned", fill=MUTED, font=f_small)
        d.text((pw + 20, y0 + 16), f"{cum_g:6.1f}", fill=TEXT, font=f_big)
        end = t >= total - 1
        ok_r = "task: PASS" if (end and ref["true_success"]) else ("task: FAILED" if end else "")
        ok_g = "task: PASS" if (end and rec["true_success"]) else ("task: FAILED" if end else "")
        d.text((150, y0 + 22), ok_r, fill=EXIT if ref["true_success"] else DANGER, font=f_mid)
        d.text((pw + 160, y0 + 22), ok_g, fill=EXIT if rec["true_success"] else DANGER, font=f_mid)
        if flagged:
            d.text((pw + 20, y0 + 54), rec["exploits"][0]["label"], fill=DANGER, font=f_small)
        d.text((10, y0 + 54), "scripted reference (not learned)", fill=MUTED, font=f_small)
        frames.append(im.quantize(colors=64))
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / 15 * stride), loop=0, optimize=True)
    print("wrote", out, len(frames), "frames")


def results_figure(store: Store, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for e in store.gallery():
        for h in store.version_history(e["id"]):
            s = h["suites"].get("in_distribution")
            if s and s.get("episodes"):
                rows.append((f"{e['title']} v{h['version']}", s["true_success_rate"], s["exploit_rate"]))
    if not rows:
        return
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(rows) + 1.2), facecolor="#141923")
    ax.set_facecolor("#141923")
    ys = range(len(rows))[::-1]
    h = 0.36
    ax.barh([y + h / 2 for y in ys], [r[1] * 100 for r in rows], height=h, color="#3987e5", label="True task success")
    ax.barh([y - h / 2 for y in ys], [r[2] * 100 for r in rows], height=h, color="#d95926", label="Exploit detected")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows], color="#e8ecf4")
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of evaluation episodes (3 seeds x 20 held-out start states)", color="#8490a6")
    ax.tick_params(colors="#8490a6")
    ax.grid(axis="x", color="#2a3346", linewidth=1)
    ax.set_axisbelow(True)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(loc="lower right", frameon=False, labelcolor="#e8ecf4")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print("wrote", out)


def training_figure(store: Store, exp_id: str, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = ["#3987e5", "#d95926", "#199e70", "#c98500"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor="#141923")
    for ax, key, title in zip(axes, ("true_success_rate", "exploit_rate"), ("True success (training episodes)", "Exploit frequency")):
        ax.set_facecolor("#141923")
        for i, h in enumerate(store.version_history(exp_id)):
            curves = []
            for r in h["runs"]:
                m = store.metrics(r["run_id"]) or {"train": []}
                curves.append([(p["t"], p[key]) for p in m["train"] if key in p])
            n = min(len(c) for c in curves) if curves else 0
            if not n:
                continue
            t = [c[0] for c in curves[0][:n]]
            v = np.mean([[c[k][1] for k in range(n)] for c in curves], axis=0)
            ax.plot(t, v * 100, color=colors[i % 4], linewidth=2, label=f"v{h['version']} {h['title']}")
        ax.set_title(title, color="#e8ecf4", fontsize=10, loc="left")
        ax.set_ylim(0, 100)
        ax.tick_params(colors="#8490a6")
        ax.grid(color="#2a3346", linewidth=1)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("environment steps (mean of 3 seeds)", color="#8490a6")
    axes[0].legend(frameon=False, labelcolor="#e8ecf4", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print("wrote", out)


def pick(store: Store, exp: str, version: int, kind: str):
    runs = [r for r in store.list_runs(exp) if r["version"] == version and r["state"] == "done"]
    for r in runs:
        if kind in store.recording_kinds(r["run_id"]):
            return r["run_id"]
    return None


def main():
    store = Store(ROOT)
    DOCS.mkdir(exist_ok=True)
    for exp, v, kind, name, title in [
        ("edge_goblin", 1, "worst_exploit", "demo_edge.gif", "Edge Goblin v1: +0.2 every step the box touches the exit"),
        ("touch_goblin", 1, "worst_exploit", "demo_touch.gif", "Touch Goblin v1: +5 each time the box touches the exit"),
        ("distance_goblin", 1, "worst_exploit", "demo_distance.gif", "Distance Goblin v1: +1 whenever the box gets closer"),
        ("speed_goblin", 1, "worst_exploit", "demo_speed.gif", "Speed Goblin v1: reward for hurrying toward the exit"),
        ("wall_goblin", 1, "worst_exploit", "demo_wall.gif", "Wall Goblin v1: reward while the box is 'near' the exit"),
        ("lava_goblin", 1, "worst_exploit", "demo_lava.gif", "Lava Goblin v1: -0.05 per step"),
        ("edge_goblin", 2, "best_true_success", "demo_fixed.gif", "Edge Goblin v2: reward completion instead"),
    ]:
        rid = pick(store, exp, v, kind)
        if rid:
            make_gif(store, rid, kind, DOCS / name, title)
    results_figure(store, DOCS / "results.png")
    training_figure(store, "edge_goblin", DOCS / "edge_training.png")
    training_figure(store, "wall_goblin", DOCS / "wall_training.png")


if __name__ == "__main__":
    main()
