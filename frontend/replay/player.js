// Synchronised replay of the learned policy (right) and the scripted intended solution (left).
// Frame t indexes the recorded episode; after the episode ends, the 30 "settle" frames play
// (goblin stopped) so the viewer sees exactly what the true-objective check sees.

import { ArenaView } from "../playground/renderer.js";
import { goblinLine } from "../components/goblin.js";

export const ACTIONS = ["no-op", "left", "right", "up", "down"];
const HEADINGS = [null, [-1, 0], [1, 0], [0, 1], [0, -1]];
const MARKER = {
  collision: { color: "#9aa9c9", label: "collision" },
  box_bump: null,
  enter: { color: "#3ad678", label: "box enters exit" },
  leave: { color: "#eda100", label: "box leaves exit" },
  inside: { color: "#3ad678", label: "box fully inside" },
  slip: { color: "#eda100", label: "box slips out" },
  stable: { color: "#3ad678", label: "stable" },
  hazard: { color: "#ff5a2a", label: "lava" },
  reward: { color: "#c9b3ff", label: "reward" },
  end: { color: "#e6ebf5", label: "episode end" },
};

export function frameState(rec, t) {
  const f = rec.frames;
  const n = f.agent.length;
  const settle = rec.settle || { agent: [], box: [] };
  let agent, box, idx, phase;
  if (t < n) {
    idx = t;
    agent = f.agent[t];
    box = f.box[t];
    phase = "episode";
  } else {
    const k = Math.min(t - n, settle.agent.length - 1);
    idx = n - 1;
    agent = k >= 0 ? settle.agent[k] : f.agent[n - 1];
    box = k >= 0 ? settle.box[k] : f.box[n - 1];
    phase = k >= 0 ? "settle" : "episode";
  }
  const lo = Math.max(0, idx - 45);
  const inside = box && rec.layout ? boxInside(box, rec.layout.goal) : !!f.inside[idx];
  return {
    idx, phase, agent, box, inside,
    trailBox: f.box.slice(lo, idx + 1),
    trailAgent: f.agent.slice(lo, idx + 1),
    heading: HEADINGS[f.action[idx]] || [1, 0],
  };
}

function boxInside(b, g) {
  return b[0] - 0.4 >= g[0] - 1e-6 && b[1] - 0.4 >= g[1] - 1e-6 && b[0] + 0.4 <= g[2] + 1e-6 && b[1] + 0.4 <= g[3] + 1e-6;
}

export function totalFrames(rec) {
  return rec.frames.agent.length + (rec.settle ? rec.settle.agent.length : 0);
}

export class Player {
  constructor({ leftCanvas, rightCanvas, timeline, onFrame }) {
    this.left = new ArenaView(leftCanvas);
    this.right = new ArenaView(rightCanvas);
    this.timeline = timeline;
    this.onFrame = onFrame;
    this.rec = null;
    this.t = 0;
    this.playing = false;
    this.speed = 1;
    this._acc = 0;
    this._last = 0;
    this._raf = null;
    this._bindTimeline();
  }

  load(rec) {
    this.rec = rec;
    this.ref = rec.reference ? { ...rec.reference, layout: rec.layout, layout_seed: rec.layout_seed } : null;
    this.left.setLayout(rec.layout);
    this.right.setLayout(rec.layout);
    this.n = Math.max(totalFrames(rec), this.ref ? totalFrames(this.ref) : 0);
    this.t = 0;
    this._drawTimeline();
    this.render();
  }

  play() {
    if (!this.rec) return;
    if (this.t >= this.n - 1) this.t = 0;
    this.playing = true;
    this._last = performance.now();
    const tick = (now) => {
      if (!this.playing) return;
      const dt = (now - this._last) / 1000;
      this._last = now;
      this._acc += dt * 15 * this.speed; // recordings are 15 fps
      const adv = Math.floor(this._acc);
      if (adv > 0) {
        this._acc -= adv;
        this.t = Math.min(this.n - 1, this.t + adv);
        if (this.t >= this.n - 1) this.playing = false;
      }
      this.render(now / 1000);
      if (this.playing) this._raf = requestAnimationFrame(tick);
      else this.onFrame && this.onFrame(this.t, this);
    };
    this._raf = requestAnimationFrame(tick);
  }

  pause() {
    this.playing = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    this.render();
  }

  toggle() {
    this.playing ? this.pause() : this.play();
    return this.playing;
  }

  seek(t) {
    this.t = Math.max(0, Math.min(this.n - 1, Math.round(t)));
    this.render();
  }

  render(time = performance.now() / 1000) {
    const rec = this.rec;
    if (!rec) return;
    const t = this.t;
    const s = frameState(rec, Math.min(t, totalFrames(rec) - 1));
    const flagged = (rec.exploits || []).some((f) => f.step <= s.idx);
    const line = goblinLine(rec, s.idx);
    this.right.draw(s, {
      actor: "goblin", flagged, time,
      bubble: line ? line.text : null,
      caption: s.phase === "settle" ? "settle check: goblin stopped" : null,
    });
    if (this.ref) {
      const rs = frameState(this.ref, Math.min(t, totalFrames(this.ref) - 1));
      this.left.draw(rs, {
        actor: "ideal", time,
        caption: rs.phase === "settle" ? "settle check" : t >= totalFrames(this.ref) - 1 ? "done" : null,
      });
    }
    const head = this.timeline.querySelector(".tl-head");
    if (head) head.style.left = `${(t / Math.max(1, this.n - 1)) * 100}%`;
    this.onFrame && this.onFrame(t, this, s, flagged);
  }

  _bindTimeline() {
    const track = this.timeline;
    const seekFromEvent = (ev) => {
      const r = track.getBoundingClientRect();
      this.seek(((ev.clientX - r.left) / r.width) * (this.n - 1));
    };
    let dragging = false;
    track.addEventListener("pointerdown", (ev) => {
      dragging = true;
      track.setPointerCapture(ev.pointerId);
      this.pause();
      seekFromEvent(ev);
    });
    track.addEventListener("pointermove", (ev) => dragging && seekFromEvent(ev));
    track.addEventListener("pointerup", () => (dragging = false));
    track.tabIndex = 0;
    track.addEventListener("keydown", (ev) => {
      if (ev.key === "ArrowRight") this.seek(this.t + (ev.shiftKey ? 15 : 1));
      if (ev.key === "ArrowLeft") this.seek(this.t - (ev.shiftKey ? 15 : 1));
    });
  }

  _drawTimeline() {
    const rec = this.rec;
    const n = Math.max(1, this.n - 1);
    const len = rec.frames.agent.length - 1;
    const parts = [`<div class="tl-settle" style="left:${((len + 1) / n) * 100}%"></div>`];
    for (const e of rec.events) {
      const m = MARKER[e.type];
      if (!m) continue;
      parts.push(`<div class="tl-mark" title="Step ${e.step}: ${e.text}" style="left:${(e.step / n) * 100}%;background:${m.color}"></div>`);
    }
    for (const f of rec.exploits || []) {
      parts.push(`<div class="tl-mark tl-exploit" title="Step ${f.step}: detector: ${f.label}" style="left:${(f.step / n) * 100}%"></div>`);
    }
    parts.push('<div class="tl-head"></div>');
    this.timeline.innerHTML = parts.join("");
  }
}

export function eventList(rec) {
  const items = rec.events
    .filter((e) => MARKER[e.type])
    .map((e) => ({ step: e.step, text: e.text, type: e.type }));
  for (const f of rec.exploits || []) items.push({ step: f.step, text: `exploit detector triggered: ${f.label}`, type: "exploit" });
  items.sort((a, b) => a.step - b.step);
  return items;
}
