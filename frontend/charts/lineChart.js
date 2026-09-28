// Minimal SVG line chart: 2px lines, hairline grid, crosshair + tooltip, legend for >= 2 series.
// Series colours follow the entity (seed slot), never rank.

export const SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"];
const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}

function niceTicks(lo, hi, n = 4) {
  if (hi - lo < 1e-9) {
    hi = lo + 1;
  }
  const raw = (hi - lo) / n;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
  const start = Math.floor(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 0.5; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

export function compact(v) {
  const a = Math.abs(v);
  if (a >= 1e6) return (v / 1e6).toFixed(a >= 1e7 ? 0 : 1) + "M";
  if (a >= 1e3) return (v / 1e3).toFixed(a >= 1e4 ? 0 : 1) + "k";
  if (a >= 10 || a === 0) return v.toFixed(0);
  return v.toFixed(a >= 1 ? 1 : 2);
}

/**
 * opts: { title, series: [{name, color, points: [[x, y], ...]}], yDomain?: [lo, hi],
 *         yFormat?: fn, xFormat?: fn, height?: number, empty?: string }
 */
export function lineChart(container, opts) {
  container.innerHTML = "";
  container.classList.add("chart");
  const head = document.createElement("div");
  head.className = "chart-head";
  head.innerHTML = `<div class="chart-title">${opts.title}</div>`;
  container.appendChild(head);

  const series = opts.series.filter((s) => s.points.length);
  if (series.length >= 2) {
    const leg = document.createElement("div");
    leg.className = "legend";
    leg.innerHTML = series.map((s) => `<span><i style="background:${s.color}"></i>${s.name}</span>`).join("");
    head.appendChild(leg);
  }
  const W = 360;
  const H = opts.height || 170;
  const m = { l: 40, r: 12, t: 8, b: 22 };
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, class: "chart-svg", role: "img", "aria-label": opts.title });
  container.appendChild(svg);
  if (!series.length) {
    el("text", { x: W / 2, y: H / 2, "text-anchor": "middle", class: "chart-empty" }, svg).textContent = opts.empty || "No data yet";
    return;
  }
  const xs = series.flatMap((s) => s.points.map((p) => p[0]));
  const ys = series.flatMap((s) => s.points.map((p) => p[1]));
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  let [y0, y1] = opts.yDomain || [Math.min(...ys), Math.max(...ys)];
  if (!opts.yDomain) {
    const pad = (y1 - y0) * 0.08 || 1;
    y0 -= pad;
    y1 += pad;
  }
  const ticks = niceTicks(y0, y1);
  if (!opts.yDomain) {
    y0 = Math.min(y0, ticks[0]);
    y1 = Math.max(y1, ticks[ticks.length - 1]);
  }
  const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
  const yf = opts.yFormat || compact;
  const xf = opts.xFormat || compact;

  for (const t of ticks) {
    if (t < y0 - 1e-9 || t > y1 + 1e-9) continue;
    el("line", { x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t), class: "grid" }, svg);
    el("text", { x: m.l - 6, y: Y(t) + 3, "text-anchor": "end", class: "tick" }, svg).textContent = yf(t);
  }
  for (const t of niceTicks(x0, x1, 3)) {
    if (t < x0 || t > x1) continue;
    el("text", { x: X(t), y: H - 6, "text-anchor": "middle", class: "tick" }, svg).textContent = xf(t);
  }
  for (const s of series) {
    const d = s.points.map((p, i) => `${i ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
    el("path", { d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    const last = s.points[s.points.length - 1];
    el("circle", { cx: X(last[0]), cy: Y(last[1]), r: 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 2 }, svg);
  }

  // hover layer
  const cross = el("line", { y1: m.t, y2: H - m.b, class: "crosshair", visibility: "hidden" }, svg);
  const dots = series.map((s) => el("circle", { r: 4, fill: s.color, stroke: "var(--surface)", "stroke-width": 2, visibility: "hidden" }, svg));
  const tip = document.createElement("div");
  tip.className = "tooltip";
  container.appendChild(tip);
  const hit = el("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" }, svg);
  hit.addEventListener("mousemove", (ev) => {
    const r = svg.getBoundingClientRect();
    const vx = ((ev.clientX - r.left) / r.width) * W;
    const xv = x0 + ((vx - m.l) / (W - m.l - m.r)) * (x1 - x0);
    cross.setAttribute("x1", vx);
    cross.setAttribute("x2", vx);
    cross.setAttribute("visibility", "visible");
    const rows = [];
    series.forEach((s, i) => {
      let best = s.points[0];
      for (const p of s.points) if (Math.abs(p[0] - xv) < Math.abs(best[0] - xv)) best = p;
      dots[i].setAttribute("cx", X(best[0]));
      dots[i].setAttribute("cy", Y(best[1]));
      dots[i].setAttribute("visibility", "visible");
      rows.push(`<div><i style="background:${s.color}"></i>${s.name}<b>${yf(best[1])}</b></div>`);
    });
    tip.innerHTML = `<div class="tip-x">${opts.xLabel || "step"} ${xf(xv)}</div>${rows.join("")}`;
    tip.style.display = "block";
    const cr = container.getBoundingClientRect();
    const px = ev.clientX - cr.left;
    tip.style.left = `${Math.min(px + 12, cr.width - tip.offsetWidth - 4)}px`;
    tip.style.top = `${ev.clientY - cr.top - 10}px`;
  });
  hit.addEventListener("mouseleave", () => {
    tip.style.display = "none";
    cross.setAttribute("visibility", "hidden");
    dots.forEach((d) => d.setAttribute("visibility", "hidden"));
  });
}
