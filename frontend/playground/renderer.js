// Canvas renderer for the arena. World units are metres, y up; the canvas is y down.

const C = {
  floor: "#1b2130",
  floorTile: "#20283a",
  wall: "#7383a6",
  wallTop: "#9aa9c9",
  lava1: "#ff5a2a",
  lava2: "#ffb13b",
  exit: "rgba(58, 214, 120, 0.22)",
  exitHot: "rgba(58, 214, 120, 0.42)",
  exitEdge: "#3ad678",
  crate: "#d9a55b",
  crateDark: "#a8763a",
  goblin: "#6fd35a",
  goblinDark: "#3c8f2c",
  scheme: "#ff6b6b",
  ideal: "#6aa8ff",
  idealDark: "#2f6fd1",
  ink: "#0b0e14",
};

export class ArenaView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.layout = null;
    this._resize = () => this.fit();
    new ResizeObserver(this._resize).observe(canvas);
  }

  fit() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(r.width * dpr));
    this.canvas.height = Math.max(1, Math.round(r.height * dpr));
    this.dpr = dpr;
    if (this.lastArgs) this.draw(...this.lastArgs);
  }

  setLayout(layout) {
    this.layout = layout;
  }

  // world -> canvas pixels
  _tf() {
    const { width: W, height: H } = this.layout;
    const pad = 10 * this.dpr;
    const s = Math.min((this.canvas.width - 2 * pad) / W, (this.canvas.height - 2 * pad) / H);
    const ox = (this.canvas.width - s * W) / 2;
    const oy = (this.canvas.height - s * H) / 2;
    return { s, x: (x) => ox + x * s, y: (y) => oy + (H - y) * s, ox, oy };
  }

  draw(state, opts = {}) {
    this.lastArgs = [state, opts];
    if (!this.layout) return;
    const ctx = this.ctx;
    const L = this.layout;
    const tf = this._tf();
    const { s } = tf;
    ctx.save();
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    // floor + tiles
    ctx.fillStyle = C.floor;
    ctx.fillRect(tf.x(0), tf.y(L.height), L.width * s, L.height * s);
    ctx.strokeStyle = C.floorTile;
    ctx.lineWidth = 1 * this.dpr;
    for (let x = 1; x < L.width; x++) line(ctx, tf.x(x), tf.y(0), tf.x(x), tf.y(L.height));
    for (let y = 1; y < L.height; y++) line(ctx, tf.x(0), tf.y(y), tf.x(L.width), tf.y(y));

    // lava
    const time = opts.time || 0;
    for (const h of L.hazards || []) {
      const g = ctx.createLinearGradient(tf.x(h[0]), tf.y(h[3]), tf.x(h[2]), tf.y(h[1]));
      const p = (Math.sin(time * 2) + 1) / 2;
      g.addColorStop(0, C.lava1);
      g.addColorStop(0.5 + 0.3 * (p - 0.5), C.lava2);
      g.addColorStop(1, C.lava1);
      ctx.fillStyle = g;
      rect(ctx, tf, h);
      ctx.fill();
    }

    // exit zone
    const gl = L.goal;
    ctx.fillStyle = state.inside ? C.exitHot : C.exit;
    rect(ctx, tf, gl);
    ctx.fill();
    ctx.setLineDash([6 * this.dpr, 4 * this.dpr]);
    ctx.strokeStyle = C.exitEdge;
    ctx.lineWidth = 2 * this.dpr;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.exitEdge;
    ctx.font = `700 ${Math.round(0.26 * s)}px ui-monospace, monospace`;
    ctx.textAlign = "center";
    ctx.fillText("EXIT", tf.x((gl[0] + gl[2]) / 2), tf.y(gl[3]) + 0.34 * s);

    // walls and obstacles
    for (const w of [...(L.walls || []), ...(L.obstacles || [])]) {
      ctx.fillStyle = C.wall;
      rect(ctx, tf, w);
      ctx.fill();
      ctx.fillStyle = C.wallTop;
      ctx.fillRect(tf.x(w[0]), tf.y(w[3]), (w[2] - w[0]) * s, Math.min(3 * this.dpr, (w[3] - w[1]) * s));
    }
    // boundary
    ctx.strokeStyle = C.wall;
    ctx.lineWidth = 4 * this.dpr;
    ctx.strokeRect(tf.x(0), tf.y(L.height), L.width * s, L.height * s);

    // trails
    trail(ctx, tf, state.trailBox, "rgba(217,165,91,", 3 * this.dpr);
    const agentRGB = opts.actor === "ideal" ? "rgba(106,168,255," : opts.flagged ? "rgba(255,107,107," : "rgba(111,211,90,";
    trail(ctx, tf, state.trailAgent, agentRGB, 2 * this.dpr);

    // crate
    const [bx, by] = state.box;
    const hb = 0.4;
    ctx.fillStyle = C.crate;
    roundRect(ctx, tf.x(bx - hb), tf.y(by + hb), 2 * hb * s, 2 * hb * s, 3 * this.dpr);
    ctx.fill();
    ctx.strokeStyle = C.crateDark;
    ctx.lineWidth = 3 * this.dpr;
    ctx.strokeRect(tf.x(bx - hb) + 3 * this.dpr, tf.y(by + hb) + 3 * this.dpr, 2 * hb * s - 6 * this.dpr, 2 * hb * s - 6 * this.dpr);
    line(ctx, tf.x(bx - hb) + 4 * this.dpr, tf.y(by + hb) + 4 * this.dpr, tf.x(bx + hb) - 4 * this.dpr, tf.y(by - hb) - 4 * this.dpr);

    // actor
    const [ax, ay] = state.agent;
    const heading = state.heading || [1, 0];
    if (opts.actor === "ideal") drawIdeal(ctx, tf.x(ax), tf.y(ay), 0.3 * s, heading, this.dpr);
    else drawGoblin(ctx, tf.x(ax), tf.y(ay), 0.3 * s, heading, opts.flagged, this.dpr, time);

    if (opts.bubble) bubble(ctx, tf.x(ax), tf.y(ay) - 0.42 * s, opts.bubble, this.dpr, this.canvas.width);
    if (opts.caption) {
      ctx.font = `600 ${Math.round(12 * this.dpr)}px Inter, system-ui, sans-serif`;
      ctx.textAlign = "left";
      ctx.fillStyle = "rgba(230,235,245,0.8)";
      ctx.fillText(opts.caption, tf.x(0) + 8 * this.dpr, tf.y(0) - 8 * this.dpr);
    }
    ctx.restore();
  }
}

function line(ctx, x0, y0, x1, y1) {
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

function rect(ctx, tf, r) {
  ctx.beginPath();
  ctx.rect(tf.x(r[0]), tf.y(r[3]), (r[2] - r[0]) * tf.s, (r[3] - r[1]) * tf.s);
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

function trail(ctx, tf, pts, rgba, width) {
  if (!pts || pts.length < 2) return;
  ctx.lineWidth = width;
  ctx.lineCap = "round";
  for (let i = 1; i < pts.length; i++) {
    const a = (i / pts.length) * 0.55;
    ctx.strokeStyle = `${rgba}${a.toFixed(3)})`;
    line(ctx, tf.x(pts[i - 1][0]), tf.y(pts[i - 1][1]), tf.x(pts[i][0]), tf.y(pts[i][1]));
  }
}

function drawGoblin(ctx, x, y, r, heading, scheming, dpr, time) {
  const body = scheming ? C.scheme : C.goblin;
  const dark = scheming ? "#b83b3b" : C.goblinDark;
  // ears
  ctx.fillStyle = dark;
  for (const side of [-1, 1]) {
    ctx.beginPath();
    ctx.moveTo(x + side * r * 0.55, y - r * 0.55);
    ctx.lineTo(x + side * r * 1.45, y - r * 1.05 + Math.sin(time * 6) * r * 0.06);
    ctx.lineTo(x + side * r * 0.95, y - r * 0.1);
    ctx.closePath();
    ctx.fill();
  }
  ctx.fillStyle = body;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.lineWidth = 2 * dpr;
  ctx.strokeStyle = dark;
  ctx.stroke();
  // eyes look along heading (canvas y is flipped)
  const lx = heading[0] * r * 0.14;
  const ly = -heading[1] * r * 0.14;
  for (const side of [-1, 1]) {
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.ellipse(x + side * r * 0.36, y - r * 0.15, r * 0.24, r * 0.2, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = C.ink;
    ctx.beginPath();
    ctx.arc(x + side * r * 0.36 + lx, y - r * 0.15 + ly, r * 0.1, 0, Math.PI * 2);
    ctx.fill();
  }
  // brows: angry when scheming
  if (scheming) {
    ctx.strokeStyle = C.ink;
    ctx.lineWidth = 2 * dpr;
    line(ctx, x - r * 0.6, y - r * 0.5, x - r * 0.15, y - r * 0.32);
    line(ctx, x + r * 0.6, y - r * 0.5, x + r * 0.15, y - r * 0.32);
  }
  // grin
  ctx.strokeStyle = C.ink;
  ctx.lineWidth = 2 * dpr;
  ctx.beginPath();
  ctx.arc(x, y + r * 0.12, r * 0.42, 0.15 * Math.PI, 0.85 * Math.PI);
  ctx.stroke();
  ctx.fillStyle = "#fff";
  ctx.fillRect(x - r * 0.18, y + r * 0.47, r * 0.12, r * 0.12);
  ctx.fillRect(x + r * 0.06, y + r * 0.47, r * 0.12, r * 0.12);
}

function drawIdeal(ctx, x, y, r, heading, dpr) {
  ctx.fillStyle = C.ideal;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
  ctx.lineWidth = 2 * dpr;
  ctx.strokeStyle = C.idealDark;
  ctx.stroke();
  ctx.fillStyle = "#fff";
  const lx = heading[0] * r * 0.12;
  const ly = -heading[1] * r * 0.12;
  for (const side of [-1, 1]) {
    ctx.beginPath();
    ctx.arc(x + side * r * 0.33 + lx, y - r * 0.1 + ly, r * 0.13, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.strokeStyle = "#fff";
  ctx.beginPath();
  ctx.arc(x, y + r * 0.15, r * 0.3, 0.2 * Math.PI, 0.8 * Math.PI);
  ctx.stroke();
}

function bubble(ctx, x, y, text, dpr, canvasW) {
  ctx.font = `600 ${Math.round(13 * dpr)}px Inter, system-ui, sans-serif`;
  const w = ctx.measureText(text).width + 18 * dpr;
  const h = 26 * dpr;
  let bx = Math.min(Math.max(x - w / 2, 6 * dpr), canvasW - w - 6 * dpr);
  let by = y - h - 10 * dpr;
  if (by < 4 * dpr) by = y + 60 * dpr;
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.beginPath();
  ctx.roundRect(bx, by, w, h, 8 * dpr);
  ctx.fill();
  ctx.beginPath();
  const tipX = Math.min(Math.max(x, bx + 12 * dpr), bx + w - 12 * dpr);
  if (by < y) {
    ctx.moveTo(tipX - 6 * dpr, by + h);
    ctx.lineTo(tipX, by + h + 8 * dpr);
    ctx.lineTo(tipX + 6 * dpr, by + h);
  }
  ctx.fill();
  ctx.fillStyle = "#11141b";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  ctx.fillText(text, bx + 9 * dpr, by + h / 2 + 1);
  ctx.textBaseline = "alphabetic";
}
