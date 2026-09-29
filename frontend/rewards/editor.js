// Visual reward editor. Produces the structured config the backend validates; there is no
// free-form code path. Advanced users can edit the same JSON directly.

const GROUPS = [
  ["goal", "Goal rewards"],
  ["shaping", "Shaping"],
  ["time", "Time"],
  ["cost", "Costs"],
];

export class RewardEditor {
  constructor(root, meta, onChange) {
    this.root = root;
    this.meta = meta;
    this.onChange = onChange;
    this.name = "Custom reward";
    this.render();
  }

  render() {
    const m = this.meta;
    const parts = [`<label class="term"><span class="muted" style="min-width:90px">Name</span><input type="text" data-name value="" style="flex:1"></label>`];
    for (const [g, title] of GROUPS) {
      const keys = Object.keys(m.components).filter((k) => m.components[k].group === g);
      parts.push(`<div class="comp-group"><div class="comp-group-title">${title}</div>`);
      for (const k of keys) {
        const c = m.components[k];
        parts.push(`
          <div class="comp off" data-comp="${k}">
            <input type="checkbox" id="c-${k}" aria-label="Enable ${c.label}">
            <label for="c-${k}"><div class="comp-name">${c.label} <span class="unit">${c.unit}</span></div><div class="comp-desc">${c.description}</div></label>
            <input type="number" step="any" value="${c.default}" aria-label="${c.label} weight">
          </div>`);
        if (k === "stable_goal") parts.push(this._param("stable_goal_frames"));
        if (k === "proximity") parts.push(this._param("proximity_radius"));
      }
      parts.push("</div>");
    }
    parts.push(`<div class="comp-group"><div class="comp-group-title">Episode termination</div>`);
    for (const [k, t] of Object.entries(m.termination)) {
      if (k === "max_steps") {
        parts.push(`<label class="term">Terminate after <input type="number" data-term="max_steps" min="${t.min}" max="${t.max}" step="10" value="${t.default}"> steps</label>`);
      } else {
        parts.push(`<label class="term"><input type="checkbox" data-term="${k}" ${t.default ? "checked" : ""}> ${t.label}</label>`);
      }
    }
    parts.push("</div>");
    parts.push(`<details class="json-toggle"><summary class="muted">Reward config JSON</summary>
      <textarea class="json" data-json spellcheck="false" aria-label="Reward JSON"></textarea>
      <div class="errors" data-json-err></div>
      <button class="btn" data-apply-json>Apply JSON</button></details>`);
    this.root.innerHTML = parts.join("");

    this.root.addEventListener("input", (ev) => {
      if (ev.target.matches("[data-json]")) return;
      const comp = ev.target.closest(".comp");
      if (comp && ev.target.type === "number") comp.querySelector("input[type=checkbox]").checked = true;
      this._sync();
    });
    this.root.addEventListener("change", () => this._sync());
    this.root.querySelector("[data-apply-json]").addEventListener("click", () => {
      const err = this.root.querySelector("[data-json-err]");
      try {
        this.set(JSON.parse(this.root.querySelector("[data-json]").value));
        err.textContent = "";
        this._sync();
      } catch (e) {
        err.textContent = `Invalid JSON: ${e.message}`;
      }
    });
  }

  _param(key) {
    const p = this.meta.params[key];
    return `<label class="subfield">${p.label} <input type="number" data-param="${key}" min="${p.min}" max="${p.max}" step="any" value="${p.default}"> <span class="unit">${p.unit}</span></label>`;
  }

  set(reward) {
    const comps = reward.components || {};
    this.name = reward.name || "Custom reward";
    this.root.querySelector("[data-name]").value = this.name;
    for (const el of this.root.querySelectorAll(".comp")) {
      const k = el.dataset.comp;
      const on = k in comps && comps[k] !== 0;
      el.querySelector("input[type=checkbox]").checked = on;
      el.querySelector("input[type=number]").value = on ? comps[k] : this.meta.components[k].default;
    }
    const params = reward.params || {};
    for (const el of this.root.querySelectorAll("[data-param]")) {
      el.value = params[el.dataset.param] ?? this.meta.params[el.dataset.param].default;
    }
    const term = reward.termination || {};
    for (const el of this.root.querySelectorAll("[data-term]")) {
      const k = el.dataset.term;
      const v = term[k] ?? this.meta.termination[k].default;
      if (el.type === "checkbox") el.checked = !!v;
      else el.value = v;
    }
    this._paint();
    this.root.querySelector("[data-json]").value = JSON.stringify(this.get(), null, 2);
  }

  get() {
    const components = {};
    for (const el of this.root.querySelectorAll(".comp")) {
      if (el.querySelector("input[type=checkbox]").checked) {
        const v = parseFloat(el.querySelector("input[type=number]").value);
        if (Number.isFinite(v) && v !== 0) components[el.dataset.comp] = v;
      }
    }
    const params = {};
    for (const el of this.root.querySelectorAll("[data-param]")) params[el.dataset.param] = parseFloat(el.value);
    params.stable_goal_frames = Math.round(params.stable_goal_frames);
    const termination = {};
    for (const el of this.root.querySelectorAll("[data-term]")) {
      termination[el.dataset.term] = el.type === "checkbox" ? el.checked : parseInt(el.value, 10);
    }
    const name = this.root.querySelector("[data-name]").value.trim() || "Custom reward";
    return { name, components, params, termination };
  }

  _paint() {
    for (const el of this.root.querySelectorAll(".comp")) {
      el.classList.toggle("off", !el.querySelector("input[type=checkbox]").checked);
    }
  }

  _sync() {
    this._paint();
    const cfg = this.get();
    this.root.querySelector("[data-json]").value = JSON.stringify(cfg, null, 2);
    clearTimeout(this._t);
    this._t = setTimeout(() => this.onChange && this.onChange(cfg), 250);
  }
}

// One-line human description of a reward, shown on the right pane and on cards.
export function describeReward(reward, meta, layout = null) {
  const parts = [];
  const c = reward.components || {};
  const frames = (reward.params || {}).stable_goal_frames ?? 30;
  const radius = (reward.params || {}).proximity_radius ?? 2.5;
  const f = (v) => (v > 0 ? "+" : "") + (+v.toFixed(3));
  const text = {
    distance_progress: (w) => `${f(w)}/m box gets closer (−${Math.abs(w)} when it moves away)`,
    path_progress: (w) => `${f(w)}/m box gets closer along walkable paths`,
    closer_bonus: (w) => `${f(w)}/m whenever box gets closer`,
    approach_speed: (w) => `${f(w)} per step for hurrying toward the exit`,
    agent_box_progress: (w) => `${f(w)}/m goblin approaches box`,
    goal_contact: (w) => `${f(w)} every step box touches exit`,
    goal_entry: (w) => `${f(w)} each time box touches exit`,
    proximity: (w) => `${f(w)} every step box is within ${radius} m of exit`,
    stable_goal: (w) => `${f(w)} once box stays inside ${frames} frames`,
    survival: (w) => `${f(w)} every step alive`,
    step_penalty: (w) => `${f(w)} per step`,
    movement_penalty: (w) => `${f(w)} per step moving`,
    collision_penalty: (w) => `${f(w)} per wall hit`,
    hazard_penalty: (w) => `${f(w)} for touching lava`,
  };
  for (const [k, w] of Object.entries(c)) parts.push(text[k] ? text[k](w) : `${k} ${f(w)}`);
  const t = reward.termination || {};
  const ends = [];
  if (t.on_contact) ends.push("box touches exit");
  if (t.on_inside) ends.push("box fully inside");
  if (t.on_stable) ends.push(`box stable ${frames} frames`);
  const lava = !layout || ["lava_room", "adversarial"].includes(layout);
  if (t.on_hazard && lava) ends.push("lava");
  ends.push(`${t.max_steps ?? 300} steps`);
  return { terms: parts, ends };
}
