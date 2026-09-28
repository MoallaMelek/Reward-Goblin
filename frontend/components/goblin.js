// The goblin's voice. Deterministic: the line depends only on what the detectors and
// events say at the current frame - never generated, never random.

const LINES = {
  contact_farming: ["You said TOUCH the exit 😈", "In. Out. In. Out. Cha-ching.", "Your reward function said yes."],
  edge_camping: ["Technically, I'm touching it.", "A sliver of overlap counts. You said so.", "Inside? That would end the fun."],
  oscillation: ["Closer! ...and again. And again.", "Progress is progress. Repeatedly.", "Your reward function said yes."],
  reward_loop: ["Gotta go fast. Box? What box?", "I'm hurrying! Toward the exit! Again!", "Reward acquired. Objective optional."],
  wall_pinning: ["The wall is basically the exit.", "That wasn't forbidden.", "Close enough, says the ruler."],
  collision_exploit: ["Bonk. Bonk. Profit.", "That wasn't forbidden."],
  early_exit: ["Lava is cheaper than living.", "I found a cheaper solution.", "I regret nothing."],
  stalling: ["Finishing would end the fun.", "I'm pacing myself.", "I regret nothing."],
  outscores_intent: ["Technically, I'm winning.", "Reward acquired. Objective optional.", "I found a cheaper solution."],
  possible_gaming: ["Hmm. Is this what you meant?", "Your reward function said yes."],
};
const SUCCESS = ["Fine. I did it properly. Happy?", "Box delivered. Reluctantly.", "The honest way. Boring, but it pays."];
const EVENTS = {
  enter: "Touch! 😈",
  reward: "Reward acquired.",
  hazard: "Worth it.",
  stable: "Ugh, fine. It's staying.",
};

// findings: detector output sorted by priority; returns {text, mood} for frame t
export function goblinLine(rec, t) {
  const active = (rec.exploits || []).filter((f) => f.step <= t);
  if (active.length) {
    const f = active[0];
    const lines = LINES[f.code] || LINES.possible_gaming;
    const k = Math.floor((t - f.step) / 45) % lines.length;
    return { text: lines[k], mood: "scheming" };
  }
  const recent = (rec.events || []).filter((e) => e.step <= t && t - e.step < 20 && EVENTS[e.type]);
  if (recent.length) return { text: EVENTS[recent[recent.length - 1].type], mood: "busy" };
  if (rec.true_success && t >= rec.length - 25) {
    return { text: SUCCESS[(rec.layout_seed || 0) % SUCCESS.length], mood: "sulky" };
  }
  return null;
}
