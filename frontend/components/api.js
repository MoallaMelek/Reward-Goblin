// Thin fetch wrapper. Every number the UI shows comes from these endpoints.
// In a static export (scripts/export_static.py) the same calls read snapshotted JSON files.

export const STATIC = !!window.REWARD_GOBLIN_STATIC;

async function request(method, path, body) {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const detail = data && data.detail ? (typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail)) : res.statusText;
    throw new Error(detail);
  }
  return data;
}

const enc = encodeURIComponent;
const live = {
  meta: () => request("GET", "/api/meta"),
  experiments: () => request("GET", "/api/experiments"),
  experiment: (id) => request("GET", `/api/experiments/${enc(id)}`),
  run: (id) => request("GET", `/api/runs/${enc(id)}`),
  runMetrics: (id) => request("GET", `/api/runs/${enc(id)}/metrics`),
  replay: (id, kind) => request("GET", `/api/replays/${enc(id)}?kind=${enc(kind)}`),
  validate: (reward, fingerprint, layout) => request("POST", "/api/rewards/validate", { reward, layout }),
  train: (config, seeds, steps) => request("POST", "/api/train", { config, seeds, steps }),
  job: (id) => request("GET", `/api/training/${enc(id)}`),
  jobMetrics: (id) => request("GET", `/api/training/${enc(id)}/metrics`),
  jobs: () => request("GET", "/api/training"),
  evaluate: (run_id, layout, episodes) => request("POST", "/api/evaluate", { run_id, layout, episodes }),
};

let lintCache = null;
const offline = () => Promise.reject(new Error("This is the static demo. Run the FastAPI backend locally to train goblins."));
const snapshot = {
  meta: () => request("GET", "api/meta.json"),
  experiments: () => request("GET", "api/experiments.json"),
  experiment: (id) => request("GET", `api/experiments/${enc(id)}.json`),
  run: (id) => request("GET", `api/runs/${enc(id)}.json`),
  runMetrics: (id) => request("GET", `api/runs/${enc(id)}/metrics.json`),
  replay: (id, kind) => request("GET", `api/replays/${enc(id)}/${enc(kind)}.json`),
  // Lint was precomputed for the gallery rewards; edited rewards need the backend.
  validate: async (reward, fingerprint) => {
    lintCache = lintCache || (await request("GET", "api/lint.json"));
    return lintCache[fingerprint] || { ok: true, errors: [], warnings: [], offline: true };
  },
  train: offline, job: offline, jobMetrics: offline, jobs: async () => [], evaluate: offline,
};

export const api = STATIC ? snapshot : live;
