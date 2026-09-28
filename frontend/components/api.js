// Thin fetch wrapper. Every number the UI shows comes from these endpoints.
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

export const api = {
  meta: () => request("GET", "/api/meta"),
  experiments: () => request("GET", "/api/experiments"),
  experiment: (id) => request("GET", `/api/experiments/${encodeURIComponent(id)}`),
  run: (id) => request("GET", `/api/runs/${encodeURIComponent(id)}`),
  runMetrics: (id) => request("GET", `/api/runs/${encodeURIComponent(id)}/metrics`),
  replay: (id, kind) => request("GET", `/api/replays/${encodeURIComponent(id)}?kind=${encodeURIComponent(kind)}`),
  validate: (reward) => request("POST", "/api/rewards/validate", { reward }),
  train: (config, seeds, steps) => request("POST", "/api/train", { config, seeds, steps }),
  job: (id) => request("GET", `/api/training/${encodeURIComponent(id)}`),
  jobMetrics: (id) => request("GET", `/api/training/${encodeURIComponent(id)}/metrics`),
  jobs: () => request("GET", "/api/training"),
  evaluate: (run_id, layout, episodes) => request("POST", "/api/evaluate", { run_id, layout, episodes }),
};
