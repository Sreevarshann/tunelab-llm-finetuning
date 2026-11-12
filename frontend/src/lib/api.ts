import type { Dataset, Evaluation, Example, LoraParams, Run, RunDetail } from "./types";

export class ApiError extends Error {
  constructor(readonly status: number, message: string, readonly details?: string[]) {
    super(message);
  }
}

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await fetch(`/api${path}`, { ...init, headers: init.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json", ...init.headers } : init.headers });
  if (!res.ok) {
    let msg = res.statusText || `Request failed (${res.status})`;
    let details: string[] | undefined;
    try {
      const b = await res.json();
      const d = b.detail;
      if (typeof d === "string") msg = d;
      else if (Array.isArray(d)) msg = d.map((x: { loc?: string[]; msg?: string }) => `${x.loc?.slice(-1)[0] ?? ""}: ${x.msg}`).join("; ");
      else if (d?.message) {
        msg = d.message;
        details = d.errors;
      }
    } catch {}
    throw new ApiError(res.status, msg, details);
  }
  return (await res.json()) as T;
}

export interface Health { status: string; base_model: string; base_revision: string | null; mlx: string; mlx_lm: string; device: string; running_job: string | null }

export const api = {
  health: () => req<Health>("/health"),
  datasets: () => req<Dataset[]>("/datasets"),
  dataset: (id: string) => req<Dataset>(`/datasets/${id}`),
  rows: (id: string, split: string, offset = 0, limit = 10) => req<{ total: number; rows: { messages: { role: string; content: string }[] }[] }>(`/datasets/${id}/rows?split=${split}&offset=${offset}&limit=${limit}`),
  createSynthetic: (b: { name?: string; seed?: number; train?: number; valid?: number; test?: number; test_heldout?: number }) => req<Dataset>("/datasets/synthetic", { method: "POST", body: JSON.stringify(b) }),
  upload: (fd: FormData) => req<Dataset>("/datasets/upload", { method: "POST", body: fd }),
  runs: () => req<Run[]>("/runs"),
  run: (id: string) => req<RunDetail>(`/runs/${id}`),
  createRun: (b: { dataset_id: string; name: string; config: Partial<LoraParams> }) => req<Run>("/runs", { method: "POST", body: JSON.stringify(b) }),
  cancel: (id: string) => req<Run>(`/runs/${id}/cancel`, { method: "POST" }),
  reevaluate: (id: string) => req<{ queued: string[] }>(`/runs/${id}/evaluate`, { method: "POST" }),
  adapterUrl: (id: string) => `/api/runs/${id}/adapter`,
  examples: (evalId: string, onlyErrors = false, limit = 200) => req<{ total: number; rows: Example[] }>(`/evaluations/${evalId}/examples?only_errors=${onlyErrors}&limit=${limit}`),
  compare: (ids: string[]) => req<{ run: Run; evaluations: Evaluation[] }[]>(`/compare?run_ids=${ids.join(",")}`),
  jobs: () => req<(Run | Evaluation)[]>("/jobs"),
  playground: (b: { messages: { role: string; content: string }[]; run_ids: (string | null)[]; max_tokens?: number }) => req<{ outputs: { run_id: string | null; text: string }[] }>("/playground", { method: "POST", body: JSON.stringify(b) }),
  triagePrompt: () => req<{ system_prompt: string }>("/tasks/ticket_triage/system_prompt"),
};

export const TERMINAL = new Set(["completed", "failed", "cancelled"]);

/** Follows a job's SSE stream (stored events replayed first). Returns a close function. */
export function followJob(id: string, on: (type: string, data: Record<string, unknown>) => void, onEnd: () => void): () => void {
  const es = new EventSource(`/api/jobs/${id}/events`);
  for (const t of ["status", "stage", "train", "val", "progress"]) es.addEventListener(t, (m) => on(t, JSON.parse((m as MessageEvent).data)));
  es.addEventListener("end", () => {
    es.close();
    onEnd();
  });
  es.onerror = () => {
    if (es.readyState === EventSource.CLOSED) onEnd();
  };
  return () => es.close();
}
