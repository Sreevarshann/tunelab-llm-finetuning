import type { Evaluation, Job, Metrics, Run } from "./types";
import { isJson } from "./types";

/** Headline number for a split: all-fields-correct for JSON tasks, exact match for text tasks. */
export const headline = (m?: Metrics | null) => (m == null ? null : isJson(m) ? m.all_fields_correct_pct : m.exact_match_pct);
export const headlineLabel = (m?: Metrics | null) => (m && !isJson(m) ? "exact match" : "all fields correct");

export interface RunSummary {
  run: Run;
  tuned: Record<string, Evaluation | undefined>;
  baseline: Record<string, Evaluation | undefined>;
}

/** Joins evaluations to runs; baseline = newest completed base-model evaluation for the dataset/split. */
export function summarise(jobs: Job[]): RunSummary[] {
  const runs = jobs.filter((j) => j.kind === "train") as Run[];
  const evals = jobs.filter((j) => j.kind === "evaluate") as Evaluation[];
  return runs.map((run) => {
    const tuned: Record<string, Evaluation | undefined> = {};
    const baseline: Record<string, Evaluation | undefined> = {};
    for (const e of evals) {
      if (e.adapter_job_id === run.id) tuned[e.split!] = pick(tuned[e.split!], e);
      else if (e.adapter_job_id == null && e.dataset_id === run.dataset_id) baseline[e.split!] = pick(baseline[e.split!], e);
    }
    return { run, tuned, baseline };
  });
}

function pick(cur: Evaluation | undefined, e: Evaluation) {
  if (!cur) return e;
  if (cur.status !== "completed" && e.status === "completed") return e;
  if (cur.status === e.status && e.created_at > cur.created_at) return e;
  return cur;
}

export const configSummary = (c: Run["config"]) => `r${c.rank} · scale ${c.scale} · lr ${c.learning_rate} · ${c.iters} steps · ${c.num_layers} layers`;
