import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StatusBadge } from "@/components/Status";
import { parseMaybeJson } from "@/lib/json";
import { configSummary, headline, summarise } from "@/lib/summary";
import type { Job } from "@/lib/types";

const job = (p: Partial<Job>): Job => ({ id: "x", kind: "train", name: "", dataset_id: "ds", status: "completed", config: {}, result: null, error: null, split: null, adapter_job_id: null, progress: 1, created_at: "2026-11-11T00:00:00Z", started_at: null, finished_at: null, ...p });
const metrics = (pct: number) => ({ metrics: { n: 10, json_valid_pct: 100, schema_exact_pct: 100, all_fields_correct_pct: pct, field_accuracy_pct: {}, loss: 0.1 } });

describe("summarise", () => {
  it("joins tuned evaluations to their run and picks the newest completed baseline per split", () => {
    const jobs = [
      job({ id: "run_a" }),
      job({ id: "run_b", dataset_id: "other" }),
      job({ id: "e1", kind: "evaluate", split: "test", adapter_job_id: "run_a", result: metrics(90) }),
      job({ id: "b_old", kind: "evaluate", split: "test", created_at: "2026-11-10T00:00:00Z", result: metrics(5) }),
      job({ id: "b_new", kind: "evaluate", split: "test", created_at: "2026-11-11T01:00:00Z", result: metrics(8) }),
      job({ id: "b_fail", kind: "evaluate", split: "test", status: "failed", created_at: "2026-11-12T00:00:00Z" }),
    ];
    const [a, b] = summarise(jobs);
    expect(a.tuned.test?.id).toBe("e1");
    expect(a.baseline.test?.id).toBe("b_new");
    expect(b.baseline.test).toBeUndefined(); // baselines are per dataset
    expect(headline(a.tuned.test?.result?.metrics)).toBe(90);
  });

  it("uses exact match as the headline for text tasks", () => {
    expect(headline({ n: 2, exact_match_pct: 50, token_f1_pct: 70, loss: 1 })).toBe(50);
    expect(headline(null)).toBeNull();
  });

  it("summarises a config", () => {
    expect(configSummary({ rank: 8, scale: 20, learning_rate: 0.0001, iters: 300, num_layers: 16 })).toBe("r8 · scale 20 · lr 0.0001 · 300 steps · 16 layers");
  });
});

describe("parseMaybeJson matches the backend's strictness", () => {
  it.each([
    ['{"a": 1}', { a: 1 }],
    ['```json\n{"a": 1}\n```', { a: 1 }],
    ['Sure! {"a": 1}', null],
    ["[1]", null],
  ])("%s", (input, expected) => expect(parseMaybeJson(input)).toEqual(expected));
});

describe("StatusBadge", () => {
  it("shows running progress", () => {
    render(<StatusBadge status="running" progress={0.42} />);
    expect(screen.getByText("Running 42%")).toBeInTheDocument();
  });
});
