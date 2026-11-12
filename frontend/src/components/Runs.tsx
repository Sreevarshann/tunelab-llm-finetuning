"use client";

import { useQuery } from "@tanstack/react-query";
import { FlaskConical } from "lucide-react";
import Link from "next/link";
import { api } from "@/lib/api";
import { configSummary, headline, summarise } from "@/lib/summary";
import { StatusBadge } from "./Status";
import { Button, Card, Empty, ErrorNote, fmtLoss, fmtPct, fmtSec, PageHeader, relTime, Skeleton } from "./ui";

function Delta({ base, tuned }: { base: number | null; tuned: number | null }) {
  if (tuned == null) return <span className="text-muted">—</span>;
  const d = base == null ? null : tuned - base;
  return (
    <span className="tabular">
      <span className="font-medium text-tuned">{fmtPct(tuned)}</span>
      {d != null && <span className="ml-1.5 text-xs text-muted">{d >= 0 ? "+" : ""}{d.toFixed(1)} vs base {fmtPct(base)}</span>}
    </span>
  );
}

export function RunsOverview() {
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, refetchInterval: (q) => (q.state.data?.some((j) => j.status === "running" || j.status === "queued") ? 3000 : false) });
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const name = (id: string) => datasets.data?.find((d) => d.id === id)?.name ?? id;
  const runs = jobs.data ? summarise(jobs.data) : [];
  return (
    <div className="mx-auto max-w-6xl">
      <PageHeader title="Runs" subtitle="LoRA fine-tuning runs and how each adapter scores against the frozen base model on the same test data." actions={<Link href="/runs/new"><Button variant="primary"><FlaskConical className="size-4" /> New run</Button></Link>} />
      {jobs.error ? <ErrorNote error={jobs.error} onRetry={() => jobs.refetch()} /> : !jobs.data ? <Skeleton className="h-48" /> : !runs.length ? (
        <Empty icon={<FlaskConical className="size-5" />} title="No runs yet" action={<Link href="/runs/new"><Button variant="primary">Start a run</Button></Link>}>
          Create the synthetic ticket-triage dataset (or upload JSONL), then train a LoRA adapter on the 4-bit Qwen2.5-0.5B base.
        </Empty>
      ) : (
        <Card className="overflow-x-auto">
          <table className="w-full min-w-[760px] text-sm">
            <thead className="border-b border-line text-left text-xs text-muted">
              <tr><th className="px-4 py-2.5 font-medium">Run</th><th className="font-medium">Status</th><th className="font-medium">Val loss</th><th className="font-medium">Test</th><th className="pr-4 font-medium">Held-out templates</th></tr>
            </thead>
            <tbody className="divide-y divide-line">
              {runs.map(({ run, tuned, baseline }) => (
                <tr key={run.id} className="hover:bg-panel-2">
                  <td className="px-4 py-3">
                    <Link href={`/runs/${run.id}`} className="font-medium hover:underline">{run.name || run.id}</Link>
                    <div className="mt-0.5 font-mono text-[11px] text-muted">{configSummary(run.config)}</div>
                    <div className="text-[11px] text-muted">{name(run.dataset_id)} · {relTime(run.created_at)}{run.result ? ` · ${fmtSec(run.result.duration_s)}` : ""}</div>
                  </td>
                  <td><StatusBadge status={run.status} progress={run.progress} /></td>
                  <td className="tabular text-xs">{run.result ? <>{fmtLoss(run.result.initial_val_loss)} → <span className="font-medium text-tuned">{fmtLoss(run.result.final_val_loss)}</span></> : "—"}</td>
                  <td><Delta base={headline(baseline.test?.result?.metrics)} tuned={headline(tuned.test?.result?.metrics)} /></td>
                  <td className="pr-4"><Delta base={headline(baseline.test_heldout?.result?.metrics)} tuned={headline(tuned.test_heldout?.result?.metrics)} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </Card>
      )}
    </div>
  );
}
