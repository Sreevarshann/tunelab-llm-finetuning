"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";
import { api } from "@/lib/api";
import { configSummary, headline, summarise } from "@/lib/summary";
import { splitLabel } from "./Datasets";
import { Plot, usePalette } from "./Plot";
import { Card, Empty, fmtPct, fmtSec, PageHeader, Skeleton } from "./ui";

export function Compare() {
  const c = usePalette();
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs });
  const all = jobs.data ? summarise(jobs.data).filter((s) => s.run.status === "completed") : [];
  const [picked, setPicked] = useState<string[] | null>(null);
  const sel = all.filter((s) => (picked ?? all.slice(0, 4).map((x) => x.run.id)).includes(s.run.id));
  const splits = ["test", "test_heldout"];
  const toggle = (id: string) => setPicked((p) => { const cur = p ?? sel.map((s) => s.run.id); return cur.includes(id) ? cur.filter((x) => x !== id) : [...cur, id].slice(-6); });
  if (!jobs.data) return <Skeleton className="h-64" />;
  return (
    <div className="mx-auto max-w-6xl">
      <PageHeader title="Compare runs" subtitle="Headline metric: all fields correct (JSON tasks) or exact match (text tasks). Base = frozen model on the same split." />
      {!all.length ? <Empty icon={<span>∅</span>} title="No completed runs">Train at least one run to compare.</Empty> : (
        <div className="space-y-5">
          <Card className="flex flex-wrap gap-2 p-3">
            {all.map(({ run }) => (
              <label key={run.id} className="flex items-center gap-2 rounded-md border border-line px-2.5 py-1.5 text-sm">
                <input type="checkbox" checked={sel.some((s) => s.run.id === run.id)} onChange={() => toggle(run.id)} />
                {run.name || run.id}
              </label>
            ))}
          </Card>
          <Card className="p-4">
            <Plot label="Headline metric by run and split" data={[
              ...splits.map((sp, i) => ({ type: "bar" as const, name: `LoRA · ${splitLabel(sp)}`, x: sel.map((s) => s.run.name || s.run.id), y: sel.map((s) => headline(s.tuned[sp]?.result?.metrics)), marker: { color: c.tuned, opacity: i ? 0.55 : 1 } })),
              ...splits.map((sp, i) => ({ type: "scatter" as const, mode: "markers" as const, name: `Base · ${splitLabel(sp)}`, x: sel.map((s) => s.run.name || s.run.id), y: sel.map((s) => headline(s.baseline[sp]?.result?.metrics)), marker: { color: c.base, symbol: (i ? "diamond-open" : "diamond") as "diamond", size: 11 } })),
            ]} layout={{ barmode: "group", yaxis: { range: [0, 100], ticksuffix: "%" } }} />
          </Card>
          <Card className="overflow-x-auto">
            <table className="w-full min-w-[720px] text-sm">
              <thead className="border-b border-line text-left text-xs text-muted"><tr><th className="px-4 py-2 font-medium">Run</th><th className="font-medium">Final val loss</th><th className="font-medium">Train time</th>{splits.map((s) => <th key={s} className="font-medium">{splitLabel(s)} · base → LoRA</th>)}</tr></thead>
              <tbody className="tabular divide-y divide-line">
                {sel.map(({ run, tuned, baseline }) => (
                  <tr key={run.id}>
                    <td className="px-4 py-2"><Link href={`/runs/${run.id}`} className="font-medium hover:underline">{run.name || run.id}</Link><div className="font-mono text-[11px] text-muted">{configSummary(run.config)}</div></td>
                    <td>{run.result?.final_val_loss.toExponential(2)}</td>
                    <td>{fmtSec(run.result?.duration_s)}</td>
                    {splits.map((s) => <td key={s}>{fmtPct(headline(baseline[s]?.result?.metrics))} → <span className="font-medium text-tuned">{fmtPct(headline(tuned[s]?.result?.metrics))}</span></td>)}
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </div>
      )}
    </div>
  );
}
