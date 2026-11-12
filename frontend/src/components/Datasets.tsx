"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Database, Sparkles, Upload } from "lucide-react";
import Link from "next/link";
import { useRef, useState } from "react";
import { api, ApiError } from "@/lib/api";
import type { Dataset } from "@/lib/types";
import { Button, Card, Chip, Empty, ErrorNote, fmtNum, Input, PageHeader, relTime, Skeleton } from "./ui";

export function ProvenanceChip({ d }: { d: Dataset }) {
  return d.synthetic ? <Chip tone="warn" title={d.source}>synthetic</Chip> : <Chip tone="base">uploaded</Chip>;
}

const SPLIT_LABEL: Record<string, string> = { train: "Train", valid: "Validation", test: "Test", test_heldout: "Test · held-out templates" };
export const splitLabel = (s: string) => SPLIT_LABEL[s] ?? s;

export function DatasetsPage() {
  const qc = useQueryClient();
  const list = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const synth = useMutation({ mutationFn: () => api.createSynthetic({}), onSuccess: () => qc.invalidateQueries({ queryKey: ["datasets"] }) });
  const [name, setName] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const upload = useMutation({
    mutationFn: () => {
      const fd = new FormData();
      fd.set("name", name.trim());
      fd.set("file", fileRef.current!.files![0]);
      return api.upload(fd);
    },
    onSuccess: () => (setName(""), fileRef.current && (fileRef.current.value = ""), qc.invalidateQueries({ queryKey: ["datasets"] })),
  });
  return (
    <div className="mx-auto max-w-5xl">
      <PageHeader title="Datasets" subtitle="Chat-format JSONL. Validation, test and held-out splits are never used for training." />
      <div className="mb-6 grid gap-4 md:grid-cols-2 [&>*]:min-w-0">
        <Card className="space-y-3 p-5">
          <h2 className="flex items-center gap-2 text-sm font-semibold"><Sparkles className="size-4 text-warn" /> Synthetic support-ticket triage</h2>
          <p className="text-sm text-muted">Deterministic generator (seed 7): 600 train · 80 validation · 120 test · 120 test with phrasings never seen in training. Labels are exact because the text is rendered from them.</p>
          <Button onClick={() => synth.mutate()} disabled={synth.isPending}>{synth.isPending ? "Generating…" : "Generate dataset"}</Button>
          {synth.error && <ErrorNote error={synth.error} />}
        </Card>
        <Card className="space-y-3 p-5">
          <h2 className="flex items-center gap-2 text-sm font-semibold"><Upload className="size-4 text-base" /> Upload JSONL</h2>
          <p className="text-sm text-muted">One <code className="font-mono text-xs">{'{"messages": [...]}'}</code> per line, ending with the assistant reply. Split 80/10/10 by content hash. JSON replies with consistent keys are scored per field.</p>
          <form className="flex flex-col gap-2" onSubmit={(e) => (e.preventDefault(), upload.mutate())}>
            <Input aria-label="Dataset name" required maxLength={80} placeholder="Dataset name" value={name} onChange={(e) => setName(e.target.value)} />
            <input ref={fileRef} aria-label="JSONL file" type="file" required accept=".jsonl,.json,application/json,text/plain" className="text-sm file:mr-3 file:rounded-md file:border file:border-line file:bg-panel-2 file:px-3 file:py-1.5 file:text-sm" />
            <Button type="submit" disabled={upload.isPending}>{upload.isPending ? "Validating…" : "Upload"}</Button>
          </form>
          {upload.error && (
            <div className="space-y-1">
              <ErrorNote error={upload.error} />
              {upload.error instanceof ApiError && upload.error.details && <ul className="max-h-32 overflow-y-auto pl-4 font-mono text-[11px] text-bad">{upload.error.details.map((d) => <li key={d} className="list-disc">{d}</li>)}</ul>}
            </div>
          )}
        </Card>
      </div>
      {list.error ? <ErrorNote error={list.error} /> : !list.data ? <Skeleton className="h-32" /> : !list.data.length ? (
        <Empty icon={<Database className="size-5" />} title="No datasets yet">Generate the synthetic task above or upload your own JSONL.</Empty>
      ) : (
        <Card className="divide-y divide-line">
          {list.data.map((d) => (
            <Link key={d.id} href={`/datasets/${d.id}`} className="flex flex-wrap items-center gap-3 px-4 py-3 hover:bg-panel-2">
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2"><span className="truncate font-medium">{d.name}</span><ProvenanceChip d={d} /></div>
                <div className="mt-0.5 text-xs text-muted">{Object.entries(d.splits).map(([k, v]) => `${splitLabel(k).toLowerCase()} ${v.count}`).join(" · ")} · {relTime(d.created_at)}</div>
              </div>
              <Chip>{d.scoring.mode === "json_fields" ? `${d.scoring.fields.length} JSON fields` : "text match"}</Chip>
            </Link>
          ))}
        </Card>
      )}
    </div>
  );
}

export function DatasetDetail({ id }: { id: string }) {
  const d = useQuery({ queryKey: ["dataset", id], queryFn: () => api.dataset(id) });
  const [split, setSplit] = useState("train");
  const [page, setPage] = useState(0);
  const rows = useQuery({ queryKey: ["rows", id, split, page], queryFn: () => api.rows(id, split, page * 10, 10), enabled: !!d.data });
  if (d.error) return <ErrorNote error={d.error} />;
  if (!d.data) return <Skeleton className="h-64" />;
  const ds = d.data;
  return (
    <div className="mx-auto max-w-5xl">
      <PageHeader title={ds.name} subtitle={<span className="flex flex-wrap items-center gap-2"><ProvenanceChip d={ds} /><span>{ds.source}</span></span>} actions={<Link href={`/runs/new?dataset=${ds.id}`}><Button variant="primary">Train on this</Button></Link>} />
      <Card className="mb-5 overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="border-b border-line text-left text-xs text-muted"><tr><th className="px-4 py-2 font-medium">Split</th><th className="font-medium">Rows</th><th className="font-medium">Tokens mean / p95 / max</th><th className="pr-4 font-medium">SHA-256</th></tr></thead>
          <tbody className="divide-y divide-line">
            {Object.entries(ds.splits).map(([k, v]) => (
              <tr key={k}><td className="px-4 py-2">{splitLabel(k)}</td><td className="tabular">{fmtNum(v.count)}</td><td className="tabular">{v.tokens.mean ?? "—"} / {v.tokens.p95 ?? "—"} / {v.tokens.max ?? "—"}</td><td className="pr-4 font-mono text-[11px] text-muted">{v.sha256.slice(0, 16)}…</td></tr>
            ))}
          </tbody>
        </table>
      </Card>
      <p className="mb-3 text-sm text-muted">Scoring: {ds.scoring.mode === "json_fields" ? <>per-field accuracy on <span className="font-mono text-xs">{ds.scoring.fields.join(", ")}</span>, plus JSON validity and all-fields-correct</> : "normalised exact match and token F1"}; teacher-forced loss on every split.</p>
      <div className="mb-3 flex flex-wrap gap-1" role="tablist" aria-label="Split">
        {Object.keys(ds.splits).map((k) => (
          <button key={k} role="tab" aria-selected={split === k} onClick={() => (setSplit(k), setPage(0))} className={`rounded-md px-3 py-1.5 text-sm ${split === k ? "bg-ink text-bg" : "text-muted hover:bg-panel-2"}`}>{splitLabel(k)}</button>
        ))}
      </div>
      <div className="space-y-2">
        {rows.data?.rows.map((r, i) => (
          <Card key={i} className="grid gap-3 p-3 text-sm md:grid-cols-2 [&>*]:min-w-0">
            <div><div className="mb-1 text-[11px] uppercase tracking-wide text-muted">User</div><p>{r.messages.filter((m) => m.role === "user").at(-1)?.content}</p></div>
            <div><div className="mb-1 text-[11px] uppercase tracking-wide text-muted">Assistant (target)</div><pre className="whitespace-pre-wrap break-words font-mono text-xs">{r.messages.at(-1)?.content}</pre></div>
          </Card>
        ))}
        {rows.data && (
          <div className="flex items-center justify-between pt-1 text-xs text-muted">
            <span>{page * 10 + 1}–{Math.min((page + 1) * 10, rows.data.total)} of {rows.data.total}</span>
            <div className="flex gap-2">
              <Button size="sm" disabled={page === 0} onClick={() => setPage((p) => p - 1)}>Previous</Button>
              <Button size="sm" disabled={(page + 1) * 10 >= rows.data.total} onClick={() => setPage((p) => p + 1)}>Next</Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
