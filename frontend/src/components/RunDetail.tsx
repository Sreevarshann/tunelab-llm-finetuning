"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Ban, Download, MessagesSquare, ShieldCheck } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, followJob, TERMINAL } from "@/lib/api";
import { configSummary } from "@/lib/summary";
import type { CurvePoint, Evaluation, Example, JsonMetrics, Metrics, RunDetail as RunT } from "@/lib/types";
import { isJson } from "@/lib/types";
import { splitLabel } from "./Datasets";
import { Plot, usePalette } from "./Plot";
import { StatusBadge } from "./Status";
import { Button, Card, Chip, ErrorNote, fmtDate, fmtLoss, fmtNum, fmtPct, fmtSec, PageHeader, Skeleton } from "./ui";

interface Live { stage?: string; train: (CurvePoint & { tokens_per_second?: number })[]; val: CurvePoint[] }

function useLive(id: string, onEnd: () => void): Live {
  const [s, setS] = useState<{ id: string; live: Live }>({ id, live: { train: [], val: [] } });
  const end = useRef(onEnd);
  useEffect(() => {
    end.current = onEnd;
  });
  useEffect(
    () =>
      followJob(
        id,
        (type, data) =>
          setS((prev) => {
            const live = prev.id === id ? prev.live : { train: [], val: [] };
            if (type === "train") return { id, live: { ...live, train: [...live.train, data as unknown as CurvePoint] } };
            if (type === "val") return { id, live: { ...live, val: [...live.val, data as unknown as CurvePoint] } };
            if (type === "stage") return { id, live: { ...live, stage: String(data.stage) } };
            return { id, live };
          }),
        () => end.current(),
      ),
    [id],
  );
  return s.id === id ? s.live : { train: [], val: [] };
}

function LossChart({ train, val }: { train: CurvePoint[]; val: CurvePoint[] }) {
  const c = usePalette();
  if (!train.length && !val.length) return <div className="grid h-64 place-items-center text-sm text-muted">Waiting for the first loss report…</div>;
  return (
    <Plot
      label="Training and validation loss by step"
      data={[
        { type: "scatter", mode: "lines+markers", name: "train loss", x: train.map((p) => p.iteration), y: train.map((p) => p.loss), line: { color: c.tuned, width: 2 }, marker: { size: 4 } },
        { type: "scatter", mode: "lines+markers", name: "validation loss", x: val.map((p) => p.iteration), y: val.map((p) => p.loss), line: { color: c.base, width: 2, dash: "dot" }, marker: { size: 6 } },
      ]}
      layout={{ xaxis: { title: { text: "step" } }, yaxis: { title: { text: "loss" }, type: "log" } }}
    />
  );
}

function Verification({ run }: { run: RunT }) {
  const r = run.result!;
  const rows: Record<string, React.ReactNode> = {
    "Trainable parameters": <>{fmtNum(r.trainable_params)} <span className="text-muted">of {fmtNum(r.total_params)} ({((100 * r.trainable_params) / r.total_params).toFixed(2)}%)</span></>,
    "LoRA B norm": <>{r.lora_b_norm_before.toFixed(3)} → <span className="font-medium text-tuned">{r.lora_b_norm_after.toFixed(3)}</span> <span className="text-muted">(starts at 0: weights were updated)</span></>,
    "Validation loss": <>{fmtLoss(r.initial_val_loss)} → <span className="font-medium text-tuned">{fmtLoss(r.final_val_loss)}</span></>,
    "Adapter SHA-256": <span className="break-all font-mono text-[11px]">{r.adapter_sha256}</span>,
    "Training time · peak memory": <>{fmtSec(r.duration_s)} · {r.peak_memory_gb} GB</>,
    "Throughput": <>{r.train_curve.length ? Math.round(r.train_curve.reduce((a, p) => a + p.tokens_per_second, 0) / r.train_curve.length) : "—"} tokens/s (mean of reports)</>,
  };
  return (
    <Card className="p-4">
      <h2 className="mb-3 flex items-center gap-2 text-sm font-semibold"><ShieldCheck className="size-4 text-tuned" /> Training verification</h2>
      <dl className="space-y-2 text-sm">
        {Object.entries(rows).map(([k, v]) => <div key={k} className="grid gap-1 sm:grid-cols-[180px_1fr]"><dt className="text-muted">{k}</dt><dd className="tabular min-w-0">{v}</dd></div>)}
      </dl>
    </Card>
  );
}

const fieldsOf = (bm?: Metrics, tm?: Metrics) => Object.keys((tm && isJson(tm) ? tm : bm && isJson(bm) ? bm : undefined)?.field_accuracy_pct ?? {});

const METRIC_ROWS: [string, (m: JsonMetrics) => number][] = [
  ["JSON valid", (m) => m.json_valid_pct],
  ["Exact schema", (m) => m.schema_exact_pct],
  ["All fields correct", (m) => m.all_fields_correct_pct],
];

function SplitComparison({ split, base, tuned, mode }: { split: string; base?: Evaluation; tuned?: Evaluation; mode: string }) {
  const c = usePalette();
  const bm = base?.result?.metrics as Metrics | undefined;
  const tm = tuned?.result?.metrics as Metrics | undefined;
  const pending = (e?: Evaluation) => !e ? "not queued" : e.status === "completed" ? null : e.status === "running" ? `running ${Math.round(e.progress * 100)}%` : e.status;
  const cell = (v?: number) => fmtPct(v);
  const delta = (a?: number, b?: number) => a == null || b == null ? "" : `${b - a >= 0 ? "+" : ""}${(b - a).toFixed(1)}`;
  const json = mode === "json_fields";
  return (
    <Card className="p-4">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="text-sm font-semibold">{splitLabel(split)}</h3>
        <span className="text-xs text-muted">n = {tm?.n ?? bm?.n ?? "—"} · greedy decoding · same prompt for both</span>
      </div>
      {(pending(base) || pending(tuned)) && <p className="mb-2 text-xs text-muted">Baseline: {pending(base) ?? "done"} · Tuned: {pending(tuned) ?? "done"}</p>}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead className="text-left text-xs text-muted"><tr><th className="py-1.5 font-medium">Metric</th><th className="font-medium text-base">Base</th><th className="font-medium text-tuned">LoRA</th><th className="font-medium">Δ</th></tr></thead>
          <tbody className="tabular divide-y divide-line">
            {json ? (
              <>
                {METRIC_ROWS.map(([label, f]) => { const a = bm && isJson(bm) ? f(bm) : undefined; const b = tm && isJson(tm) ? f(tm) : undefined; return <tr key={label}><td className="py-1.5">{label}</td><td>{cell(a)}</td><td className="font-medium">{cell(b)}</td><td className="text-muted">{delta(a, b)}</td></tr>; })}
                {(fieldsOf(bm, tm)).map((f) => { const a = bm && isJson(bm) ? bm.field_accuracy_pct[f] : undefined; const b = tm && isJson(tm) ? tm.field_accuracy_pct[f] : undefined; return <tr key={f}><td className="py-1.5 pl-3 font-mono text-xs">{f}</td><td>{cell(a)}</td><td>{cell(b)}</td><td className="text-muted">{delta(a, b)}</td></tr>; })}
              </>
            ) : (
              <>
                <tr><td className="py-1.5">Exact match</td><td>{cell(bm && !isJson(bm) ? bm.exact_match_pct : undefined)}</td><td className="font-medium">{cell(tm && !isJson(tm) ? tm.exact_match_pct : undefined)}</td><td /></tr>
                <tr><td className="py-1.5">Token F1</td><td>{cell(bm && !isJson(bm) ? bm.token_f1_pct : undefined)}</td><td>{cell(tm && !isJson(tm) ? tm.token_f1_pct : undefined)}</td><td /></tr>
              </>
            )}
            <tr><td className="py-1.5">Test loss</td><td>{fmtLoss(bm?.loss)}</td><td>{fmtLoss(tm?.loss)}</td><td /></tr>
          </tbody>
        </table>
      </div>
      {bm && tm && isJson(bm) && isJson(tm) && (
        <Plot height={220} label={`Field accuracy, base vs LoRA, ${splitLabel(split)}`} data={[
          { type: "bar", name: "base", x: Object.keys(bm.field_accuracy_pct), y: Object.values(bm.field_accuracy_pct), marker: { color: c.base } },
          { type: "bar", name: "LoRA", x: Object.keys(tm.field_accuracy_pct), y: Object.values(tm.field_accuracy_pct), marker: { color: c.tuned } },
        ]} layout={{ barmode: "group", yaxis: { range: [0, 100], ticksuffix: "%" } }} />
      )}
    </Card>
  );
}

function ExamplesBrowser({ splits, base, tuned }: { splits: string[]; base: Record<string, Evaluation | undefined>; tuned: Record<string, Evaluation | undefined> }) {
  const [split, setSplit] = useState(splits.at(-1)!);
  const [filter, setFilter] = useState<"changed" | "tuned_wrong" | "all">("changed");
  const b = base[split], t = tuned[split];
  const ready = b?.status === "completed" && t?.status === "completed";
  const be = useQuery({ queryKey: ["examples", b?.id], queryFn: () => api.examples(b!.id, false, 500), enabled: ready });
  const te = useQuery({ queryKey: ["examples", t?.id], queryFn: () => api.examples(t!.id, false, 500), enabled: ready });
  const ok = (e?: Example) => e?.all_correct ?? e?.exact ?? false;
  const pairs = useMemo(() => {
    if (!be.data || !te.data) return [];
    return te.data.rows.map((e, i) => ({ tuned: e, base: be.data.rows[i] })).filter((p) => filter === "all" || (filter === "tuned_wrong" ? !ok(p.tuned) : ok(p.tuned) !== ok(p.base)));
  }, [be.data, te.data, filter]);
  if (!splits.length) return null;
  return (
    <Card className="p-4">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h2 className="mr-auto text-sm font-semibold">Examples: base vs LoRA</h2>
        {splits.map((s) => <button key={s} onClick={() => setSplit(s)} aria-pressed={split === s} className={`rounded-md px-2.5 py-1 text-xs ${split === s ? "bg-ink text-bg" : "text-muted hover:bg-panel-2"}`}>{splitLabel(s)}</button>)}
        <select aria-label="Example filter" value={filter} onChange={(e) => setFilter(e.target.value as typeof filter)} className="h-8 rounded-md border border-line bg-panel px-2 text-xs">
          <option value="changed">Outcome changed</option>
          <option value="tuned_wrong">LoRA still wrong</option>
          <option value="all">All</option>
        </select>
      </div>
      {!ready ? <p className="py-6 text-center text-sm text-muted">Available when both evaluations finish.</p> : !pairs.length ? <p className="py-6 text-center text-sm text-muted">No examples match.</p> : (
        <div className="max-h-[640px] space-y-3 overflow-y-auto pr-1">
          {pairs.slice(0, 60).map(({ base: bx, tuned: tx }) => (
            <div key={tx.index} className="rounded-lg border border-line p-3 text-sm">
              <div className="mb-2 flex gap-2 text-xs text-muted"><span>#{tx.index}</span><span className="min-w-0 flex-1">{tx.input}</span></div>
              <div className="grid gap-2 md:grid-cols-3 [&>*]:min-w-0">
                <Output title="Reference" text={tx.reference} />
                <Output title="Base" text={bx?.output ?? ""} ok={ok(bx)} fields={bx?.fields} tone="base" />
                <Output title="LoRA" text={tx.output} ok={ok(tx)} fields={tx.fields} tone="tuned" />
              </div>
            </div>
          ))}
          {pairs.length > 60 && <p className="text-center text-xs text-muted">Showing 60 of {pairs.length}</p>}
        </div>
      )}
    </Card>
  );
}

function Output({ title, text, ok, fields, tone }: { title: string; text: string; ok?: boolean; fields?: Record<string, boolean>; tone?: "base" | "tuned" }) {
  return (
    <div className="rounded-md bg-panel-2 p-2">
      <div className="mb-1 flex flex-wrap items-center gap-1 text-[11px]">
        <span className={tone === "base" ? "font-medium text-base" : tone === "tuned" ? "font-medium text-tuned" : "text-muted"}>{title}</span>
        {ok != null && <Chip tone={ok ? "tuned" : "bad"}>{ok ? "correct" : "wrong"}</Chip>}
        {fields && Object.entries(fields).filter(([, v]) => !v).map(([f]) => <Chip key={f} tone="bad">{f}</Chip>)}
      </div>
      <pre className="whitespace-pre-wrap break-words font-mono text-[11px]">{text || "(empty)"}</pre>
    </div>
  );
}

export function RunDetail({ id }: { id: string }) {
  const qc = useQueryClient();
  const q = useQuery({ queryKey: ["run", id], queryFn: () => api.run(id), refetchInterval: (s) => (s.state.data && (!TERMINAL.has(s.state.data.status) || s.state.data.evaluations.some((e) => !TERMINAL.has(e.status))) ? 3000 : false) });
  const live = useLive(id, () => qc.invalidateQueries({ queryKey: ["run", id] }));
  const cancel = useMutation({ mutationFn: () => api.cancel(id), onSuccess: () => qc.invalidateQueries({ queryKey: ["run", id] }) });
  const reeval = useMutation({ mutationFn: () => api.reevaluate(id), onSuccess: () => qc.invalidateQueries({ queryKey: ["run", id] }) });
  if (q.error) return <ErrorNote error={q.error} />;
  if (!q.data) return <Skeleton className="h-96" />;
  const run = q.data;
  const tuned: Record<string, Evaluation | undefined> = {};
  const base: Record<string, Evaluation | undefined> = {};
  for (const e of run.evaluations) (e.adapter_job_id ? tuned : base)[e.split!] = e;
  const splits = ["test", "test_heldout"].filter((s) => s in run.dataset.splits);
  const train = run.result?.train_curve ?? live.train;
  const val = run.result?.val_curve?.length ? [...run.result.val_curve, { iteration: run.config.iters ?? 0, loss: run.result.final_val_loss }] : live.val;
  return (
    <div className="mx-auto max-w-6xl space-y-5">
      <Link href="/" className="inline-flex items-center gap-1 text-xs text-muted hover:text-ink"><ArrowLeft className="size-3" />Runs</Link>
      <PageHeader
        title={run.name || run.id}
        subtitle={<span className="flex flex-wrap items-center gap-2"><StatusBadge status={run.status} progress={run.progress} /><span className="font-mono text-xs">{configSummary(run.config)}</span><Link className="underline" href={`/datasets/${run.dataset.id}`}>{run.dataset.name}</Link><span>{fmtDate(run.created_at)}</span></span>}
        actions={
          <>
            {!TERMINAL.has(run.status) && <Button size="sm" variant="danger" onClick={() => cancel.mutate()} disabled={cancel.isPending}><Ban className="size-3.5" />Cancel</Button>}
            {run.status === "completed" && <a href={api.adapterUrl(run.id)}><Button size="sm"><Download className="size-3.5" />Adapter</Button></a>}
            {run.status === "completed" && <Link href={`/playground?run=${run.id}`}><Button size="sm"><MessagesSquare className="size-3.5" />Try it</Button></Link>}
          </>
        }
      />
      {run.error && <ErrorNote error={run.error} />}
      {run.status === "running" && (
        <div className="h-1.5 overflow-hidden rounded-full bg-panel-2" role="progressbar" aria-label="Training progress" aria-valuenow={Math.round(run.progress * 100)} aria-valuemin={0} aria-valuemax={100}>
          <div className="h-full bg-tuned transition-[width]" style={{ width: `${run.progress * 100}%` }} />
        </div>
      )}
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_400px] [&>*]:min-w-0">
        <Card className="p-4">
          <div className="mb-1 flex items-center justify-between"><h2 className="text-sm font-semibold">Loss</h2><span className="text-xs text-muted">{run.status === "running" ? `live · ${live.stage ?? "starting"}` : run.status}</span></div>
          <LossChart train={train} val={val} />
        </Card>
        {run.result ? <Verification run={run} /> : <Card className="p-4 text-sm text-muted">Verification (trainable parameters, adapter checksum, weight-update check) appears when training finishes.</Card>}
      </div>
      {run.status === "completed" && (
        <>
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold">Base model vs LoRA adapter</h2>
            {splits.some((s) => !tuned[s]) && <Button size="sm" onClick={() => reeval.mutate()} disabled={reeval.isPending}>Queue evaluations</Button>}
          </div>
          <div className="grid gap-5 lg:grid-cols-2 [&>*]:min-w-0">{splits.map((s) => <SplitComparison key={s} split={s} base={base[s]} tuned={tuned[s]} mode={run.dataset.scoring.mode} />)}</div>
          <ExamplesBrowser splits={splits} base={base} tuned={tuned} />
        </>
      )}
    </div>
  );
}


