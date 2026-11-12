"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { api } from "@/lib/api";
import type { LoraParams } from "@/lib/types";
import { Button, Card, ErrorNote, Field, Input, PageHeader, Select } from "./ui";

const DEFAULTS: LoraParams = { rank: 8, scale: 20, dropout: 0, num_layers: 16, learning_rate: 1e-4, iters: 300, batch_size: 4, max_seq_length: 512, steps_per_report: 10, steps_per_eval: 50, seed: 0, mask_prompt: true };
const PRESETS: { id: string; label: string; hint: string; p: Partial<LoraParams> }[] = [
  { id: "quick", label: "Quick", hint: "100 steps", p: { iters: 100, steps_per_eval: 25 } },
  { id: "standard", label: "Standard", hint: "300 steps", p: { iters: 300 } },
  { id: "thorough", label: "Thorough", hint: "600 steps · rank 16", p: { iters: 600, rank: 16, steps_per_eval: 100 } },
];
const FIELDS: { k: keyof LoraParams; label: string; hint: string; step?: string }[] = [
  { k: "rank", label: "LoRA rank", hint: "1–128; low-rank update size" },
  { k: "scale", label: "Scale", hint: "multiplier on the LoRA update (mlx-lm 'scale')", step: "0.5" },
  { k: "num_layers", label: "Layers adapted", hint: "last N transformer blocks (model has 24)" },
  { k: "learning_rate", label: "Learning rate", hint: "Adam, constant", step: "0.00001" },
  { k: "iters", label: "Steps", hint: "optimiser updates" },
  { k: "batch_size", label: "Batch size", hint: "sequences per step" },
  { k: "dropout", label: "LoRA dropout", hint: "0 to <1", step: "0.05" },
  { k: "max_seq_length", label: "Max sequence length", hint: "tokens; longer sequences are truncated" },
  { k: "seed", label: "Seed", hint: "for initialisation and batching" },
];

export function NewRun() {
  const params = useSearchParams();
  const router = useRouter();
  const qc = useQueryClient();
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.datasets });
  const [ds, setDs] = useState(params.get("dataset") ?? "");
  const [name, setName] = useState("");
  const [cfg, setCfg] = useState<LoraParams>(DEFAULTS);
  const datasetId = ds || datasets.data?.[0]?.id || "";
  const create = useMutation({
    mutationFn: () => api.createRun({ dataset_id: datasetId, name: name.trim() || `r${cfg.rank} · lr ${cfg.learning_rate} · ${cfg.iters} steps`, config: cfg }),
    onSuccess: (r) => (qc.invalidateQueries({ queryKey: ["jobs"] }), router.push(`/runs/${r.id}`)),
  });
  return (
    <div className="mx-auto max-w-4xl">
      <PageHeader title="New LoRA run" subtitle="The base model stays frozen; only low-rank adapter matrices are trained, then evaluated against the base on the test splits." />
      {datasets.data?.length === 0 ? (
        <Card className="p-5 text-sm">No datasets yet. <Link href="/datasets" className="text-tuned underline">Create one first.</Link></Card>
      ) : (
        <form className="space-y-5" onSubmit={(e) => (e.preventDefault(), create.mutate())}>
          <Card className="grid gap-4 p-5 sm:grid-cols-2 [&>*]:min-w-0">
            <Field label="Dataset">
              <Select value={datasetId} onChange={(e) => setDs(e.target.value)}>
                {datasets.data?.map((d) => <option key={d.id} value={d.id}>{d.name}{d.synthetic ? " (synthetic)" : ""}</option>)}
              </Select>
            </Field>
            <Field label="Run name"><Input value={name} maxLength={80} onChange={(e) => setName(e.target.value)} placeholder="optional" /></Field>
          </Card>
          <Card className="space-y-4 p-5">
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((p) => (
                <button type="button" key={p.id} onClick={() => setCfg({ ...DEFAULTS, ...p.p })} className="rounded-lg border border-line px-3 py-2 text-left text-sm hover:bg-panel-2">
                  <div className="font-medium">{p.label}</div><div className="text-xs text-muted">{p.hint}</div>
                </button>
              ))}
            </div>
            <div className="grid gap-4 sm:grid-cols-3 [&>*]:min-w-0">
              {FIELDS.map((f) => (
                <Field key={f.k} label={f.label} hint={f.hint}>
                  <Input type="number" step={f.step ?? "1"} value={String(cfg[f.k])} onChange={(e) => setCfg({ ...cfg, [f.k]: Number(e.target.value) })} />
                </Field>
              ))}
            </div>
            <label className="flex items-center gap-2 text-sm">
              <input type="checkbox" checked={cfg.mask_prompt} onChange={(e) => setCfg({ ...cfg, mask_prompt: e.target.checked })} />
              Train on assistant tokens only (mask the prompt)
            </label>
          </Card>
          {create.error && <ErrorNote error={create.error} />}
          <div className="flex justify-end gap-2">
            <Link href="/"><Button variant="ghost">Cancel</Button></Link>
            <Button type="submit" variant="primary" disabled={!datasetId || create.isPending}>{create.isPending ? "Queuing…" : "Start training"}</Button>
          </div>
        </form>
      )}
    </div>
  );
}
