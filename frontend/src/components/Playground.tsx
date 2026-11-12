"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { useSearchParams } from "next/navigation";
import { useState } from "react";
import { api } from "@/lib/api";
import { parseMaybeJson } from "@/lib/json";
import { Button, Card, Chip, ErrorNote, Field, PageHeader, Select } from "./ui";

const SAMPLE = "Hello support, the Nimbus Backup restore fails with a corrupted-data error on order ORD-55120. Our whole team is blocked. Please refund me if this cannot be fixed. - Dana";

export function Playground() {
  const params = useSearchParams();
  const runs = useQuery({ queryKey: ["runs"], queryFn: api.runs });
  const prompt = useQuery({ queryKey: ["triage-prompt"], queryFn: api.triagePrompt });
  const done = runs.data?.filter((r) => r.status === "completed") ?? [];
  const [runId, setRunId] = useState(params.get("run") ?? "");
  const chosen = runId || done[0]?.id || "";
  const [system, setSystem] = useState<string | null>(null);
  const [user, setUser] = useState(SAMPLE);
  const sys = system ?? prompt.data?.system_prompt ?? "";
  const go = useMutation({ mutationFn: () => api.playground({ messages: [...(sys.trim() ? [{ role: "system", content: sys }] : []), { role: "user", content: user }], run_ids: [null, chosen], max_tokens: 160 }) });
  return (
    <div className="mx-auto max-w-5xl">
      <PageHeader title="Playground" subtitle="The same messages go to the frozen base model and to the base model with a LoRA adapter loaded. Greedy decoding." />
      <Card className="space-y-4 p-5">
        <Field label="Adapter">
          <Select value={chosen} onChange={(e) => setRunId(e.target.value)} disabled={!done.length}>
            {done.length ? done.map((r) => <option key={r.id} value={r.id}>{r.name || r.id}</option>) : <option>No completed runs</option>}
          </Select>
        </Field>
        <Field label="System prompt" hint="Prefilled with the triage task prompt the adapter was trained with."><textarea value={sys} onChange={(e) => setSystem(e.target.value)} rows={5} className="rounded-lg border border-line bg-panel-2 p-3 font-mono text-xs" /></Field>
        <Field label="User message"><textarea value={user} onChange={(e) => setUser(e.target.value)} rows={4} maxLength={8000} className="rounded-lg border border-line bg-panel p-3 text-sm" /></Field>
        <div className="flex justify-end"><Button variant="primary" onClick={() => go.mutate()} disabled={!chosen || !user.trim() || go.isPending}>{go.isPending ? "Generating…" : "Run both"}</Button></div>
        {go.error && <ErrorNote error={go.error} />}
      </Card>
      {go.data && (
        <div className="mt-5 grid gap-4 md:grid-cols-2 [&>*]:min-w-0">
          {go.data.outputs.map((o) => {
            const parsed = parseMaybeJson(o.text);
            return (
              <Card key={o.run_id ?? "base"} className="p-4">
                <div className="mb-2 flex items-center gap-2 text-sm font-semibold">
                  <span className={o.run_id ? "text-tuned" : "text-base"}>{o.run_id ? "Base + LoRA" : "Base model"}</span>
                  <Chip tone={parsed ? "tuned" : "warn"}>{parsed ? "valid JSON" : "not JSON"}</Chip>
                </div>
                <pre className="whitespace-pre-wrap break-words rounded-md bg-panel-2 p-3 font-mono text-xs">{o.text}</pre>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
}
