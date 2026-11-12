import { CheckCircle2, CircleDashed, Loader2, Ban, XCircle } from "lucide-react";
import type { JobStatus } from "@/lib/types";
import { Chip } from "./ui";

export function StatusBadge({ status, progress }: { status: JobStatus; progress?: number }) {
  const m = {
    queued: { icon: CircleDashed, tone: "neutral", label: "Queued" },
    running: { icon: Loader2, tone: "tuned", label: progress != null ? `Running ${Math.round(progress * 100)}%` : "Running" },
    completed: { icon: CheckCircle2, tone: "tuned", label: "Completed" },
    failed: { icon: XCircle, tone: "bad", label: "Failed" },
    cancelled: { icon: Ban, tone: "neutral", label: "Cancelled" },
  }[status] as { icon: typeof CircleDashed; tone: "neutral" | "tuned" | "bad"; label: string };
  return (
    <Chip tone={m.tone}>
      <m.icon className={`size-3 ${status === "running" ? "animate-spin motion-reduce:animate-none" : ""}`} />
      {m.label}
    </Chip>
  );
}
