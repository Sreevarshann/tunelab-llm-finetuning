"use client";

import { forwardRef } from "react";
import { AlertTriangle } from "lucide-react";
import { cn } from "@/lib/cn";

type BtnProps = React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "outline" | "ghost" | "danger"; size?: "sm" | "md" };
export const Button = forwardRef<HTMLButtonElement, BtnProps>(function Button({ className, variant = "outline", size = "md", type = "button", ...p }, ref) {
  return (
    <button
      ref={ref}
      type={type}
      className={cn(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        size === "sm" ? "h-8 px-3 text-xs" : "h-10 px-4 text-sm",
        variant === "primary" && "bg-ink text-bg hover:opacity-90",
        variant === "outline" && "border border-line bg-panel hover:bg-panel-2",
        variant === "ghost" && "text-muted hover:bg-panel-2 hover:text-ink",
        variant === "danger" && "border border-bad/40 text-bad hover:bg-bad-soft",
        className,
      )}
      {...p}
    />
  );
});

export const Card = ({ className, ...p }: React.HTMLAttributes<HTMLDivElement>) => <div className={cn("rounded-xl border border-line bg-panel", className)} {...p} />;

export function Field({ label, hint, error, children }: { label: string; hint?: React.ReactNode; error?: string; children: React.ReactNode }) {
  return (
    <label className="flex flex-col gap-1.5 text-sm">
      <span className="font-medium">{label}</span>
      {children}
      {hint && !error && <span className="text-xs text-muted">{hint}</span>}
      {error && <span className="text-xs text-bad">{error}</span>}
    </label>
  );
}

export const Input = forwardRef<HTMLInputElement, React.InputHTMLAttributes<HTMLInputElement>>(function Input({ className, ...p }, ref) {
  return <input ref={ref} className={cn("h-10 rounded-lg border border-line bg-panel px-3 text-sm placeholder:text-muted/70 focus:border-tuned focus:outline-none", className)} {...p} />;
});

export const Select = forwardRef<HTMLSelectElement, React.SelectHTMLAttributes<HTMLSelectElement>>(function Select({ className, ...p }, ref) {
  return <select ref={ref} className={cn("h-10 rounded-lg border border-line bg-panel px-3 text-sm focus:border-tuned focus:outline-none", className)} {...p} />;
});

type Tone = "neutral" | "base" | "tuned" | "warn" | "bad";
export function Chip({ tone = "neutral", className, ...p }: React.HTMLAttributes<HTMLSpanElement> & { tone?: Tone }) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-md px-1.5 py-0.5 text-[11px] font-medium",
        { neutral: "bg-panel-2 text-muted", base: "bg-base-soft text-base", tuned: "bg-tuned-soft text-tuned", warn: "bg-warn-soft text-warn", bad: "bg-bad-soft text-bad" }[tone],
        className,
      )}
      {...p}
    />
  );
}


export const Skeleton = ({ className }: { className?: string }) => <div aria-hidden className={cn("skeleton h-4", className)} />;

export function Empty({ icon, title, children, action }: { icon: React.ReactNode; title: string; children?: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed border-line px-6 py-14 text-center">
      <div className="grid size-11 place-items-center rounded-full bg-tuned-soft text-tuned">{icon}</div>
      <h3 className="font-semibold text-lg">{title}</h3>
      {children && <div className="max-w-md text-sm text-muted">{children}</div>}
      {action}
    </div>
  );
}

export function ErrorNote({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-lg border border-bad/30 bg-bad-soft px-4 py-3 text-sm text-bad">
      <AlertTriangle className="mt-0.5 size-4 shrink-0" />
      <div className="flex-1">{error instanceof Error ? error.message : String(error)}</div>
      {onRetry && (
        <button className="underline" onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

export function Meter({ used, limit, label }: { used: number; limit: number; label: string }) {
  const pct = limit > 0 ? Math.min(100, (used / limit) * 100) : 0;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between text-sm">
        <span className="text-muted">{label}</span>
        <span className="font-mono text-xs tabular-nums">
          {used.toLocaleString()} / {limit.toLocaleString()}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-panel-2" role="progressbar" aria-label={label} aria-valuenow={used} aria-valuemin={0} aria-valuemax={limit}>
        <div className={cn("h-full rounded-full", pct >= 90 ? "bg-bad" : pct >= 70 ? "bg-warn" : "bg-tuned")} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function PageHeader({ title, subtitle, actions }: { title: string; subtitle?: React.ReactNode; actions?: React.ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
      <div>
        <h1 className="font-semibold text-2xl font-semibold">{title}</h1>
        {subtitle && <div className="mt-1 text-sm text-muted">{subtitle}</div>}
      </div>
      {actions && <div className="flex items-center gap-2">{actions}</div>}
    </div>
  );
}

export const fmtPct = (v: number | null | undefined, digits = 1) => (v == null ? "—" : `${v.toFixed(digits)}%`);
export const fmtNum = (v: number | null | undefined) => (v == null ? "—" : v.toLocaleString());
export const fmtSec = (s: number | null | undefined) => (s == null ? "—" : s < 90 ? `${s.toFixed(0)} s` : `${(s / 60).toFixed(1)} min`);
export const fmtMs = (ms: number | null | undefined) => (ms == null ? "—" : ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1)} s`);
export const fmtDate = (s: string | null | undefined) => (s ? new Date(s).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" }) : "—");
export const relTime = (s: string) => {
  const d = (Date.now() - new Date(s).getTime()) / 1000;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return `${Math.floor(d / 86400)}d ago`;
};
