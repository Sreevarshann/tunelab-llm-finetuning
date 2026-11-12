"use client";

import { useQuery } from "@tanstack/react-query";
import { Activity, Database, FlaskConical, GitCompareArrows, LayoutGrid, MessagesSquare, Moon, Sun } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Suspense } from "react";
import { api } from "@/lib/api";
import { cn } from "@/lib/cn";
import { Logo } from "./Logo";
import { useTheme } from "./theme";

const NAV = [
  { href: "/", label: "Runs", icon: LayoutGrid, match: (p: string) => p === "/" || (p.startsWith("/runs/") && p !== "/runs/new") },
  { href: "/runs/new", label: "New run", icon: FlaskConical, match: (p: string) => p === "/runs/new" },
  { href: "/datasets", label: "Datasets", icon: Database, match: (p: string) => p.startsWith("/datasets") },
  { href: "/compare", label: "Compare", icon: GitCompareArrows, match: (p: string) => p.startsWith("/compare") },
  { href: "/playground", label: "Playground", icon: MessagesSquare, match: (p: string) => p.startsWith("/playground") },
];

function Nav({ mobile = false }: { mobile?: boolean }) {
  const path = usePathname();
  return (
    <nav aria-label="Main" className={mobile ? "flex gap-1 overflow-x-auto px-2 pb-2" : "mt-4 flex flex-col gap-0.5 px-3"}>
      {NAV.map((n) => {
        const active = n.match(path);
        return (
          <Link key={n.href} href={n.href} aria-current={active ? "page" : undefined} className={cn("flex shrink-0 items-center gap-2.5 rounded-lg px-3 py-2 text-sm", mobile && "px-2.5 py-1.5 text-xs", active ? "bg-panel-2 font-medium text-ink" : "text-muted hover:bg-panel-2 hover:text-ink")}>
            <n.icon className="size-4" />
            {n.label}
          </Link>
        );
      })}
    </nav>
  );
}

function Engine() {
  const h = useQuery({ queryKey: ["health"], queryFn: api.health, refetchInterval: 5000 });
  const { theme, toggle } = useTheme();
  return (
    <div className="space-y-2 border-t border-line p-4 text-[11px] text-muted">
      <div className="flex items-center justify-between">
        <span className="flex items-center gap-1.5">
          <Activity className={cn("size-3.5", h.data?.running_job ? "text-tuned" : "")} />
          {h.isError ? <span className="text-bad">API offline</span> : h.data?.running_job ? "Job running" : "Idle"}
        </span>
        <button onClick={toggle} className="rounded-md p-1.5 hover:bg-panel-2 hover:text-ink" aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
          {theme === "dark" ? <Sun className="size-3.5" /> : <Moon className="size-3.5" />}
        </button>
      </div>
      {h.data && (
        <div className="space-y-0.5 font-mono">
          <div className="truncate" title={h.data.base_model}>{h.data.base_model.replace("mlx-community/", "")}</div>
          <div>rev {h.data.base_revision?.slice(0, 7) ?? "latest"} · mlx {h.data.mlx} · {h.data.device.includes("gpu") ? "Metal GPU" : h.data.device}</div>
        </div>
      )}
    </div>
  );
}

export function Shell({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-dvh">
      <aside className="sticky top-0 hidden h-dvh w-56 shrink-0 flex-col border-r border-line bg-panel md:flex">
        <Link href="/" className="flex items-center gap-2.5 px-5 py-5">
          <Logo />
          <span className="text-base font-semibold tracking-tight">TuneLab</span>
        </Link>
        <Suspense>
          <Nav />
        </Suspense>
        <div className="mt-auto">
          <Engine />
        </div>
      </aside>
      <div className="min-w-0 flex-1">
        <header className="sticky top-0 z-10 border-b border-line bg-panel md:hidden">
          <div className="flex items-center gap-2 px-4 pt-3">
            <Logo className="size-6" />
            <span className="text-sm font-semibold">TuneLab</span>
          </div>
          <Suspense>
            <Nav mobile />
          </Suspense>
        </header>
        <main className="px-4 py-6 sm:px-8 lg:px-10">{children}</main>
      </div>
    </div>
  );
}
