"""Export completed runs and evaluations from the TuneLab database to eval/results.json and a
Markdown table. Numbers are copied from job results; nothing is recomputed or edited.

    uv run python scripts/export_results.py ../eval
"""

from __future__ import annotations

import importlib.metadata as md
import json
import platform
import sys
from pathlib import Path

from sqlalchemy import select

import tunelab  # noqa: F401
from tunelab.config import get_settings
from tunelab.db import Dataset, Job, session

out = Path(sys.argv[1] if len(sys.argv) > 1 else "../eval")
out.mkdir(parents=True, exist_ok=True)
s = get_settings()
with session() as db:
    runs = db.scalars(select(Job).where(Job.kind == "train").order_by(Job.created_at)).all()
    evals = db.scalars(select(Job).where(Job.kind == "evaluate", Job.status == "completed")).all()
    datasets = {d.id: d for d in db.scalars(select(Dataset))}

report = {
    "base_model": s.base_model,
    "base_revision": s.base_revision,
    "environment": {"machine": platform.machine(), "mac_ver": platform.mac_ver()[0], "mlx": md.version("mlx"), "mlx_lm": md.version("mlx-lm")},
    "datasets": {d.id: {"name": d.name, "source": d.source, "synthetic": d.synthetic, "splits": {k: {"count": v["count"], "sha256": v["sha256"]} for k, v in d.splits.items()}} for d in datasets.values()},
    "baselines": {},
    "runs": [],
}
for e in evals:
    if e.adapter_job_id is None:
        report["baselines"].setdefault(e.dataset_id, {})[e.split] = {"eval_id": e.id, "metrics": e.result["metrics"], "duration_s": e.result["duration_s"]}
for r in runs:
    tuned = {e.split: {"eval_id": e.id, "metrics": e.result["metrics"], "duration_s": e.result["duration_s"]} for e in evals if e.adapter_job_id == r.id}
    res = r.result or {}
    report["runs"].append({
        "id": r.id, "name": r.name, "status": r.status, "error": r.error, "dataset_id": r.dataset_id, "config": r.config,
        "training": {k: res.get(k) for k in ("trainable_params", "total_params", "lora_b_norm_before", "lora_b_norm_after", "initial_val_loss", "final_val_loss", "duration_s", "peak_memory_gb", "adapter_sha256")} if res else None,
        "evaluations": tuned,
    })
(out / "results.json").write_text(json.dumps(report, indent=2))

head = lambda m: m.get("all_fields_correct_pct") if m else None  # noqa: E731
lines = ["| Run | Status | Steps | Rank | Train time | Val loss (start → end) | Test: base → LoRA | Held-out: base → LoRA |", "|---|---|---|---|---|---|---|---|"]
for r in report["runs"]:
    b = report["baselines"].get(r["dataset_id"], {})
    t = r["training"] or {}
    cell = lambda sp: f"{head(b.get(sp, {}).get('metrics'))}% → **{head(r['evaluations'].get(sp, {}).get('metrics'))}%**" if r["evaluations"].get(sp) else "—"  # noqa: E731
    vl = f"{t['initial_val_loss']:.3f} → {t['final_val_loss']:.5f}" if t else "—"
    tt = f"{t['duration_s'] / 60:.1f} min" if t else "—"
    lines.append(f"| {r['name']} (`{r['id']}`) | {r['status']} | {r['config'].get('iters')} | {r['config'].get('rank')} | {tt} | {vl} | {cell('test')} | {cell('test_heldout')} |")
(out / "results.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
