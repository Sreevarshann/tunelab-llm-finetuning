"""Worker subprocess: `python -m tunelab.worker <job_id>`."""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import asdict

import tunelab  # noqa: F401  (sets MLX env before mlx import)

from .config import get_settings
from .datasets import read_split
from .db import Dataset, Job, now, session
from .jobs import emit, ensure_evaluations


class Cancelled(Exception):
    pass


def _cancel_check(job_id: str):
    last = [0.0, False]

    def check() -> bool:
        if time.time() - last[0] > 2:
            with session() as s:
                last[1] = bool(s.get(Job, job_id).cancel_requested)
            last[0] = time.time()
        return last[1]

    return check


def run_train(job: Job, ds: Dataset) -> dict:
    from .training import LoraConfig, run_lora

    s = get_settings()
    cfg = LoraConfig(base_model=s.base_model, base_revision=s.base_revision, **job.config)
    total = cfg.iters

    def on(kind: str, data: dict) -> None:
        emit(job.id, kind, data)
        if kind == "train":
            with session() as db:
                db.get(Job, job.id).progress = round(data["iteration"] / total, 3)
                db.commit()

    out = s.data_dir / "jobs" / job.id / "adapter"
    result = run_lora(cfg, read_split(ds.id, "train"), read_split(ds.id, "valid"), out, emit=on, should_stop=_cancel_check(job.id))
    return asdict(result)


def run_eval(job: Job, ds: Dataset) -> dict:
    from .evaluation import evaluate_rows, load_model

    adapter = None
    if job.adapter_job_id:
        with session() as s:
            run = s.get(Job, job.adapter_job_id)
            if run is None or run.status != "completed":
                raise RuntimeError("adapter run is not completed")
            adapter = run.result["adapter_dir"]
    emit(job.id, "stage", {"stage": "loading_model", "adapter": bool(adapter)})
    model, tok = load_model(job.config["base_model"], job.config.get("base_revision"), adapter)
    rows = read_split(ds.id, job.split)

    def progress(done: int, total: int) -> None:
        emit(job.id, "progress", {"done": done, "total": total})
        with session() as s:
            s.get(Job, job.id).progress = round(done / total, 3)
            s.commit()

    out = evaluate_rows(model, tok, rows, ds.scoring, on_progress=progress, should_stop=_cancel_check(job.id))
    path = get_settings().data_dir / "jobs" / job.id / "examples.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out.examples, ensure_ascii=False))
    return {"metrics": out.metrics, "duration_s": out.duration_s, "tokens_per_s": out.tokens_per_s, "examples_file": str(path)}


def main(job_id: str) -> int:
    with session() as s:
        job = s.get(Job, job_id)
        ds = s.get(Dataset, job.dataset_id)
    try:
        result = run_train(job, ds) if job.kind == "train" else run_eval(job, ds)
        status, error = "completed", None
    except KeyboardInterrupt:
        result, status, error = None, "cancelled", "cancelled by user"
    except Exception as e:  # noqa: BLE001 - any failure is recorded on the job
        traceback.print_exc()
        result, status, error = None, "failed", f"{type(e).__name__}: {e}"[:2000]
    with session() as s:
        j = s.get(Job, job_id)
        j.status, j.result, j.error, j.finished_at = status, result, error, now()
        if status == "completed":
            j.progress = 1.0
        s.commit()
    emit(job_id, "status", {"status": status, **({"error": error} if error else {})})
    if status == "completed" and job.kind == "train":
        ensure_evaluations(job_id)
    return 0 if status == "completed" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
