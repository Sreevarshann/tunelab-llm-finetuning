from __future__ import annotations

import asyncio
import importlib.metadata as md
import json
from contextlib import asynccontextmanager
from datetime import UTC
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

import tunelab  # noqa: F401

from . import datasets as dsmod
from . import jobs
from .config import get_settings
from .db import Dataset, Job, JobEvent, session
from .tasks.ticket_triage import SYSTEM_PROMPT


@asynccontextmanager
async def lifespan(_: FastAPI):
    d = jobs.Dispatcher() if get_settings().dispatcher else None
    if d:
        d.start()
    yield
    if d:
        d.stop()


app = FastAPI(title="TuneLab API", version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=get_settings().cors_origins, allow_methods=["*"], allow_headers=["*"])


class LoraParams(BaseModel):
    rank: int = Field(8, ge=1, le=128)
    scale: float = Field(20.0, gt=0, le=100)
    dropout: float = Field(0.0, ge=0, lt=1)
    num_layers: int = Field(16, ge=1, le=128)
    learning_rate: float = Field(1e-4, ge=1e-6, le=1e-2)
    iters: int = Field(300, ge=1, le=20000)
    batch_size: int = Field(4, ge=1, le=64)
    max_seq_length: int = Field(512, ge=32, le=8192)
    steps_per_report: int = Field(10, ge=1, le=1000)
    steps_per_eval: int = Field(50, ge=1, le=5000)
    seed: int = Field(0, ge=0, le=2**31)
    mask_prompt: bool = True


class RunIn(BaseModel):
    dataset_id: str
    name: str = Field("", max_length=80)
    config: LoraParams = LoraParams()


class SyntheticIn(BaseModel):
    name: str = Field("Support ticket triage (synthetic)", min_length=1, max_length=80)
    seed: int = Field(7, ge=0, le=2**31)
    train: int = Field(600, ge=20, le=5000)
    valid: int = Field(80, ge=8, le=1000)
    test: int = Field(120, ge=8, le=2000)
    test_heldout: int = Field(120, ge=0, le=2000)


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=8000)


class PlaygroundIn(BaseModel):
    messages: list[Message] = Field(min_length=1, max_length=20)
    run_ids: list[str | None] = Field(default=[None], min_length=1, max_length=3)
    max_tokens: int = Field(160, ge=1, le=512)


def iso(dt):
    """SQLite drops tzinfo; every stored timestamp is UTC."""
    if dt is None:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).isoformat()


def _job_out(j: Job) -> dict:
    return {
        "id": j.id,
        "kind": j.kind,
        "name": j.name,
        "dataset_id": j.dataset_id,
        "status": j.status,
        "config": j.config,
        "result": j.result,
        "error": j.error,
        "split": j.split,
        "adapter_job_id": j.adapter_job_id,
        "progress": j.progress,
        "created_at": j.created_at,
        "started_at": j.started_at,
        "finished_at": j.finished_at,
    }


def _ds_out(d: Dataset) -> dict:
    return {"id": d.id, "name": d.name, "task": d.task, "source": d.source, "synthetic": d.synthetic, "scoring": d.scoring, "splits": d.splits, "created_at": iso(d.created_at)}


@app.get("/api/health")
def health():
    import mlx.core as mx

    s = get_settings()
    with session() as db:
        running = db.scalars(select(Job).where(Job.status == "running")).first()
    return {
        "status": "ok",
        "base_model": s.base_model,
        "base_revision": s.base_revision,
        "mlx": md.version("mlx"),
        "mlx_lm": md.version("mlx-lm"),
        "device": str(mx.default_device()),
        "running_job": running.id if running else None,
    }


# ---- datasets ----------------------------------------------------------------------------
@app.get("/api/datasets")
def list_datasets():
    with session() as db:
        return [_ds_out(d) for d in db.scalars(select(Dataset).order_by(Dataset.created_at.desc()))]


@app.post("/api/datasets/synthetic", status_code=201)
def create_synthetic(body: SyntheticIn):
    return _ds_out(dsmod.create_synthetic(body.name, body.seed, body.train, body.valid, body.test, body.test_heldout))


@app.post("/api/datasets/upload", status_code=201)
async def upload_dataset(
    name: str = Form(..., min_length=1, max_length=80),
    file: UploadFile | None = File(None),
    train: UploadFile | None = File(None),
    valid: UploadFile | None = File(None),
    test: UploadFile | None = File(None),
):
    limit = get_settings().max_upload_mb * 1024 * 1024
    files: dict[str, str] = {}
    for key, f in {"all": file, "train": train, "valid": valid, "test": test}.items():
        if f is None:
            continue
        raw = await f.read(limit + 1)
        if len(raw) > limit:
            raise HTTPException(413, f"{key}: file larger than {get_settings().max_upload_mb} MB")
        try:
            files[key] = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(422, f"{key}: file must be UTF-8 JSONL") from None
    if not files or ("all" in files and len(files) > 1) or ("all" not in files and not {"train", "valid"} <= files.keys()):
        raise HTTPException(422, "upload either one JSONL file, or train + valid (+ optional test) files")
    try:
        return _ds_out(dsmod.create_from_upload(name, files))
    except dsmod.DatasetError as e:
        raise HTTPException(422, {"message": "invalid dataset", "errors": e.errors[:20]}) from None


def _dataset(db, ds_id: str) -> Dataset:
    d = db.get(Dataset, ds_id)
    if d is None:
        raise HTTPException(404, "dataset not found")
    return d


@app.get("/api/datasets/{ds_id}")
def get_dataset(ds_id: str):
    with session() as db:
        return _ds_out(_dataset(db, ds_id))


@app.get("/api/datasets/{ds_id}/rows")
def dataset_rows(ds_id: str, split: str = Query("train", pattern=r"^[a-z_]{1,20}$"), offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=200)):
    with session() as db:
        d = _dataset(db, ds_id)
    if split not in d.splits:
        raise HTTPException(404, "split not found")
    rows = dsmod.read_split(ds_id, split)
    return {"total": len(rows), "rows": rows[offset : offset + limit]}


# ---- runs & evaluations ------------------------------------------------------------------
@app.post("/api/runs", status_code=201)
def create_run(body: RunIn):
    with session() as db:
        _dataset(db, body.dataset_id)
    return _job_out(jobs.enqueue_train(body.dataset_id, body.config.model_dump(), body.name))


@app.get("/api/runs")
def list_runs():
    with session() as db:
        return [_job_out(j) for j in db.scalars(select(Job).where(Job.kind == "train").order_by(Job.created_at.desc()))]


def _job(db, job_id: str, kind: str | None = None) -> Job:
    j = db.get(Job, job_id)
    if j is None or (kind and j.kind != kind):
        raise HTTPException(404, "not found")
    return j


def _evals_for(db, run: Job) -> list[dict]:
    tuned = db.scalars(select(Job).where(Job.kind == "evaluate", Job.adapter_job_id == run.id).order_by(Job.created_at)).all()
    base = db.scalars(select(Job).where(Job.kind == "evaluate", Job.dataset_id == run.dataset_id, Job.adapter_job_id.is_(None)).order_by(Job.created_at.desc())).all()
    latest_base: dict[str, Job] = {}
    for b in base:  # newest completed baseline per split, else newest
        cur = latest_base.get(b.split)
        if cur is None or (cur.status != "completed" and b.status == "completed"):
            latest_base[b.split] = b
    return [_job_out(e) for e in [*tuned, *latest_base.values()]]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    with session() as db:
        run = _job(db, run_id, "train")
        return {**_job_out(run), "dataset": _ds_out(db.get(Dataset, run.dataset_id)), "evaluations": _evals_for(db, run)}


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str):
    with session() as db:
        _job(db, run_id)
    return _job_out(jobs.request_cancel(run_id))


@app.post("/api/runs/{run_id}/evaluate", status_code=202)
def evaluate_run(run_id: str):
    with session() as db:
        if _job(db, run_id, "train").status != "completed":
            raise HTTPException(409, "run is not completed")
    return {"queued": jobs.ensure_evaluations(run_id)}


@app.get("/api/runs/{run_id}/adapter")
def download_adapter(run_id: str):
    with session() as db:
        run = _job(db, run_id, "train")
    if run.status != "completed":
        raise HTTPException(409, "run is not completed")
    return FileResponse(Path(run.result["adapter_dir"]) / "adapters.safetensors", filename=f"{run_id}-adapters.safetensors", media_type="application/octet-stream")


@app.get("/api/evaluations/{eval_id}/examples")
def eval_examples(eval_id: str, only_errors: bool = False, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500)):
    with session() as db:
        e = _job(db, eval_id, "evaluate")
    if e.status != "completed":
        raise HTTPException(409, "evaluation is not completed")
    rows = json.loads(Path(e.result["examples_file"]).read_text())
    for i, r in enumerate(rows):
        r["index"] = i
    if only_errors:
        rows = [r for r in rows if not r.get("all_correct", r.get("exact", False))]
    return {"total": len(rows), "rows": rows[offset : offset + limit]}


@app.get("/api/jobs")
def list_jobs(status: str | None = None):
    with session() as db:
        q = select(Job).order_by(Job.created_at.desc()).limit(200)
        if status:
            q = q.where(Job.status == status)
        return [_job_out(j) for j in db.scalars(q)]


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: str, after: int = 0):
    with session() as db:
        _job(db, job_id)

    async def stream():
        last = after
        idle = 0
        while True:
            with session() as db:
                evs = db.scalars(select(JobEvent).where(JobEvent.job_id == job_id, JobEvent.id > last).order_by(JobEvent.id)).all()
                status = db.get(Job, job_id).status
            for e in evs:
                last = e.id
                yield {"id": str(e.id), "event": e.type, "data": json.dumps({**e.data, "at": iso(e.created_at)})}
            if status in jobs.TERMINAL and not evs:
                yield {"event": "end", "data": json.dumps({"status": status})}
                return
            idle = 0 if evs else idle + 1
            if idle and idle % 30 == 0:
                yield {"comment": "keep-alive"}
            await asyncio.sleep(0.5)

    return EventSourceResponse(stream(), headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


@app.get("/api/compare")
def compare(run_ids: str = Query(..., pattern=r"^[A-Za-z0-9_,]+$", max_length=400)):
    ids = [i for i in run_ids.split(",") if i][:6]
    with session() as db:
        out = []
        for rid in ids:
            run = _job(db, rid, "train")
            out.append({"run": _job_out(run), "evaluations": _evals_for(db, run)})
        return out


@app.post("/api/playground")
def playground(body: PlaygroundIn):
    from .inference import complete

    adapters: list[tuple[str | None, str | None]] = []
    with session() as db:
        for rid in body.run_ids:
            if rid is None:
                adapters.append((None, None))
                continue
            run = _job(db, rid, "train")
            if run.status != "completed":
                raise HTTPException(409, f"{rid} is not completed")
            adapters.append((rid, run.result["adapter_dir"]))
    msgs = [m.model_dump() for m in body.messages]
    return {"outputs": [{"run_id": rid, "text": complete(msgs, path, body.max_tokens)} for rid, path in adapters]}


@app.get("/api/tasks/ticket_triage/system_prompt")
def triage_prompt():
    return {"system_prompt": SYSTEM_PROMPT}
