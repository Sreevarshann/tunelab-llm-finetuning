"""Job queue: one job at a time, each in its own worker subprocess (isolates GPU memory and
lets a cancel or crash never take down the API). Events go to SQLite and are streamed by the API.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time

from sqlalchemy import select

from .config import get_settings
from .db import Dataset, Job, JobEvent, new_id, now, session

TERMINAL = {"completed", "failed", "cancelled"}
EVAL_SPLITS = ("test", "test_heldout")


def emit(job_id: str, type_: str, data: dict) -> None:
    with session() as s:
        s.add(JobEvent(job_id=job_id, type=type_, data=data))
        s.commit()


def enqueue_train(dataset_id: str, config: dict, name: str = "") -> Job:
    job = Job(id=new_id("run"), kind="train", name=name or "LoRA run", dataset_id=dataset_id, config=config)
    with session() as s:
        s.add(job)
        s.commit()
    emit(job.id, "status", {"status": "queued"})
    return job


def enqueue_eval(dataset_id: str, split: str, adapter_job_id: str | None) -> Job:
    job = Job(
        id=new_id("eval"),
        kind="evaluate",
        name=f"{'tuned' if adapter_job_id else 'baseline'} · {split}",
        dataset_id=dataset_id,
        split=split,
        adapter_job_id=adapter_job_id,
        config={"base_model": get_settings().base_model, "base_revision": get_settings().base_revision},
    )
    with session() as s:
        s.add(job)
        s.commit()
    emit(job.id, "status", {"status": "queued"})
    return job


def ensure_evaluations(train_job_id: str) -> list[str]:
    """After a run completes: evaluate it on every test split, plus the base model once per split."""
    created = []
    s = get_settings()
    with session() as db:
        run = db.get(Job, train_job_id)
        ds = db.get(Dataset, run.dataset_id)
        splits = [sp for sp in EVAL_SPLITS if sp in ds.splits]
        for sp in splits:
            created.append(enqueue_eval(ds.id, sp, train_job_id).id)
            has_base = db.scalars(
                select(Job).where(Job.kind == "evaluate", Job.dataset_id == ds.id, Job.split == sp, Job.adapter_job_id.is_(None), Job.status.in_(["queued", "running", "completed"]))
            ).first()
            if has_base is None or has_base.config.get("base_revision") != s.base_revision:
                created.append(enqueue_eval(ds.id, sp, None).id)
    return created


def request_cancel(job_id: str) -> Job:
    with session() as s:
        job = s.get(Job, job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status == "queued":
            job.status, job.finished_at = "cancelled", now()
        elif job.status == "running":
            job.cancel_requested = True
        s.commit()
    if job.status == "cancelled":
        emit(job_id, "status", {"status": "cancelled"})
    return job


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def recover_stale() -> None:
    """Jobs marked running whose worker is gone (API restart, crash) are failed, not left hanging."""
    with session() as s:
        for job in s.scalars(select(Job).where(Job.status == "running")):
            if not _alive(job.pid):
                job.status, job.error, job.finished_at = "failed", "worker process exited unexpectedly", now()
        s.commit()


def start_next() -> str | None:
    with session() as s:
        if s.scalars(select(Job).where(Job.status == "running")).first() is not None:
            return None
        # evaluations first: they are short and they are what a finished run is waiting on
        job = s.scalars(select(Job).where(Job.status == "queued").order_by((Job.kind != "evaluate"), Job.created_at)).first()
        if job is None:
            return None
        job.status, job.started_at = "running", now()
        s.commit()
        job_id = job.id
    env = {**os.environ, "TUNELAB_DATA_DIR": str(get_settings().data_dir.resolve())}
    log = get_settings().data_dir / "jobs" / job_id
    log.mkdir(parents=True, exist_ok=True)
    with (log / "worker.log").open("ab") as logf:  # the child keeps its own copy of the fd
        proc = subprocess.Popen([sys.executable, "-m", "tunelab.worker", job_id], stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    with session() as s:
        s.get(Job, job_id).pid = proc.pid
        s.commit()
    emit(job_id, "status", {"status": "running"})
    threading.Thread(target=_reap, args=(job_id, proc), daemon=True).start()
    return job_id


def _reap(job_id: str, proc: subprocess.Popen) -> None:
    deadline = None
    while proc.poll() is None:
        time.sleep(1)
        with session() as s:
            job = s.get(Job, job_id)
            if job.cancel_requested and deadline is None:
                deadline = time.time() + 30  # grace for the worker to stop at a step boundary
        if deadline and time.time() > deadline:
            os.killpg(proc.pid, signal.SIGKILL)
    with session() as s:
        job = s.get(Job, job_id)
        if job.status == "running":  # worker died without recording an outcome
            job.status = "cancelled" if job.cancel_requested else "failed"
            job.error = job.error or ("cancelled" if job.cancel_requested else f"worker exited with code {proc.returncode}")
            job.finished_at = now()
            s.commit()
            emit(job_id, "status", {"status": job.status})


class Dispatcher(threading.Thread):
    def __init__(self, interval: float = 1.0):
        super().__init__(daemon=True, name="tunelab-dispatcher")
        self.interval, self._stop = interval, threading.Event()

    def run(self) -> None:
        recover_stale()
        while not self._stop.is_set():
            try:
                start_next()
            except Exception as e:  # noqa: BLE001 - keep dispatching; the error is logged
                print(f"dispatcher error: {e}", file=sys.stderr)
            self._stop.wait(self.interval)

    def stop(self) -> None:
        self._stop.set()
