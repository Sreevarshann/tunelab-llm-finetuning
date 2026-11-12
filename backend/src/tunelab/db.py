"""SQLite persistence (WAL) shared by the API and job worker processes."""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from .config import get_settings


def now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class Base(DeclarativeBase):
    pass


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    task: Mapped[str] = mapped_column(String)  # "ticket_triage" | "chat"
    source: Mapped[str] = mapped_column(String)  # human-readable provenance
    synthetic: Mapped[bool] = mapped_column(Boolean, default=False)
    scoring: Mapped[dict] = mapped_column(JSON)  # {"mode": "json_fields", "fields": [...]} | {"mode": "text"}
    splits: Mapped[dict] = mapped_column(JSON)  # split -> {"count", "sha256", "tokens": {...}}
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Job(Base):
    """A training run or an evaluation; both run in a worker subprocess, one at a time."""

    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    kind: Mapped[str] = mapped_column(String)  # "train" | "evaluate"
    name: Mapped[str] = mapped_column(String, default="")
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.id"))
    status: Mapped[str] = mapped_column(String, default="queued")  # queued|running|completed|failed|cancelled
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # evaluation jobs: which adapter (training job id) or None for the base model
    adapter_job_id: Mapped[str | None] = mapped_column(String, nullable=True)
    split: Mapped[str | None] = mapped_column(String, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class JobEvent(Base):
    __tablename__ = "job_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(ForeignKey("jobs.id"), index=True)
    type: Mapped[str] = mapped_column(String)
    data: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


@lru_cache
def engine():
    s = get_settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    eng = create_engine(s.db_url, connect_args={"check_same_thread": False, "timeout": 30})

    @event.listens_for(eng, "connect")
    def _pragmas(conn, _):
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    Base.metadata.create_all(eng)
    return eng


def session():
    return sessionmaker(engine(), expire_on_commit=False)()
