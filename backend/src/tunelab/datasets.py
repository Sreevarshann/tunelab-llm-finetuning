"""Dataset creation, validation, splitting and statistics. Data lives in JSONL files."""

from __future__ import annotations

import hashlib
import json
import statistics
from functools import lru_cache
from pathlib import Path

from .config import get_settings
from .db import Dataset, new_id, session
from .scoring import parse_json_object
from .tasks import ticket_triage

ROLES = {"system", "user", "assistant"}
MAX_ROWS = 20_000
MAX_CHARS = 8_000
TRAIN_SPLITS = ("train", "valid")


class DatasetError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors[:5]))
        self.errors = errors


def dataset_dir(dataset_id: str) -> Path:
    return get_settings().data_dir / "datasets" / dataset_id


def read_split(dataset_id: str, split: str) -> list[dict]:
    p = dataset_dir(dataset_id) / f"{split}.jsonl"
    if not p.exists():
        raise FileNotFoundError(split)
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def validate_rows(rows: list, where: str = "") -> list[dict]:
    """Chat format: {"messages": [{"role", "content"}...]} ending with a non-empty assistant turn."""
    errors: list[str] = []
    clean: list[dict] = []
    if len(rows) > MAX_ROWS:
        raise DatasetError([f"{where}too many rows ({len(rows)} > {MAX_ROWS})"])
    for i, r in enumerate(rows, 1):
        loc = f"{where}line {i}: "
        msgs = r.get("messages") if isinstance(r, dict) else None
        if not isinstance(msgs, list) or len(msgs) < 2:
            errors.append(loc + 'expected {"messages": [...]} with at least a user and an assistant turn')
            continue
        bad = next((m for m in msgs if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(m.get("content"), str)), None)
        if bad is not None:
            errors.append(loc + "each message needs role system|user|assistant and string content")
            continue
        if msgs[-1]["role"] != "assistant" or not msgs[-1]["content"].strip():
            errors.append(loc + "last message must be a non-empty assistant reply")
            continue
        if not any(m["role"] == "user" for m in msgs):
            errors.append(loc + "needs at least one user message")
            continue
        if sum(len(m["content"]) for m in msgs) > MAX_CHARS:
            errors.append(loc + f"conversation longer than {MAX_CHARS} characters")
            continue
        clean.append({"messages": [{"role": m["role"], "content": m["content"]} for m in msgs]})
        if len(errors) >= 50:
            break
    if errors:
        raise DatasetError(errors)
    return clean


def parse_jsonl(text: str, where: str = "") -> list:
    rows, errors = [], []
    for i, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            errors.append(f"{where}line {i}: invalid JSON ({e.msg})")
    if errors:
        raise DatasetError(errors)
    return rows


def split_rows(rows: list[dict], valid_pct: float = 0.1, test_pct: float = 0.1) -> dict[str, list[dict]]:
    """Deterministic split by content hash, so re-uploading the same data gives the same splits."""

    def bucket(r: dict) -> float:
        h = hashlib.sha256(json.dumps(r, sort_keys=True).encode()).digest()
        return int.from_bytes(h[:8], "big") / 2**64

    out: dict[str, list[dict]] = {"train": [], "valid": [], "test": []}
    for r in rows:
        b = bucket(r)
        out["test" if b < test_pct else "valid" if b < test_pct + valid_pct else "train"].append(r)
    return out


def detect_scoring(rows: list[dict]) -> dict:
    """JSON-field accuracy when every assistant reply is a JSON object with the same keys."""
    replies = [r["messages"][-1]["content"] for r in rows]
    objs = [parse_json_object(t) for t in replies]
    if objs and all(o is not None for o in objs):
        keys = set(objs[0])
        if keys and all(set(o) == keys for o in objs):
            return {"mode": "json_fields", "fields": sorted(keys)}
    return {"mode": "text"}


@lru_cache
def _tokenizer():
    from transformers import AutoTokenizer

    s = get_settings()
    return AutoTokenizer.from_pretrained(s.base_model, revision=s.base_revision)


def token_stats(rows: list[dict]) -> dict:
    tok = _tokenizer()
    lens = [len(tok.apply_chat_template(r["messages"], tokenize=True, return_dict=False)) for r in rows]
    if not lens:
        return {}
    lens.sort()
    return {"mean": round(statistics.fmean(lens), 1), "p95": lens[int(0.95 * (len(lens) - 1))], "max": lens[-1], "total": sum(lens)}


def _store(name: str, task: str, source: str, synthetic: bool, splits: dict[str, list[dict]], scoring: dict) -> Dataset:
    if not splits.get("train") or not splits.get("valid"):
        raise DatasetError(["need non-empty train and valid splits (add more rows)"])
    ds_id = new_id("ds")
    d = dataset_dir(ds_id)
    d.mkdir(parents=True, exist_ok=True)
    meta = {}
    for split, rows in splits.items():
        if not rows:
            continue
        body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (d / f"{split}.jsonl").write_text(body)
        meta[split] = {"count": len(rows), "sha256": hashlib.sha256(body.encode()).hexdigest(), "tokens": token_stats(rows)}
    row = Dataset(id=ds_id, name=name, task=task, source=source, synthetic=synthetic, scoring=scoring, splits=meta)
    with session() as s:
        s.add(row)
        s.commit()
    return row


def create_synthetic(name: str = "Support ticket triage (synthetic)", seed: int = 7, train: int = 600, valid: int = 80, test: int = 120, test_heldout: int = 120) -> Dataset:
    raw = ticket_triage.generate(seed, train, valid, test, test_heldout)
    splits = {k: [{"messages": r["messages"]} for r in v] for k, v in raw.items()}
    src = f"synthetic generator tasks/ticket_triage.py, seed={seed}; test_heldout uses only template phrasings unseen in training"
    return _store(name, "ticket_triage", src, True, splits, {"mode": "json_fields", "fields": ticket_triage.FIELDS})


def create_from_upload(name: str, files: dict[str, str]) -> Dataset:
    """files: {"all": text} (auto split 80/10/10) or explicit {"train", "valid", "test"?}."""
    if "all" in files:
        rows = validate_rows(parse_jsonl(files["all"]))
        splits = split_rows(rows)
    else:
        splits = {k: validate_rows(parse_jsonl(v, f"{k}: "), f"{k}: ") for k, v in files.items() if v.strip()}
    everything = [r for v in splits.values() for r in v]
    return _store(name, "chat", "uploaded JSONL", False, splits, detect_scoring(everything))
