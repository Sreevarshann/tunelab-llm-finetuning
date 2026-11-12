"""Scoring of structured predictions against reference JSON. Pure functions, no model needed."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_object(text: str) -> dict | None:
    """Strict-ish: accepts the object alone or inside a code fence; nothing else is repaired."""
    t = FENCE.sub("", text.strip()).strip()
    try:
        v = json.loads(t)
    except (json.JSONDecodeError, ValueError):
        return None
    return v if isinstance(v, dict) else None


@dataclass
class Score:
    parsed: bool
    exact_keys: bool
    fields: dict[str, bool] = field(default_factory=dict)

    @property
    def all_correct(self) -> bool:
        return self.parsed and self.exact_keys and all(self.fields.values())


def score(prediction: str, reference: str, fields: list[str]) -> Score:
    ref = json.loads(reference)
    pred = parse_json_object(prediction)
    if pred is None:
        return Score(False, False, {f: False for f in fields})
    return Score(True, set(pred) == set(fields), {f: pred.get(f, "<missing>") == ref[f] for f in fields})


def aggregate(scores: list[Score], fields: list[str]) -> dict:
    n = len(scores)
    if n == 0:
        return {"n": 0}

    def pct(k: int) -> float:
        return round(100 * k / n, 1)

    return {
        "n": n,
        "json_valid_pct": pct(sum(s.parsed for s in scores)),
        "schema_exact_pct": pct(sum(s.parsed and s.exact_keys for s in scores)),
        "all_fields_correct_pct": pct(sum(s.all_correct for s in scores)),
        "field_accuracy_pct": {f: pct(sum(s.fields[f] for s in scores)) for f in fields},
    }


def _norm_tokens(s: str) -> list[str]:
    return re.findall(r"\w+", s.lower())


def token_f1(prediction: str, reference: str) -> float:
    p, r = _norm_tokens(prediction), _norm_tokens(reference)
    if not p or not r:
        return float(p == r)
    common: dict[str, int] = {}
    for t in p:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in r:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(r)
    return 2 * prec * rec / (prec + rec)


def score_text(prediction: str, reference: str) -> dict:
    return {"exact": " ".join(_norm_tokens(prediction)) == " ".join(_norm_tokens(reference)), "f1": round(token_f1(prediction, reference), 4)}


def aggregate_text(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    return {"n": n, "exact_match_pct": round(100 * sum(r["exact"] for r in rows) / n, 1), "token_f1_pct": round(100 * sum(r["f1"] for r in rows) / n, 1)}
