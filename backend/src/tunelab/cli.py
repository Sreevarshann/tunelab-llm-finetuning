"""`tunelab serve` and `tunelab experiment` (headless, reproducible train + evaluate report)."""

from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import tunelab  # noqa: F401


def serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("tunelab.main:app", host=args.host, port=args.port)


def experiment(args: argparse.Namespace) -> None:
    import importlib.metadata as md

    from .config import get_settings
    from .evaluation import evaluate_rows, load_model
    from .tasks import ticket_triage as tt
    from .training import LoraConfig, run_lora

    s = get_settings()
    data = tt.generate(args.seed, args.train, args.valid, args.test, args.test_heldout)
    rows = {k: [{"messages": r["messages"]} for r in v] for k, v in data.items()}
    scoring = {"mode": "json_fields", "fields": tt.FIELDS}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"machine": platform.machine(), "platform": platform.platform(), "mlx": md.version("mlx"), "mlx_lm": md.version("mlx-lm")},
        "base_model": s.base_model,
        "base_revision": s.base_revision,
        "dataset": {"generator": "tasks/ticket_triage.py", "seed": args.seed, "sizes": {k: len(v) for k, v in rows.items()}},
        "evaluations": {},
    }
    base_model, base_tok = load_model(s.base_model, s.base_revision)
    for split in ("test", "test_heldout"):
        report["evaluations"][f"baseline/{split}"] = asdict(evaluate_rows(base_model, base_tok, rows[split], scoring)) | {"examples": None}
        print("baseline", split, json.dumps(report["evaluations"][f"baseline/{split}"]["metrics"]))
    del base_model
    for spec in args.configs:
        cfg = LoraConfig(base_model=s.base_model, base_revision=s.base_revision, **json.loads(spec))
        name = "lora:" + ",".join(f"{k}={v}" for k, v in json.loads(spec).items())
        res = run_lora(cfg, rows["train"], rows["valid"], out / f"adapter-{len(report['evaluations'])}", emit=lambda k, v: k == "stage" and print(k, v))
        report.setdefault("runs", {})[name] = asdict(res)
        m, t = load_model(s.base_model, s.base_revision, res.adapter_dir)
        for split in ("test", "test_heldout"):
            ev = evaluate_rows(m, t, rows[split], scoring)
            report["evaluations"][f"{name}/{split}"] = asdict(ev) | {"examples": [e for e in ev.examples if not e["all_correct"]][:15]}
            print(name, split, json.dumps(ev.metrics))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("wrote", out / "report.json")


def main() -> None:
    p = argparse.ArgumentParser(prog="tunelab")
    sub = p.add_subparsers(required=True)
    sp = sub.add_parser("serve")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8114)
    sp.set_defaults(fn=serve)
    ep = sub.add_parser("experiment", help="train LoRA configs and evaluate against the base model")
    ep.add_argument("--out", required=True)
    ep.add_argument("--seed", type=int, default=7)
    ep.add_argument("--train", type=int, default=600)
    ep.add_argument("--valid", type=int, default=80)
    ep.add_argument("--test", type=int, default=120)
    ep.add_argument("--test-heldout", type=int, default=120)
    ep.add_argument("--configs", nargs="+", default=['{"iters": 300}'], help="JSON LoraConfig overrides, one run each")
    ep.set_defaults(fn=experiment)
    a = p.parse_args()
    a.fn(a)
