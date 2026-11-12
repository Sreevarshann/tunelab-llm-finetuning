"""Evaluate a base model with or without a LoRA adapter on held-out examples.

Decoding is greedy (deterministic) and identical for baseline and tuned runs; both receive the
same system prompt and user text. The only difference is whether adapter weights are loaded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from mlx_lm import batch_generate, load

from .scoring import aggregate, aggregate_text, score, score_text

MAX_TOKENS = 96


@dataclass
class EvalOutput:
    metrics: dict
    examples: list[dict]
    duration_s: float
    tokens_per_s: float | None


def load_model(base_model: str, revision: str | None = None, adapter_dir: str | None = None):
    kw = {"revision": revision} if revision else {}
    if adapter_dir:
        if not (Path(adapter_dir) / "adapters.safetensors").exists():
            raise FileNotFoundError(f"no adapter at {adapter_dir}")
        kw["adapter_path"] = adapter_dir
    return load(base_model, **kw)


def prompt_tokens(tokenizer, messages: list[dict]) -> list[int]:
    """Everything except the reference answer, ending with the assistant turn opener."""
    convo = [m for m in messages if m["role"] != "assistant"]
    return tokenizer.apply_chat_template(convo, add_generation_prompt=True, return_dict=False)


def generate_batch(model, tokenizer, prompts: list[list[int]], max_tokens: int = MAX_TOKENS, batch_size: int = 16) -> tuple[list[str], int]:
    texts: list[str] = []
    generated = 0
    for i in range(0, len(prompts), batch_size):
        r = batch_generate(model, tokenizer, prompts[i : i + batch_size], max_tokens=max_tokens, return_token_ids=True)
        texts.extend(r.texts)
        generated += sum(len(t) for t in (r.token_ids or []))
    return texts, generated


def split_loss(model, tokenizer, rows: list[dict], batch_size: int = 4, max_seq_length: int = 512) -> float:
    """Mean cross-entropy on the assistant tokens (teacher forcing), same masking as training."""
    from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
    from mlx_lm.tuner.trainer import evaluate

    ds = CacheDataset(ChatDataset(rows, tokenizer, mask_prompt=True))
    return round(float(evaluate(model=model, dataset=ds, batch_size=min(batch_size, len(rows)), num_batches=-1, max_seq_length=max_seq_length)), 4)


def evaluate_rows(model, tokenizer, rows: list[dict], scoring: dict, max_tokens: int = MAX_TOKENS, on_progress=lambda done, total: None, should_stop=lambda: False) -> EvalOutput:
    t0 = time.time()
    prompts = [prompt_tokens(tokenizer, r["messages"]) for r in rows]
    outputs: list[str] = []
    generated = 0
    for i in range(0, len(prompts), 16):
        if should_stop():
            raise KeyboardInterrupt("cancelled")
        texts, n = generate_batch(model, tokenizer, prompts[i : i + 16], max_tokens)
        outputs.extend(texts)
        generated += n
        on_progress(len(outputs), len(prompts))
    examples, json_scores, text_scores = [], [], []
    for r, out in zip(rows, outputs, strict=True):
        ref = r["messages"][-1]["content"]
        ex = {"input": next(m["content"] for m in reversed(r["messages"][:-1]) if m["role"] == "user"), "reference": ref, "output": out}
        if scoring["mode"] == "json_fields":
            sc = score(out, ref, scoring["fields"])
            json_scores.append(sc)
            ex |= {"parsed": sc.parsed, "fields": sc.fields, "all_correct": sc.all_correct}
        else:
            st = score_text(out, ref)
            text_scores.append(st)
            ex |= st
        examples.append(ex)
    metrics = aggregate(json_scores, scoring["fields"]) if scoring["mode"] == "json_fields" else aggregate_text(text_scores)
    metrics["loss"] = split_loss(model, tokenizer, rows)
    dt = time.time() - t0
    return EvalOutput(metrics, examples, round(dt, 1), round(generated / dt, 1) if dt > 0 and generated else None)
