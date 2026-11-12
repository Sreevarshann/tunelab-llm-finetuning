# TuneLab — Plan

Ports: API 8114, web 3104 (verified free 2026-11-11). Apple Silicon (M4, 16 GB), MLX 0.32 / MLX-LM 0.32.

## Non-negotiables
- Training is **real LoRA SFT**: only LoRA matrices are trainable, they are updated by an optimiser,
  saved as safetensors, checksummed, and verified non-zero. Prompting or RAG is never presented as tuning.
- Baseline and tuned models are evaluated with the same prompt, decoding and test data; only the
  adapter differs. No metric is reported that was not computed by a run in this repo.
- Synthetic data is labelled synthetic everywhere; a held-out-template split measures generalisation.

## Architecture
- **Core (Python 3.13, uv):** `tasks/ticket_triage.py` (deterministic synthetic task), `datasets.py`
  (JSONL validation, deterministic splits, token stats), `training.py` (MLX-LM LoRA), `evaluation.py`
  (batched greedy decoding, JSON-field / text scoring, teacher-forced test loss), `scoring.py`.
- **Jobs:** SQLite (WAL) queue; one job at a time in a worker subprocess; events persisted and
  streamed over SSE; cancel and crash recovery; completed runs auto-queue tuned + baseline evals.
- **API (FastAPI):** datasets, runs, evaluations, compare, playground, adapter download.
- **Web (Next.js):** datasets, new run, live run view with loss curves + verification, evaluation
  comparison with per-example diffs, run comparison, playground.
- **Base model:** `mlx-community/Qwen2.5-0.5B-Instruct-4bit` pinned to revision `a5339a4` (QLoRA-style:
  4-bit frozen base, higher-precision LoRA matrices).

## Milestones
1. Core: task, scoring, genuine LoRA training, evaluation + tests (incl. tiny-model training test)
2. Jobs + API + tests
3. Real experiments: baseline vs tuned on in-distribution and held-out splits; small sweep
4. Frontend
5. E2E, screenshots, docs, CI, Docker
