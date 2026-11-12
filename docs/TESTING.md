# Testing

| Suite | Command | Count |
|---|---|---|
| Backend (pytest) | `cd backend && uv run pytest -q` | 27: scoring, task generator (determinism, label/text consistency, held-out templates), datasets, job queue (one-at-a-time, eval priority, dead-worker recovery, cancel), API, and **real LoRA training on a tiny random model** |
| Real-model tests | `TUNELAB_SLOW=1 uv run pytest -m slow` | reserved for local Apple Silicon runs |
| Frontend unit | `cd frontend && npm test` | 8 (run/evaluation join, headline metric, strict JSON parsing, status) |
| E2E | `cd frontend && npx playwright test` | 7: upload validation, synthetic dataset, **a real 10-step LoRA run on the pinned base model** with live loss and verification, base vs LoRA evaluation on both test splits, compare, playground, cancel |

The E2E stack is isolated: a fresh data dir, its own API port (8194) and a production web build in
`.next-e2e`.

**Two E2E configurations, both with real jobs:**
- **Local, Apple Silicon (Metal):** the pinned Qwen2.5-0.5B-Instruct-4bit. 7/7 pass in ~1.3 min,
  including the real 10-step run.
- **CI, Ubuntu (MLX CPU backend):** the first CI attempt with the pinned model did not reach its first
  loss report within 12 minutes. A local benchmark on CPU did not finish 4 steps plus an 8-example
  evaluation within 10 minutes, because MLX's CPU path for 4-bit matmuls is slow. CI therefore builds a
  tiny randomly initialised Llama-style model with the base tokenizer
  (`backend/scripts/make_tiny_model.py`) and runs the same journey on it via `TUNELAB_BASE_MODEL`.
  Training, the weight-update check, evaluation, SSE and the UI are real; the tiny model's outputs are
  meaningless, so CI asserts mechanics, never quality. 7/7 locally with this configuration too.

The backend suite (including tiny-model training) also passes on Linux CPU, in CI and in a
`python:3.13-slim` container.
