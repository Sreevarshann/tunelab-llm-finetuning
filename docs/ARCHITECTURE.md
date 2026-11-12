# Architecture

```
 Browser ── Next.js 16 (:3104) ── /api rewrite ──▶ FastAPI (:8114) ──▶ SQLite (WAL): datasets, jobs, events
                                                     │  Dispatcher thread: one job at a time
                                                     ▼
                                          worker subprocess: python -m tunelab.worker <job>
                                             ├─ train:    mlx-lm LoRA on the frozen base → adapters.safetensors
                                             └─ evaluate: base or base+adapter → batched greedy decoding + scoring
                                                     │
                                         var/datasets/<id>/*.jsonl · var/jobs/<id>/{adapter,examples.json,worker.log}
```

| Piece | Notes |
|---|---|
| Base model | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` pinned to revision `a5339a4` (494M logical parameters, 4-bit weights) |
| Training | `training.py`: freeze base → `linear_to_lora_layers` (last N blocks) → mlx-lm `train()` with Adam, prompt-masked chat loss, loss callbacks |
| Evaluation | `evaluation.py`: same system prompt and greedy decoding for base and tuned models; JSON-field or text scoring; teacher-forced loss |
| Jobs | Worker subprocesses isolate GPU memory, allow cancel (checked every ~2 s at step/batch boundaries, hard kill after 30 s) and cannot crash the API; stale `running` jobs are failed on restart |
| Streaming | Events are rows in `job_events`; `GET /api/jobs/:id/events` (SSE) replays then follows until a terminal status |
| Playground | API process keeps the base and one adapter warm (LRU of 2); generation is serialised |

Device: Metal GPU when run natively on Apple Silicon; MLX's CPU backend on Linux (CI, Docker).
