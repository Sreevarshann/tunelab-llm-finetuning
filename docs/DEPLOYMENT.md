# Deployment

**Native (recommended on Apple Silicon, Metal GPU):**

```bash
cd backend && uv sync && uv run tunelab serve          # :8114
cd frontend && npm ci && npm run dev -- --port 3104
```

The first job downloads the pinned base model (~290 MB) into the Hugging Face cache.

**Docker Compose:** `docker compose up --build` builds `api` and `web` (project, network and volume
`tunelab`; ports bound to 127.0.0.1). Linux containers cannot use the Apple GPU, so **MLX runs on
its CPU backend there**. Training and evaluation are real but much slower. The data dir and HF cache
persist in the `tunelab-data` volume.

| Variable | Default |
|---|---|
| `TUNELAB_DATA_DIR` | `var` (database, datasets, adapters, logs) |
| `TUNELAB_BASE_MODEL` / `TUNELAB_BASE_REVISION` | `mlx-community/Qwen2.5-0.5B-Instruct-4bit` / `a5339a41…` |
| `TUNELAB_CORS_ORIGINS` | `["http://localhost:3104","http://127.0.0.1:3104"]` |
| `TUNELAB_MAX_UPLOAD_MB` | 20 |
| `TUNELAB_DISPATCHER` | `true` (set `false` to stop the API from starting jobs) |
| `TUNELAB_API_ORIGIN` (web) | `http://127.0.0.1:8114` |

No cloud resources are used or required.
