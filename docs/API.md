# API

Base path `/api`. Validation errors are 422 (FastAPI `detail`). There is no authentication: TuneLab is a single-user local tool and binds to 127.0.0.1 (see [SECURITY.md](SECURITY.md)).

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | base model + revision, mlx / mlx-lm versions, device, running job |
| GET | `/datasets` · `/datasets/{id}` · `/datasets/{id}/rows?split=&offset=&limit=` | |
| POST | `/datasets/synthetic` | `{name?, seed?, train?, valid?, test?, test_heldout?}` → 201 |
| POST | `/datasets/upload` | multipart: `name` + `file` (auto split) or `train` + `valid` (+ `test`); 422 lists line errors |
| POST | `/runs` | `{dataset_id, name?, config: {rank, scale, dropout, num_layers, learning_rate, iters, batch_size, max_seq_length, steps_per_report, steps_per_eval, seed, mask_prompt}}` → queued job |
| GET | `/runs` · `/runs/{id}` | detail includes the dataset and tuned + baseline evaluations |
| POST | `/runs/{id}/cancel` | queued → cancelled at once; running → stops at the next step boundary |
| POST | `/runs/{id}/evaluate` | (re)queue evaluations for a completed run |
| GET | `/runs/{id}/adapter` | `adapters.safetensors` |
| GET | `/evaluations/{id}/examples?only_errors=&offset=&limit=` | per-example input, reference, output, field correctness |
| GET | `/jobs` · `/jobs/{id}/events` | SSE: `status`, `stage`, `train`, `val`, `progress`, then `end` |
| GET | `/compare?run_ids=a,b` | runs with their evaluations |
| POST | `/playground` | `{messages, run_ids: [null, "run_…"], max_tokens}`; `null` = base model |
| GET | `/tasks/ticket_triage/system_prompt` | |
