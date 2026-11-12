# TuneLab — Progress

| # | Milestone | Commit | Checks |
|---|---|---|---|
| 1 | Core: synthetic triage task (exact labels, held-out templates), strict scoring, genuine MLX LoRA training with weight-update verification, batched evaluation; MPICH abort fixed via `MLX_MPI_LIBNAME` | `54a0125` | pytest; 30-step smoke on the pinned model (val loss 1.14 → 0.0096) |
| 2 | Datasets, SQLite job queue with worker subprocesses, SSE, API, CLI | `5984808` | pytest 24/24 |
| 3 | Next.js lab UI | `65326e6` | eslint · tsc · vitest 8/8 |
| — | Dispatcher runs evaluations before queued training | `ceaa953` | pytest 25/25 |
| 4 | Real-training Playwright journey, Linux `mlx[cpu]`, Docker images, CI, results exporter | `bba46fc` | E2E 7/7 (real 10-step run); backend suite passes in a Linux container; images build |
| — | UTC timestamps, small-loss formatting, empty-revision config | `ad51c4e` | pytest 27/27 · vitest 8/8 |
| — | CI E2E on a tiny random model (pinned model too slow on MLX CPU) | `5304443` | E2E 7/7 locally in both configurations |
| 5 | Experiments (3 completed runs + shared baselines; docs/RESULTS.md, eval/results.json), 10 screenshots, README + docs | see git log | — |

Note: one r16 run (`run_4f92a7e9a3a3`) is recorded as failed. It was interrupted (SIGTERM) when I
restarted the API during development, and its tool stopped the worker's process tree. It was re-run
as `run_319cfabbdfcc`.
