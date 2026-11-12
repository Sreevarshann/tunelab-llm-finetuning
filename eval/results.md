| Run | Status | Steps | Rank | Train time | Val loss (start → end) | Test: base → LoRA | Held-out: base → LoRA |
|---|---|---|---|---|---|---|---|
| r8 · lr1e-4 · 300 steps (`run_afb78b8ed7ee`) | completed | 300 | 8 | 9.7 min | 1.126 → 0.00007 | 8.3% → **100.0%** | 5.0% → **63.3%** |
| Short · 100 steps (`run_4b5d00e6071a`) | completed | 100 | 8 | 4.4 min | 1.126 → 0.00043 | 8.3% → **100.0%** | 5.0% → **50.0%** |
| r16 · dropout 0.1 · 300 steps (`run_4f92a7e9a3a3`) | failed | 300 | 16 | — | — | — | — |
| r16 · dropout 0.1 · 300 steps (`run_319cfabbdfcc`) | completed | 300 | 16 | 10.1 min | 1.126 → 0.00130 | 8.3% → **99.2%** | 5.0% → **40.0%** |
