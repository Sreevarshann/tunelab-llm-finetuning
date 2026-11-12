# Decisions

1. **MLX-LM on Apple Silicon.** Native Metal training of a 0.5B model fits in under 3 GB peak memory on a 16 GB M4; no cloud GPU or cost.
2. **4-bit base + LoRA (QLoRA-style).** A frozen quantised base keeps memory and download small; the LoRA matrices are trained at higher precision.
3. **Pinned base revision.** Results are tied to exact weights; baselines are keyed by revision.
4. **Synthetic task with exact labels and held-out templates.** It gives perfect ground truth. The held-out split separates learning the mapping from memorising phrasing, which in-distribution test accuracy alone would hide.
5. **Same prompt for base and tuned.** The comparison isolates the effect of the adapter rather than prompt engineering. The base gets the full schema description, not a strawman.
6. **Worker subprocess per job.** It frees GPU memory deterministically, makes cancel and kill safe, and a crash in MLX cannot take down the API.
7. **SQLite + SSE.** A local single-user tool needs no external services; events are rows that the stream replays.
8. **Evaluations before queued training.** They are short and are what a finished run is waiting on.
9. **Linux CI on MLX's CPU backend** instead of mocking training. The tiny-model test and a short real E2E run keep "the weights really change" verified on every push.
10. **No auth.** It is a local lab tool, documented as such, rather than a half-built multi-user system.
