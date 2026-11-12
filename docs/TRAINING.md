# Training: what makes it real

TuneLab runs **supervised LoRA fine-tuning**, not prompting or retrieval:

1. The base model is loaded and **frozen** (`model.freeze()`).
2. `mlx_lm.tuner.utils.linear_to_lora_layers` wraps the linear layers of the last `num_layers`
   transformer blocks with LoRA (`W x + scale · B A x`). Only `A` and `B` are trainable: 2.93M of
   494M parameters (0.59%) at rank 8 over 16 blocks.
3. Chat examples are tokenised with the model's chat template. With `mask_prompt`, the loss is
   computed only on assistant tokens.
4. mlx-lm's training loop runs Adam on those parameters for `iters` steps, reporting train loss
   every `steps_per_report` and validation loss every `steps_per_eval`.
5. The adapter (`adapters.safetensors`) and an mlx-lm-compatible `adapter_config.json` are saved.

**Verification recorded on every run** (`training_result.json`, run detail page):
- `lora_b_norm_before` is 0.0, because LoRA initialises `B = 0` so a fresh adapter is a no-op.
  `lora_b_norm_after` must be > 0, or the run fails with "training did not produce updated adapter weights".
- SHA-256 of the adapter file, trainable vs total parameters, loss curves, duration and peak memory.
- Evaluation loads the saved file into a fresh base model (`load(..., adapter_path=...)`), so the
  scores measure the persisted artefact rather than in-memory state.

**Tests** (`tests/test_training_tiny.py`) train a tiny randomly initialised Llama-style model
through the same `run_lora` code in seconds (CPU in CI). They assert that `lora_b` moves off zero,
validation loss drops, the adapter round-trips, and that loading it changes a fresh model's outputs.

**Hyperparameters** (validated by the API): rank 1–128, scale (mlx-lm's multiplier on the LoRA
update), dropout, number of adapted blocks, learning rate, steps, batch size, max sequence length,
seed, prompt masking.

**Environment note:** MLX's distributed init selects any MPI library it finds. On a machine with
Anaconda's MPICH (not Open MPI) it aborted on the first collective (exit 134). TuneLab trains on a
single device, so `tunelab/__init__.py` sets `MLX_MPI_LIBNAME` to a non-existent library and MLX
falls back to its single-process group.
