"""Genuine LoRA supervised fine-tuning with MLX-LM.

Only LoRA matrices are trainable (the base weights are frozen). Training runs mlx-lm's optimiser
loop with a callback that reports real losses. The result is verified: the adapter must contain
non-zero ``lora_b`` matrices (LoRA initialises ``lora_b`` to zero, so a zero norm means the
weights were never updated), and its SHA-256 is recorded.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import CacheDataset, ChatDataset
from mlx_lm.tuner.trainer import TrainingArgs, evaluate, train
from mlx_lm.tuner.utils import linear_to_lora_layers

Event = Callable[[str, dict], None]


@dataclass
class LoraConfig:
    base_model: str
    base_revision: str | None = None
    rank: int = 8
    scale: float = 20.0
    dropout: float = 0.0
    num_layers: int = 16
    learning_rate: float = 1e-4
    iters: int = 300
    batch_size: int = 4
    max_seq_length: int = 512
    steps_per_report: int = 10
    steps_per_eval: int = 50
    val_batches: int = 20
    mask_prompt: bool = True
    seed: int = 0
    grad_checkpoint: bool = False

    def validate(self) -> None:
        checks = [
            (1 <= self.rank <= 128, "rank must be 1-128"),
            (0 < self.scale <= 100, "scale must be in (0, 100]"),
            (0 <= self.dropout < 1, "dropout must be in [0, 1)"),
            (1 <= self.num_layers <= 128, "num_layers must be 1-128"),
            (1e-6 <= self.learning_rate <= 1e-2, "learning_rate must be 1e-6 to 1e-2"),
            (1 <= self.iters <= 20000, "iters must be 1-20000"),
            (1 <= self.batch_size <= 64, "batch_size must be 1-64"),
            (32 <= self.max_seq_length <= 8192, "max_seq_length must be 32-8192"),
        ]
        bad = [m for ok, m in checks if not ok]
        if bad:
            raise ValueError("; ".join(bad))


@dataclass
class TrainResult:
    adapter_dir: str
    adapter_sha256: str
    trainable_params: int
    total_params: int
    lora_b_norm_before: float
    lora_b_norm_after: float
    initial_val_loss: float
    final_val_loss: float
    final_train_loss: float | None
    duration_s: float
    peak_memory_gb: float
    train_curve: list[dict] = field(default_factory=list)
    val_curve: list[dict] = field(default_factory=list)


def _lora_b_norm(model) -> float:
    total = 0.0
    for name, w in tree_flatten(model.trainable_parameters()):
        if name.endswith("lora_b"):
            total += float(mx.sum(w.astype(mx.float32) ** 2).item())
    return math.sqrt(total)


def _count(params) -> int:
    return sum(v.size for _, v in tree_flatten(params))


def base_parameter_count(model) -> int:
    """Logical parameter count. Quantized layers store packed uint32 weights plus per-group
    scales/biases, so their logical size is packed_elements * 32 / bits (scales excluded)."""
    from mlx import nn

    total = 0
    for _, m in model.named_modules():
        if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            total += m.weight.size * 32 // m.bits + (m["bias"].size if "bias" in m else 0)
        elif isinstance(m, (nn.Linear, nn.Embedding, nn.RMSNorm, nn.LayerNorm)):
            total += sum(v.size for _, v in tree_flatten(m.parameters()))
    return total


class _Reporter(TrainingCallback):
    def __init__(self, emit: Event, should_stop: Callable[[], bool]):
        self.emit, self.should_stop = emit, should_stop
        self.train: list[dict] = []
        self.val: list[dict] = []

    def on_train_loss_report(self, info: dict) -> None:
        row = {
            "iteration": info["iteration"],
            "loss": float(info["train_loss"]),
            "lr": float(info["learning_rate"]),
            "tokens_per_second": float(info["tokens_per_second"]),
            "peak_memory_gb": float(info["peak_memory"]),
        }
        self.train.append(row)
        self.emit("train", row)
        if self.should_stop():
            raise KeyboardInterrupt("cancelled")

    def on_val_loss_report(self, info: dict) -> None:
        row = {"iteration": info["iteration"], "loss": float(info["val_loss"])}
        self.val.append(row)
        self.emit("val", row)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_lora(
    cfg: LoraConfig, train_rows: list[dict], valid_rows: list[dict], out_dir: Path, emit: Event = lambda *_: None, should_stop: Callable[[], bool] = lambda: False, loaded: tuple | None = None
) -> TrainResult:
    cfg.validate()
    if not train_rows or not valid_rows:
        raise ValueError("training and validation sets must be non-empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.random.seed(cfg.seed)
    t0 = time.time()
    emit("stage", {"stage": "loading_model", "model": cfg.base_model})
    if loaded is not None:  # tests pass a tiny in-memory model
        model, tokenizer = loaded
    else:
        model, tokenizer = load(cfg.base_model, revision=cfg.base_revision) if cfg.base_revision else load(cfg.base_model)
    total_params = base_parameter_count(model)

    model.freeze()
    num_layers = min(cfg.num_layers, len(model.layers))
    lora_parameters = {"rank": cfg.rank, "scale": cfg.scale, "dropout": cfg.dropout}
    linear_to_lora_layers(model, num_layers, lora_parameters)
    trainable = _count(model.trainable_parameters())
    before = _lora_b_norm(model)
    emit("stage", {"stage": "training", "trainable_params": trainable, "total_params": total_params, "lora_layers": num_layers})

    train_set = CacheDataset(ChatDataset(train_rows, tokenizer, mask_prompt=cfg.mask_prompt))
    valid_set = CacheDataset(ChatDataset(valid_rows, tokenizer, mask_prompt=cfg.mask_prompt))
    adapter_file = out_dir / "adapters.safetensors"
    args = TrainingArgs(
        batch_size=cfg.batch_size,
        iters=cfg.iters,
        val_batches=cfg.val_batches,
        steps_per_report=cfg.steps_per_report,
        steps_per_eval=cfg.steps_per_eval,
        steps_per_save=cfg.iters + 1,
        adapter_file=str(adapter_file),
        max_seq_length=cfg.max_seq_length,
        grad_checkpoint=cfg.grad_checkpoint,
    )

    reporter = _Reporter(emit, should_stop)  # mlx-lm reports validation loss at iteration 1 (before any update)
    model.train()
    train(model=model, optimizer=optim.Adam(learning_rate=cfg.learning_rate), train_dataset=train_set, val_dataset=valid_set, args=args, training_callback=reporter)

    model.eval()
    final_val = float(evaluate(model=model, dataset=valid_set, batch_size=cfg.batch_size, num_batches=cfg.val_batches, max_seq_length=cfg.max_seq_length))
    after = _lora_b_norm(model)
    initial_val = reporter.val[0]["loss"] if reporter.val else float("nan")
    if not adapter_file.exists() or after == 0.0:
        raise RuntimeError("training did not produce updated adapter weights")

    (out_dir / "adapter_config.json").write_text(
        json.dumps(
            {"fine_tune_type": "lora", "num_layers": num_layers, "lora_parameters": lora_parameters, "base_model": cfg.base_model, "base_revision": cfg.base_revision, "tunelab": asdict(cfg)}, indent=2
        )
    )
    result = TrainResult(
        adapter_dir=str(out_dir),
        adapter_sha256=sha256_file(adapter_file),
        trainable_params=trainable,
        total_params=total_params,
        lora_b_norm_before=before,
        lora_b_norm_after=after,
        initial_val_loss=initial_val,
        final_val_loss=final_val,
        final_train_loss=reporter.train[-1]["loss"] if reporter.train else None,
        duration_s=round(time.time() - t0, 1),
        peak_memory_gb=round(mx.get_peak_memory() / 1e9, 2),
        train_curve=reporter.train,
        val_curve=reporter.val,
    )
    (out_dir / "training_result.json").write_text(json.dumps(asdict(result), indent=2))
    emit("stage", {"stage": "saved", "adapter_sha256": result.adapter_sha256})
    return result
