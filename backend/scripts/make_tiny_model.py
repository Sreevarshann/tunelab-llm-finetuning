"""Write a tiny randomly initialised Llama-style model in MLX format, with the pinned base model's
tokenizer, for CPU-only CI. Training, evaluation and the UI run for real on it; its outputs are
meaningless (random weights), so CI asserts mechanics, never quality.

    uv run python scripts/make_tiny_model.py var-tiny-model
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten
from mlx_lm.models.llama import Model, ModelArgs

import tunelab  # noqa: F401

out = Path(sys.argv[1] if len(sys.argv) > 1 else "var-tiny-model")
out.mkdir(parents=True, exist_ok=True)
patterns = ["tokenizer*", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"]
src = Path(snapshot_download("mlx-community/Qwen2.5-0.5B-Instruct-4bit", revision="a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3", allow_patterns=patterns))
TOKENIZER_FILES = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"}
for f in src.iterdir():  # the cached snapshot may also hold the real weights; copy tokenizer files only
    if f.name in TOKENIZER_FILES:
        shutil.copyfile(f, out / f.name)
vocab = json.loads((src / "tokenizer.json").read_text())
size = max([t["id"] for t in vocab.get("added_tokens", [])] + list(vocab["model"]["vocab"].values())) + 1
cfg = {
    "model_type": "llama",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "rms_norm_eps": 1e-5,
    "vocab_size": size,
    "tie_word_embeddings": True,
    "max_position_embeddings": 2048,
}
mx.random.seed(0)
model = Model(ModelArgs(**cfg))
mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(model.parameters())))
(out / "config.json").write_text(json.dumps(cfg, indent=2))
print(f"wrote {out} (vocab {size}, {sum(v.size for _, v in tree_flatten(model.parameters())):,} params)")
