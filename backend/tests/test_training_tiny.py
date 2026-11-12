"""Genuine LoRA training on a tiny randomly initialised model (no download, seconds on CPU or GPU).
Proves the TuneLab training path updates and saves adapter weights and lowers loss."""

import json

import mlx.core as mx
from mlx_lm.models.llama import Model, ModelArgs
from mlx_lm.tuner.utils import load_adapters

from tunelab.training import LoraConfig, run_lora, sha256_file


class CharTokenizer:
    """Minimal tokenizer with a chat template, enough for mlx-lm's ChatDataset."""

    eos_token_id = 1

    def _encode(self, text: str) -> list[int]:
        return [2 + (ord(c) % 60) for c in text]

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=False, return_dict=False, **_):
        ids = [0]
        for m in messages:
            ids += [*self._encode(f"<{m['role']}>{m['content']}"), 1]
        if add_generation_prompt:
            ids += self._encode("<assistant>")
        return ids


def tiny():
    mx.random.seed(0)
    model = Model(ModelArgs(model_type="llama", hidden_size=64, num_hidden_layers=2, intermediate_size=128, num_attention_heads=4, rms_norm_eps=1e-5, vocab_size=64))
    mx.eval(model.parameters())
    return model, CharTokenizer()


def rows(n):
    return [{"messages": [{"role": "user", "content": f"colour {i % 3}"}, {"role": "assistant", "content": ["red", "green", "blue"][i % 3]}]} for i in range(n)]


def test_lora_training_updates_and_saves_weights(tmp_path):
    cfg = LoraConfig(base_model="tiny-test", rank=4, num_layers=2, iters=60, batch_size=4, learning_rate=5e-3, steps_per_report=10, steps_per_eval=30, val_batches=2, max_seq_length=64)
    events = []
    r = run_lora(cfg, rows(48), rows(8), tmp_path, emit=lambda k, v: events.append(k), loaded=tiny())

    assert r.lora_b_norm_before == 0.0  # LoRA starts as a no-op
    assert r.lora_b_norm_after > 0.0  # ...and training changed it
    assert r.final_val_loss < r.initial_val_loss * 0.8  # real optimisation happened
    assert 0 < r.trainable_params < r.total_params
    assert r.adapter_sha256 == sha256_file(tmp_path / "adapters.safetensors")
    assert events.count("train") == 6 and "val" in events
    cfg_json = json.loads((tmp_path / "adapter_config.json").read_text())
    assert cfg_json["fine_tune_type"] == "lora" and cfg_json["lora_parameters"]["rank"] == 4

    # the saved adapter loads into a fresh base model and changes its outputs
    fresh, _ = tiny()
    x = mx.array([[0, 5, 6, 7]])
    before = fresh(x)
    load_adapters(fresh, str(tmp_path))
    assert not mx.allclose(before, fresh(x)).item()


def test_config_validation():
    import pytest

    with pytest.raises(ValueError, match="rank"):
        LoraConfig(base_model="x", rank=0).validate()
    with pytest.raises(ValueError, match="learning_rate"):
        LoraConfig(base_model="x", learning_rate=1.0).validate()
