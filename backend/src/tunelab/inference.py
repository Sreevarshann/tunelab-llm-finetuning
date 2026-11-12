"""Playground inference in the API process: base model and one adapter kept warm."""

from __future__ import annotations

import threading
from collections import OrderedDict

from mlx_lm import generate

from .config import get_settings
from .evaluation import load_model

_lock = threading.Lock()
_cache: OrderedDict[str, tuple] = OrderedDict()
MAX_LOADED = 2


def _get(adapter_dir: str | None):
    key = adapter_dir or "<base>"
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    s = get_settings()
    pair = load_model(s.base_model, s.base_revision, adapter_dir)
    _cache[key] = pair
    while len(_cache) > MAX_LOADED:
        _cache.popitem(last=False)
    return pair


def complete(messages: list[dict], adapter_dir: str | None, max_tokens: int = 160) -> str:
    with _lock:  # MLX generation is not re-entrant; serialise requests
        model, tok = _get(adapter_dir)
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, return_dict=False)
        return generate(model, tok, prompt, max_tokens=max_tokens)
