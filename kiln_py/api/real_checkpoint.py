"""Loads a real trained checkpoint and serves it on the real forward pass,
CPU or CUDA depending on what's asked for. This is the non-toy path: no
random weights, no byte-level placeholder tokenizer.

CudaModel owns exactly one device-resident KV cache (see
csrc/executor/cuda_model.cu), unlike the CPU Model's per-request KVCache
objects. So this path serves one request at a time behind a lock.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterator

from kiln_py import _C
from kiln_py.runtime.generate import generate, generate_stream


def config_from_hf(config_path: Path) -> "_C.ModelConfig":
    """Same field mapping tools/hf_parity.py uses to build a ModelConfig
    from a Hugging Face Llama-family config.json.
    """
    raw = json.loads(config_path.read_text())
    if raw.get("model_type") != "llama":
        raise ValueError(f"expected a Llama-family checkpoint, got {raw.get('model_type')!r}")
    config = _C.ModelConfig()
    config.vocab_size = raw["vocab_size"]
    config.hidden_size = raw["hidden_size"]
    config.n_layers = raw["num_hidden_layers"]
    config.n_heads = raw["num_attention_heads"]
    config.n_kv_heads = raw["num_key_value_heads"]
    config.head_dim = raw.get("head_dim") or (config.hidden_size // config.n_heads)
    config.ffn_hidden = raw["intermediate_size"]
    config.max_seq_len = raw["max_position_embeddings"]
    config.rms_eps = raw["rms_norm_eps"]
    config.rope_theta = raw["rope_theta"]
    return config


def load_real_model_and_tokenizer(checkpoint_dir: Path, device: str):
    """device is "cpu" or "cuda". Raises if "cuda" is requested but this
    build has no CUDA support, rather than silently falling back to CPU.
    """
    config = config_from_hf(checkpoint_dir / "config.json")
    cpu_model = _C.Model.load_from_safetensors(
        config, str(checkpoint_dir / "model.safetensors")
    )
    tokenizer = _C.BpeTokenizer.load(str(checkpoint_dir / "tokenizer.json"))
    if device == "cuda":
        if not hasattr(_C, "CudaModel"):
            raise RuntimeError(
                "KILN_DEVICE=cuda but this build has no CUDA support -- "
                "rebuild with -DKILN_BUILD_CUDA=ON"
            )
        return _C.CudaModel(cpu_model), tokenizer
    return cpu_model, tokenizer


class RealCheckpointService:
    """Same .complete()/.stream() shape as CompletionBatchService, so
    kiln_py/api/app.py doesn't need to know which backend is live.
    """

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer
        self._lock = threading.Lock()

    def _reset(self) -> None:
        if hasattr(self._model, "reset_cache"):
            self._model.reset_cache()

    def complete(self, prompt: str, max_new_tokens: int, sampler_config,
                 seed: int) -> str:
        with self._lock:
            self._reset()
            return generate(self._model, self._tokenizer, prompt,
                             max_new_tokens, sampler_config, seed)

    def stream(self, prompt: str, max_new_tokens: int, sampler_config,
               seed: int) -> Iterator[int]:
        with self._lock:
            self._reset()
            yield from generate_stream(self._model, self._tokenizer, prompt,
                                       max_new_tokens, sampler_config, seed)
