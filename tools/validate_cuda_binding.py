"""Check the CUDA pybind surface against the CPU model on an assigned GPU."""

from __future__ import annotations

import numpy as np

from kiln_py import _C


def tiny_config() -> _C.ModelConfig:
    config = _C.ModelConfig()
    config.vocab_size = 16
    config.hidden_size = 8
    config.n_layers = 2
    config.n_heads = 2
    config.n_kv_heads = 1
    config.head_dim = 4
    config.ffn_hidden = 16
    config.max_seq_len = 8
    return config


def main() -> None:
    if not hasattr(_C, "CudaModel"):
        raise RuntimeError("Kiln was not built with KILN_BUILD_CUDA=ON")
    model = _C.Model.load_random(tiny_config(), 43)
    cuda_model = _C.CudaModel(model)
    prompt = np.array([1, 2, 3], dtype=np.int32)
    cpu_cache = _C.KVCache(2, 8, 1, 4)
    expected_prompt = model.forward(prompt, 1, len(prompt), None, 0, cpu_cache)
    actual_prompt = cuda_model.forward_cached(prompt, 0)
    np.testing.assert_allclose(actual_prompt, expected_prompt, rtol=1e-4, atol=1e-4)

    next_token = np.array([4], dtype=np.int32)
    expected_next = model.forward(next_token, 1, 1, None, len(prompt), cpu_cache)
    actual_next = cuda_model.forward_cached(next_token, len(prompt))
    np.testing.assert_allclose(actual_next, expected_next, rtol=1e-4, atol=1e-4)
    print("CUDA pybind prefill and cached decode match the CPU model")


if __name__ == "__main__":
    main()
