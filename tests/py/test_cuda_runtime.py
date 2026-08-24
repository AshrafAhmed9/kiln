from types import SimpleNamespace

import numpy as np

from kiln_py import _C
from kiln_py.runtime.generate import generate


class FakeCudaModel:
    config = SimpleNamespace(n_layers=1, max_seq_len=8, n_kv_heads=1,
                             head_dim=4)

    def __init__(self) -> None:
        self.calls: list[tuple[list[int], int]] = []

    def forward_cached(self, tokens: np.ndarray, start_pos: int) -> np.ndarray:
        self.calls.append((tokens.tolist(), start_pos))
        return np.array([[0.0, 1.0]], dtype=np.float32)


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        assert text == "prompt"
        return [0]

    def decode(self, ids: list[int]) -> bytes:
        assert ids == [1, 1]
        return b"ok"


def test_generate_uses_device_owned_cache_for_cuda_model(monkeypatch) -> None:
    monkeypatch.setattr(_C, "sample", lambda *_args: 1)
    model = FakeCudaModel()

    result = generate(model, FakeTokenizer(), "prompt", max_new_tokens=2,
                      sampler_config=object())

    assert result == "ok"
    assert model.calls == [([0], 0), ([1], 1), ([1], 2)]
