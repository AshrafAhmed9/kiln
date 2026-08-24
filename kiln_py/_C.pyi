"""Typed stub for the kiln_py._C pybind11 extension -- the §6 boundary
contract, written down. Every symbol the C++ compute layer exposes to
Python must be listed here; if it isn't, it isn't part of the contract.
"""
from typing import List, Optional

import numpy as np
import numpy.typing as npt

def ping() -> str: ...

class ModelConfig:
    vocab_size: int
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_hidden: int
    max_seq_len: int
    rms_eps: float
    rope_theta: float
    def __init__(self) -> None: ...

class KVCache:
    def __init__(self, n_layers: int, max_seq_len: int, n_kv_heads: int,
                 head_dim: int) -> None: ...
    @property
    def length(self) -> int: ...

class Model:
    @staticmethod
    def load_random(config: ModelConfig, seed: int) -> "Model": ...
    @staticmethod
    def load_from_safetensors(config: ModelConfig, path: str) -> "Model": ...
    def forward(self, tokens: npt.NDArray[np.int32], batch_size: int,
                seq_len: int, valid_lengths: Optional[List[int]],
                start_pos: int,
                cache: Optional[KVCache]) -> npt.NDArray[np.float32]: ...
    def forward_decode_batch(self, tokens: npt.NDArray[np.int32],
                             positions: List[int], caches: List[KVCache]) -> npt.NDArray[np.float32]: ...
    def merge_lora_into_layer(self, layer_idx: int, which: str,
                              lora_a: npt.NDArray[np.float32],
                              lora_b: npt.NDArray[np.float32], scale: float) -> None: ...
    @property
    def config(self) -> ModelConfig: ...

class SamplerConfig:
    temperature: float
    top_k: int
    top_p: float
    repetition_penalty: float
    def __init__(self) -> None: ...

def sample(logits: npt.NDArray[np.float32], config: SamplerConfig,
           previous_tokens: List[int], seed: int) -> int: ...

class BpeTokenizer:
    @staticmethod
    def load(tokenizer_json_path: str) -> "BpeTokenizer": ...
    def encode(self, text: str) -> List[int]: ...
    def decode(self, ids: List[int]) -> bytes: ...
