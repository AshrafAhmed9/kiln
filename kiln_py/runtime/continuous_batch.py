"""Real cached decode batching for the Python scheduler.

The scheduler chooses which requests advance. This adapter owns the per-request
KV caches and turns one scheduler step into one C++ batched-decode call for all
already-prefilled requests. New prompts of different lengths are prefilled
together too, in one ragged (unpadded) batched call -- see
Model::ForwardPrefillBatch and docs/learning/phase-25.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from kiln_py import _C
from kiln_py.scheduler.scheduler import Request


@dataclass
class _SequenceState:
    cache: _C.KVCache
    logits: np.ndarray
    sampler_config: _C.SamplerConfig
    seed: int


class ContinuousBatchExecutor:
    """Owns runtime state for requests driven by one Scheduler instance."""

    def __init__(self, model: Any):
        self._model = model
        self._states: dict[int, _SequenceState] = {}

    def register(self, request: Request, sampler_config: _C.SamplerConfig,
                 seed: int) -> None:
        self._states[request.request_id] = _SequenceState(
            cache=_C.KVCache(
                self._model.config.n_layers,
                self._model.config.max_seq_len,
                self._model.config.n_kv_heads,
                self._model.config.head_dim,
            ),
            logits=np.empty(0, dtype=np.float32),
            sampler_config=sampler_config,
            seed=seed,
        )

    def forget(self, request_id: int) -> None:
        self._states.pop(request_id, None)

    def __call__(self, requests: list[Request]) -> list[int]:
        fresh = [request for request in requests if not self._states[request.request_id].logits.size]
        fresh_ids = {request.request_id for request in fresh}
        if fresh:
            # One ragged batched call for every new prompt this step, instead
            # of one Model::Forward call per prompt -- no padding, and every
            # matmul in every layer runs once over the true total token count.
            concatenated = np.concatenate(
                [np.asarray(request.prompt_tokens, dtype=np.int32) for request in fresh]
            )
            seq_lengths = [len(request.prompt_tokens) for request in fresh]
            fresh_states = [self._states[request.request_id] for request in fresh]
            logits = self._model.forward_prefill_batch(
                concatenated, seq_lengths, [state.cache for state in fresh_states]
            )
            row = 0
            for state, length in zip(fresh_states, seq_lengths):
                state.logits = logits[row + length - 1]
                row += length

        chosen = []
        decode_requests = []
        for request in requests:
            state = self._states[request.request_id]
            token = _C.sample(
                state.logits, state.sampler_config, request.tokens,
                state.seed + request.new_tokens_generated,
            )
            chosen.append(token)
            if request.request_id not in fresh_ids:
                decode_requests.append(request)

        # Fresh requests already have their prompt in the cache. Their chosen
        # token becomes the input to the next step; existing requests can all
        # consume their previous output together in one cached batch now.
        if decode_requests:
            decode_tokens = np.asarray(
                [chosen[index] for index, request in enumerate(requests)
                 if request.request_id not in fresh_ids],
                dtype=np.int32,
            )
            states = [self._states[request.request_id] for request in decode_requests]
            logits = self._model.forward_decode_batch(
                decode_tokens,
                [state.cache.length for state in states],
                [state.cache for state in states],
            )
            for state, row in zip(states, logits):
                state.logits = row
        return chosen
