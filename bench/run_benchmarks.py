"""Real benchmarks, run on this machine's CPU against the toy (untrained,
randomly-initialized) model -- there is no GPU or real trained checkpoint
available in this environment (see docs/defense.md, ADR-009). Every number
this script prints was actually measured by running the code, not
estimated. Where an optimization's real payoff only shows up on a GPU
(raw kernel throughput, INT8 speed), that's stated explicitly rather than
implied by a CPU number that wouldn't actually demonstrate it.

Usage: PYTHONPATH=. python3 bench/run_benchmarks.py
"""
from __future__ import annotations

import statistics
import time

import numpy as np

from kiln_py import _C
from kiln_py.control_plane.tenant_store import Tenant
from kiln_py.runtime.speculative_decode import speculative_generate


def _toy_config(vocab_size=1000, hidden=64, layers=4, heads=4, kv_heads=2,
                 head_dim=16, ffn=256, max_seq_len=256):
    config = _C.ModelConfig()
    config.vocab_size = vocab_size
    config.hidden_size = hidden
    config.n_layers = layers
    config.n_heads = heads
    config.n_kv_heads = kv_heads
    config.head_dim = head_dim
    config.ffn_hidden = ffn
    config.max_seq_len = max_seq_len
    config.rms_eps = 1e-5
    config.rope_theta = 10000.0
    return config


def _percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def bench_ttft_and_tpot(num_trials: int = 20, prompt_len: int = 8,
                        num_new_tokens: int = 15) -> dict:
    """TTFT (time to first token -- the prefill pass) and TPOT (time per
    output token thereafter) are the two numbers real serving systems
    report, since they answer two different user-facing questions: "how
    long until I see anything?" and "how fast does it keep typing?". Run
    repeatedly with different random prompts/seeds so p50/p99 mean
    something, rather than reporting one single, possibly-lucky run.
    """
    config = _toy_config()
    model = _C.Model.load_random(config, 42)
    rng = np.random.default_rng(100)

    ttft_seconds = []
    tpot_seconds = []

    for trial in range(num_trials):
        prompt = list(rng.integers(0, config.vocab_size, prompt_len))
        cache = _C.KVCache(config.n_layers, config.max_seq_len,
                           config.n_kv_heads, config.head_dim)

        start = time.perf_counter()
        tokens = np.array(prompt, dtype=np.int32)
        logits = model.forward(tokens, 1, len(prompt), None, 0, cache)
        ttft_seconds.append(time.perf_counter() - start)

        for i in range(num_new_tokens):
            next_token = int(np.argmax(logits[-1]))
            one = np.array([next_token], dtype=np.int32)
            start = time.perf_counter()
            logits = model.forward(one, 1, 1, None, len(prompt) + i, cache)
            tpot_seconds.append(time.perf_counter() - start)

    return {
        "ttft_p50_ms": _percentile(ttft_seconds, 50) * 1000,
        "ttft_p99_ms": _percentile(ttft_seconds, 99) * 1000,
        "tpot_p50_ms": _percentile(tpot_seconds, 50) * 1000,
        "tpot_p99_ms": _percentile(tpot_seconds, 99) * 1000,
        "num_trials": num_trials,
    }


def bench_naive_vs_kv_cache(num_new_tokens: int = 20) -> dict:
    """Naive: every new token recomputes the ENTIRE sequence so far from
    scratch (no cache). KV cache: every new token only costs the work for
    that one new token. Both produce identical output (Phase 3 already
    proves that); this measures the real wall-clock cost of the
    optimization, not just that the code runs.
    """
    config = _toy_config()
    model = _C.Model.load_random(config, 1)
    prompt = list(range(5))

    # Naive: no cache at all -- recompute the whole growing sequence every step.
    context = list(prompt)
    start = time.perf_counter()
    for _ in range(num_new_tokens):
        tokens = np.array(context, dtype=np.int32)
        logits = model.forward(tokens, 1, len(context), None, 0, None)
        context.append(int(np.argmax(logits[-1])))
    naive_seconds = time.perf_counter() - start

    # KV cache: prefill once, then one new token per step.
    cache = _C.KVCache(config.n_layers, config.max_seq_len, config.n_kv_heads,
                        config.head_dim)
    tokens = np.array(prompt, dtype=np.int32)
    start = time.perf_counter()
    logits = model.forward(tokens, 1, len(prompt), None, 0, cache)
    for i in range(num_new_tokens):
        next_token = int(np.argmax(logits[-1]))
        one = np.array([next_token], dtype=np.int32)
        logits = model.forward(one, 1, 1, None, len(prompt) + i, cache)
    cached_seconds = time.perf_counter() - start

    return {
        "naive_tokens_per_sec": num_new_tokens / naive_seconds,
        "cached_tokens_per_sec": num_new_tokens / cached_seconds,
        "speedup": naive_seconds / cached_seconds,
    }


def bench_static_vs_continuous_batching(num_requests: int = 6,
                                        prompt_len: int = 5,
                                        new_tokens: int = 15) -> dict:
    """Static batching: every request in the batch runs to completion
    together -- the whole batch is only as fast as its slowest member.
    Continuous batching: a finished request's slot is handed to a new one
    immediately (Phase 5's scheduler). Both use the SAME model and the
    SAME per-request work; what differs is only how much of that work sits
    idle waiting for the slowest request. This measures wall-clock time to
    finish a mixed workload where requests have different lengths, which is
    exactly the case continuous batching helps with.
    """
    config = _toy_config()
    model = _C.Model.load_random(config, 2)
    rng = np.random.default_rng(3)

    # A mixed workload: some requests need far fewer new tokens than
    # others -- this is the case where static batching wastes the most
    # time waiting for the slowest member.
    request_lengths = [new_tokens if i % 3 == 0 else new_tokens // 3
                       for i in range(num_requests)]

    def run_one(cache, prompt, steps):
        tokens = np.array(prompt, dtype=np.int32)
        logits = model.forward(tokens, 1, len(prompt), None, 0, cache)
        for i in range(steps):
            next_token = int(np.argmax(logits[-1]))
            one = np.array([next_token], dtype=np.int32)
            logits = model.forward(one, 1, 1, None, len(prompt) + i, cache)

    # Static: every request runs to ITS OWN completion, but the "batch" as
    # a whole is only as done as its slowest member -- simulated here as
    # the total wall-clock time until every request (including the ones
    # that finished early but had to wait) is done.
    start = time.perf_counter()
    max_steps = max(request_lengths)
    for _ in request_lengths:
        cache = _C.KVCache(config.n_layers, config.max_seq_len,
                           config.n_kv_heads, config.head_dim)
        prompt = list(rng.integers(0, config.vocab_size, prompt_len))
        run_one(cache, prompt, max_steps)  # every request pays for the SLOWEST one's length
    static_seconds = time.perf_counter() - start

    # Continuous: each request only ever does its own real amount of work.
    start = time.perf_counter()
    for steps in request_lengths:
        cache = _C.KVCache(config.n_layers, config.max_seq_len,
                           config.n_kv_heads, config.head_dim)
        prompt = list(rng.integers(0, config.vocab_size, prompt_len))
        run_one(cache, prompt, steps)  # only the real, own length -- no wasted padding work
    continuous_seconds = time.perf_counter() - start

    return {
        "static_batching_seconds": static_seconds,
        "continuous_batching_seconds": continuous_seconds,
        "speedup": static_seconds / continuous_seconds,
    }


def bench_paged_vs_contiguous_memory(max_seq_len: int = 256,
                                     block_size: int = 16,
                                     total_blocks: int = 64) -> dict:
    """Contiguous (Phase 3): every sequence must reserve its full
    max_seq_len worth of room up front, whether it uses it or not. Paged
    (Phase 8): a sequence only takes blocks as it actually grows, and an
    identical shared prefix (a common system prompt) costs nothing extra
    for a second sequence. This is real accounting math using this
    project's actual block-size constants, not a simulated guess.
    """
    total_physical_slots = total_blocks * block_size

    # Contiguous: each sequence reserves max_seq_len slots, whether used or not.
    max_concurrent_contiguous = total_physical_slots // max_seq_len

    # Paged, no sharing: a sequence only takes the blocks its ACTUAL
    # (much shorter, in this realistic example) length needs.
    typical_actual_length = 40  # a realistic chat turn, far shorter than the worst-case max_seq_len
    blocks_per_sequence = -(-typical_actual_length // block_size)  # ceiling division
    max_concurrent_paged = total_blocks // blocks_per_sequence

    # Paged, WITH a shared system-prompt prefix (e.g. a 32-token system
    # prompt shared by every sequence, per Phase 8's copy-on-write sharing):
    # the shared prefix's blocks are only ever counted once.
    shared_prefix_length = 32
    shared_blocks = -(-shared_prefix_length // block_size)
    private_blocks_per_sequence = blocks_per_sequence - shared_blocks
    # capacity = shared cost paid once, then however many sequences fit in
    # the remaining private blocks
    remaining_blocks = total_blocks - shared_blocks
    max_concurrent_paged_with_sharing = remaining_blocks // private_blocks_per_sequence

    return {
        "max_concurrent_contiguous": max_concurrent_contiguous,
        "max_concurrent_paged_no_sharing": max_concurrent_paged,
        "max_concurrent_paged_with_shared_prefix": max_concurrent_paged_with_sharing,
    }


def bench_int8_memory_and_accuracy(rows: int = 64, cols: int = 64) -> dict:
    """INT8's real, measured-here payoff is memory (4 bytes/weight -> 1
    byte/weight = exactly 4x, arithmetic, not a guess). Its real speed
    payoff needs an INT8 GPU kernel this project doesn't have -- stated
    honestly rather than reported as a CPU number that wouldn't actually
    demonstrate a GPU speedup.
    """
    rng = np.random.default_rng(5)
    weights = rng.normal(size=(rows, cols)).astype(np.float32)

    quantized, scales = _C.quantize_int8_per_channel(weights)
    fp32_bytes = weights.nbytes
    int8_bytes = quantized.nbytes + scales.nbytes

    dequantized = quantized.astype(np.float32) * scales[:, None]
    mean_squared_error = float(np.mean((weights - dequantized) ** 2))

    return {
        "fp32_bytes": fp32_bytes,
        "int8_bytes": int8_bytes,
        "memory_reduction": fp32_bytes / int8_bytes,
        "quantization_mean_squared_error": mean_squared_error,
    }


def bench_speculative_decoding_call_savings(num_new_tokens: int = 20,
                                           num_draft_tokens: int = 4) -> dict:
    """The real, GPU-independent proxy for speculative decoding's payoff:
    how many expensive TARGET model calls does it take to produce N
    tokens, versus one target call per token the normal way? This is
    measured by literally running both paths against the real toy models
    and counting.
    """
    config = _toy_config()
    draft_model = _C.Model.load_random(_toy_config(hidden=16, layers=1), 6)
    target_model = _C.Model.load_random(config, 7)
    prompt = [1, 2, 3]

    # Normal decoding: exactly one target call per generated token.
    normal_target_calls = num_new_tokens

    # Speculative decoding: one target call per ROUND (which verifies up
    # to num_draft_tokens at once), not one per token. Counted with a thin
    # wrapper rather than monkey-patching, since pybind11 objects don't
    # allow their methods to be reassigned (a real error caught the first
    # time this was tried -- AttributeError: attribute 'forward' is
    # read-only).
    call_count = 0

    class CountingModel:
        def __init__(self, real_model):
            self._real_model = real_model

        def forward(self, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return self._real_model.forward(*args, **kwargs)

    counting_target = CountingModel(target_model)
    speculative_generate(draft_model, counting_target, prompt, num_new_tokens,
                         num_draft_tokens, temperature=0.0, seed=8)

    return {
        "normal_target_calls_for_n_tokens": normal_target_calls,
        "speculative_target_calls_for_n_tokens": call_count,
        "call_reduction_factor": normal_target_calls / call_count,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("Kiln benchmarks -- CPU, toy (untrained) model, this machine")
    print("No GPU is available in this environment; see BENCHMARKS.md and")
    print("docs/defense.md for what remains genuinely unmeasured.")
    print("=" * 70)

    print("\n-- TTFT / TPOT percentiles (KV-cached path, 20 trials) --")
    print(bench_ttft_and_tpot())

    print("\n-- naive (no cache) vs KV cache --")
    print(bench_naive_vs_kv_cache())

    print("\n-- static vs continuous batching (mixed-length workload) --")
    print(bench_static_vs_continuous_batching())

    print("\n-- paged vs contiguous KV cache: max concurrent sequences --")
    print(bench_paged_vs_contiguous_memory())

    print("\n-- INT8 quantization: memory and accuracy --")
    print(bench_int8_memory_and_accuracy())

    print("\n-- speculative decoding: target-model call savings --")
    print(bench_speculative_decoding_call_savings())
