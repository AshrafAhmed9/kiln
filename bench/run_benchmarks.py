"""Real benchmarks, run on this machine's CPU against the toy (untrained,
randomly-initialized) model -- there is no GPU or real trained checkpoint
available in this environment. Every number
this script prints was actually measured by running the code, not
estimated. Where an optimization's real payoff only shows up on a GPU
(raw kernel throughput, INT8 speed), that's stated explicitly rather than
implied by a CPU number that wouldn't actually demonstrate it.

Usage: PYTHONPATH=. python3 bench/run_benchmarks.py
"""
from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from typing import Any

import numpy as np

from kiln_py import _C


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


def bench_naive_vs_kv_cache(num_new_tokens: int = 20) -> dict:
    """Naive: every new token recomputes the ENTIRE sequence so far from
    scratch (no cache). KV cache: every new token only costs the work for
    that one new token. Both produce identical output (the KV-cache tests already
    prove that); this measures the real wall-clock cost of the
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


SEED = 1234


def _percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def _generate_with_timing(model, config, prompt: list[int], num_new_tokens: int,
                          use_cache: bool) -> tuple[float, list[float]]:
    """Greedy-decodes and returns (time to first token, per-token gaps after it).

    The first token is timed from the call start to the first argmax, so it
    includes the whole prompt pass. Every later gap is the time to produce
    one more token. With use_cache=False this is the baseline: each step
    re-runs the entire sequence so far, so the gaps grow with context.
    """
    cache = (_C.KVCache(config.n_layers, config.max_seq_len, config.n_kv_heads,
                        config.head_dim) if use_cache else None)
    context = list(prompt)

    start = time.perf_counter()
    logits = model.forward(np.array(context, dtype=np.int32), 1, len(context),
                           None, 0, cache)
    next_token = int(np.argmax(logits[-1]))
    ttft = time.perf_counter() - start

    gaps = []
    for step in range(num_new_tokens - 1):
        context.append(next_token)
        start = time.perf_counter()
        if use_cache:
            logits = model.forward(np.array([next_token], dtype=np.int32), 1, 1,
                                   None, len(context) - 1, cache)
        else:
            logits = model.forward(np.array(context, dtype=np.int32), 1,
                                   len(context), None, 0, None)
        next_token = int(np.argmax(logits[-1]))
        gaps.append(time.perf_counter() - start)
    return ttft, gaps


def bench_latency(num_trials: int = 20, prompt_len: int = 32,
                  num_new_tokens: int = 33, warmup: int = 3) -> dict:
    """Time to first token (TTFT) and inter-token latency (ITL), p50 and p99,
    for the KV-cached path and for the no-cache baseline. Same model, same
    seeded prompts, same token counts, so the two columns differ only in
    whether the cache is used. One warmup round is dropped so first-call
    costs don't land in the percentiles.
    """
    config = _toy_config()
    model = _C.Model.load_random(config, SEED)
    rng = np.random.default_rng(SEED)
    prompts = [[int(t) for t in rng.integers(0, config.vocab_size, prompt_len)]
               for _ in range(warmup + num_trials)]

    out = {"prompt_len": prompt_len, "new_tokens": num_new_tokens,
           "trials": num_trials}
    for name, use_cache in (("kv_cache", True), ("no_cache_baseline", False)):
        ttfts, gaps = [], []
        for i, prompt in enumerate(prompts):
            ttft, g = _generate_with_timing(model, config, prompt,
                                            num_new_tokens, use_cache)
            if i >= warmup:
                ttfts.append(ttft)
                gaps.extend(g)
        out[name] = {
            "ttft_p50_ms": _percentile(ttfts, 50) * 1000,
            "ttft_p99_ms": _percentile(ttfts, 99) * 1000,
            "itl_p50_ms": _percentile(gaps, 50) * 1000,
            "itl_p99_ms": _percentile(gaps, 99) * 1000,
        }
    return out


def _peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux reports kilobytes.
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def bench_memory(max_seq_len: int = 256) -> dict:
    """KV cache memory is exact arithmetic from the model shape (FP32 K and V
    for every layer), so it is reported per token and per full-length
    sequence. Peak process memory is measured, before and after a full
    cached generation, to show what the cache costs in practice.
    """
    config = _toy_config(max_seq_len=max_seq_len)
    bytes_per_token = 2 * config.n_layers * config.n_kv_heads * config.head_dim * 4
    rss_before = _peak_rss_mb()

    model = _C.Model.load_random(config, SEED)
    rng = np.random.default_rng(SEED)
    prompt = [int(t) for t in rng.integers(0, config.vocab_size, 32)]
    _generate_with_timing(model, config, prompt, max_seq_len - len(prompt), True)

    return {
        "kv_bytes_per_token": bytes_per_token,
        "kv_bytes_per_full_sequence": bytes_per_token * max_seq_len,
        "peak_rss_mb_before": rss_before,
        "peak_rss_mb_after_full_generation": _peak_rss_mb(),
    }


CONCURRENCY_LIMITS = [
    "The API serves one request at a time; concurrent requests queue behind a lock.",
    "There is no batching across requests, so throughput does not rise with load.",
    "Latency numbers above are single-request numbers on CPU with a small random model.",
    "The GPU path owns one device-resident KV cache, which is why it is single-request.",
]


def _environment() -> dict:
    return {"python": platform.python_version(), "machine": platform.machine(),
            "system": platform.platform(), "processor": platform.processor(),
            "seed": SEED}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="also write all results to this file")
    args = parser.parse_args()

    results: dict[str, Any] = {"environment": _environment()}
    print("=" * 70)
    print("Kiln benchmarks -- CPU, toy (untrained) model, fixed seed", SEED)
    print("=" * 70)

    print("\n-- TTFT / inter-token latency: KV cache vs no-cache baseline --")
    results["latency"] = bench_latency()
    print(json.dumps(results["latency"], indent=2))

    print("\n-- naive (no cache) vs KV cache: throughput --")
    results["naive_vs_kv_cache"] = bench_naive_vs_kv_cache()
    print(results["naive_vs_kv_cache"])

    print("\n-- memory: KV cache size and peak process memory --")
    results["memory"] = bench_memory()
    print(results["memory"])

    print("\n-- INT8 quantization: memory and accuracy --")
    results["int8"] = bench_int8_memory_and_accuracy()
    print(results["int8"])

    results["concurrency_limits"] = CONCURRENCY_LIMITS
    print("\n-- concurrency limits --")
    for line in CONCURRENCY_LIMITS:
        print(" *", line)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
