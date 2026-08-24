"""Time the Triton side of ADR-007's raw-CUDA-versus-Triton RoPE comparison.

Run this on a CUDA host after building and running `kiln_rope_cuda_benchmark`
with the same four dimensions. CUDA events time device work only, excluding
Python dispatch, allocation, and setup from both implementations.
"""
from __future__ import annotations

import argparse
import statistics

import torch

from csrc.kernels.triton.rope import apply_rope_triton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--heads", type=int, default=9)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=1_000)
    args = parser.parse_args()
    if min(args.tokens, args.heads, args.head_dim, args.iterations) <= 0:
        parser.error("all dimensions and iterations must be positive")
    if args.head_dim % 2:
        parser.error("--head-dim must be even")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires a CUDA-capable PyTorch install")
    x = torch.zeros(args.tokens * args.heads * args.head_dim,
                    device="cuda", dtype=torch.float32)
    positions = torch.arange(args.tokens, device="cuda", dtype=torch.int64)
    for _ in range(100):
        apply_rope_triton(x, positions, args.tokens, args.heads, args.head_dim,
                          10000.0)
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(21):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.iterations):
            apply_rope_triton(x, positions, args.tokens, args.heads,
                              args.head_dim, 10000.0)
        end.record()
        end.synchronize()
        samples_ms.append(start.elapsed_time(end) / args.iterations)
    median_ms = statistics.median(samples_ms)
    elements = args.tokens * args.heads * args.head_dim
    effective_gb_s = (elements * 2 * 4) / (median_ms * 1e6)
    print(
        "implementation=triton"
        f" tokens={args.tokens} heads={args.heads} head_dim={args.head_dim}"
        f" iterations={args.iterations} median_ms={median_ms:.6f}"
        f" effective_gb_s={effective_gb_s:.6f}"
    )


if __name__ == "__main__":
    main()
