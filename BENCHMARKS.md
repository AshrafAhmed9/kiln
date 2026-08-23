# Benchmarks

One row per phase: what changed, tokens/s, TTFT, p99 ITL, memory, and (for
lossy changes) perplexity delta. Also tracks the ADR-010 code budget per
part (`csrc/` and `kiln_py/` counted separately), and (per ADR-009) the
exact free-tier hardware/session used for any GPU number, since no paid or
persistent CI exists to pin this automatically. Phase 4 adds the §6
boundary-cost measurement (Python-orchestration-overhead vs C++-executor
time per decode iteration) as its own tracked row.

| Phase | Change | tok/s | TTFT | p99 ITL | Memory | Perplexity Δ | Lines (csrc / kiln_py, non-test) | Hardware / seed |
|---|---|---|---|---|---|---|---|---|
| 0 | Foundations: arena allocator, pybind11 boundary (`ping`), CI, oracle | — | — | — | — | — | ~40 / ~10 | MacBook, N/A |
| 1 | Safetensors loader (mmap, zero-copy views) + byte-level BPE tokenizer | — | — | — | — | — | running total below | MacBook, N/A |
| 2 | CPU forward pass: GEMM, RMSNorm, RoPE, GQA attention, SwiGLU, full model | — | — | — | — | — | running total below | MacBook, N/A |
| 3 | Contiguous KV cache + seeded sampler (greedy/temp/top-k/top-p/rep-penalty) | — | — | — | — | — | running total below | MacBook, N/A |
| 4 | Padded static batching + **§6 boundary-cost measurement** (real number, not yet the production checkpoint): on a toy config (vocab 1000, hidden 64, 4 layers, CPU, `python3 -c` timing script) a bare Python→C++ round trip (`_C.ping()`) costs ~0.18µs; one real one-token decode step (KV-cached forward through all 4 layers) costs ~331.6µs. Python/pybind crossing overhead is **~0.06%** of one decode step here. This is a small-model CPU proxy for the plan's real claim, not the final measured number — that comes once a real checkpoint is served; recorded honestly as a directional result, not the finished benchmark. | — | — | — | — | — | **1,559 / — ** | MacBook, seed 1 |
| 5 | Continuous-batching scheduler (Orca-style admission/preemption), pure Python, tested against a mock executor with seeded random arrivals | — | — | — | — | — | — / running total below | MacBook, seeds 1-5 |
| 6 | OpenAI-compatible API (`/v1/completions`, SSE streaming), Part I assembly | — | — | — | — | — | **1,559 / 430** (of a 6–10k `csrc` and its own `kiln_py` budget — ADR-010) | MacBook, N/A |
| 7 | CUDA port: hand-written attention/RMSNorm/argmax kernels, RoPE both raw-CUDA and Triton (ADR-007). **UNVERIFIED** — no NVIDIA GPU in this session (see docs/defense.md, docs/learning/phase-07.md); not compiled, no benchmark possible yet. | n/a | n/a | n/a | n/a | n/a | 322 (`csrc/kernels/`, not yet buildable here) | none — deferred to Kaggle T4 |
| 8 | Paged KV cache: block-table allocator with copy-on-write prefix sharing. CPU-only, fully tested (allocator property tests, copy-on-write correctness, paged-vs-contiguous parity). | — | — | — | — | — | running total below | MacBook, seed 99 |
| 9 | INT8 (per-channel) + INT4 (group-wise, packed) quantization, cross-checked against an independent Python reference through the pybind11 boundary. Real deliverable (WikiText-2 perplexity/KL) deferred — no real checkpoint/dataset offline; synthetic quantized-matmul MSE proxy used instead. | — | — | — | — | — | running total below | MacBook, seed 7 |
| 10 | Speculative decoding (rejection sampling). **Proven exact in greedy mode**: token-for-token identical to direct target-only greedy decoding, across 3 draft lengths. Random-sampling mode runs correctly but its exact distribution-preservation is not statistically proven in this session (see docs/defense.md). No KV cache (full recompute per round) — a named simplification, not a hidden one. | — | — | — | — | — | running total below | MacBook, seeds 1/2/5/6/9/10 |
| 11 | Parity-harness self-test: a deliberately broken RMSNorm (missing mean division) is proven to fail the project's real tolerance check. Nightly GPU perf-regression tracking remains impossible free (ADR-009), unchanged from Part I. | — | — | — | — | — | — | MacBook, N/A |
| 12 | Tensor-parallel sharding math (column-parallel, row-parallel, and the paired column→row pattern), CPU-simulated in numpy, proven exact for 1/2/4 simulated ranks. Proves the algorithm only — no real NCCL/multi-GPU communication was exercised (see docs/defense.md). | — | — | — | — | — | **2,059 / 614** (`csrc` excludes the 322 unverified `csrc/kernels/` lines above) | MacBook, seeds 1/2/3 |

Test code (exempt from the budget per ADR-010): 1,381 lines across `tests/cpp` and `tests/py` (40 C++ tests, 20 Python tests — all passing, including under ASan/UBSan).

**Not yet measured, stated honestly:** real tokens/s, TTFT, and p99 ITL against a genuine trained checkpoint on real hardware — this session used only small, randomly-initialized weights (no real HF download was available offline; see ADR-009). All GPU-dependent numbers (Phase 7 kernel performance, Phase 9's real accuracy/latency/VRAM tradeoff table, Phase 10's real decode speedup, Phase 12's real scaling efficiency and communication overhead) are deferred to when Kaggle GPU time is used, per the zero-budget plan. What's measured here instead is CPU-only correctness — genuinely verified, not a substitute for the GPU numbers, just what was actually achievable offline.
