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

Test code (exempt from the budget per ADR-010): 817 lines across `tests/cpp` and `tests/py` (29 C++ tests, 13 Python tests — all passing, including under ASan/UBSan).

**Not yet measured, stated honestly:** real tokens/s, TTFT, and p99 ITL against a genuine trained checkpoint on real hardware — this session used only small, randomly-initialized weights (no real HF download was available offline; see ADR-009). Those numbers are deferred to when a real checkpoint is served, per the zero-budget plan.
