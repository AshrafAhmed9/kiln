# Benchmarks

One row per phase: what changed, tokens/s, TTFT, p99 ITL, memory, and (for
lossy changes) perplexity delta. Also tracks the ADR-010 code budget per
part (`csrc/` and `kiln_py/` counted separately), and (per ADR-009) the
exact free-tier hardware/session used for any GPU number, since no paid or
persistent CI exists to pin this automatically. Phase 4 adds the §6
boundary-cost measurement (Python-orchestration-overhead vs C++-executor
time per decode iteration) as its own tracked row.

| Phase | Change | tok/s | TTFT | p99 ITL | Memory | Perplexity Δ | Lines (csrc / kiln_py) | Hardware / seed |
|---|---|---|---|---|---|---|---|---|
| 0 | Foundations: arena allocator, pybind11 boundary (`ping`), CI, oracle | — | — | — | — | — | ~40 / ~10 | MacBook, N/A |
