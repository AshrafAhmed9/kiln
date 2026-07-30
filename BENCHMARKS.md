# Benchmarks

One row per phase: what changed, tokens/s, TTFT, p99 ITL, memory, and (for
lossy changes) perplexity delta. Also tracks the Phase 0b/ADR-007 code
budget per part, and (per ADR-006) the exact free-tier hardware/session used
for any GPU number, since no paid or persistent CI exists to pin this
automatically.

| Phase | Change | tok/s | TTFT | p99 ITL | Memory | Perplexity Δ | Lines (part total) | Hardware / seed |
|---|---|---|---|---|---|---|---|---|
| 0 | Foundations: arena allocator, CI, oracle | — | — | — | — | — | ~40 (Part I) | MacBook, N/A |
