# Kiln

A vLLM-class LLM serving engine built from scratch in C++/CUDA, verified at
every step against a HuggingFace reference by a numerical parity harness.
Full plan: [`../KILN PLAN.md`](../KILN%20PLAN.md).

Status: **Phase 0 in progress** (foundations — repo, CI, memory arena,
parity oracle).

## Layout

See `docs/walkthrough.md` for a literate tour, `docs/adr/` for design
decisions, `docs/defense.md` for interview-facing explanations, and
`docs/learning/` for the derivations and notes behind each phase.

## Build

```
cmake -B build -G Ninja
cmake --build build
ctest --test-dir build
```
