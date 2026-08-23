# Kiln

A vLLM-class LLM serving engine, built hybrid the way real engines are
actually built: **Python orchestrates** (API, scheduler, control plane),
**C++/CUDA computes** (forward pass, kernels, KV memory) — verified at every
step against a HuggingFace reference by a numerical parity harness. Full
plan: [`../KILN PLAN.md`](../KILN%20PLAN.md).

Status: **Part I (Phases 0–6) complete; Part II (Phases 7–12) complete
wherever CPU-testable.** Part I is a CPU-only, correct continuous-batching
LLM server: safetensors loader, byte-level BPE tokenizer, a full
Llama-architecture forward pass (GEMM/RMSNorm/RoPE/GQA attention/SwiGLU), a
contiguous KV cache with seeded sampling, padded static batching, an
Orca-style continuous-batching scheduler, and an OpenAI-compatible API with
SSE streaming. Part II adds a paged, copy-on-write KV cache; INT8/INT4
quantization (cross-checked against an independent Python reference);
speculative decoding (**proven token-for-token exact against direct greedy
decoding**, its strongest correctness result); a self-test proving the
parity methodology actually catches a real class of bug; and tensor-parallel
sharding math proven correct via CPU simulation. 60/60 tests pass (40 C++
under ASan/UBSan, 20 Python).

**Honestly incomplete, on purpose:** this machine has no NVIDIA GPU, so
Phase 7's CUDA/Triton kernels are written to spec but **not compiled or
run** in this session — real GPU work (kernel benchmarks, real
quantization accuracy/latency tables, real speculative-decoding speedups,
real multi-GPU tensor-parallel scaling) is deferred to the Kaggle T4 path
the plan already budgets for (ADR-009). No real trained checkpoint has been
run through any of this code either (no HF install offline), so there is
no verified numerical parity against the HuggingFace reference yet. See
`docs/walkthrough.md`'s "Honest state" section and `BENCHMARKS.md` for the
full, itemized list of what's proven vs. deferred.

## Layout

```
csrc/      C++/CUDA compute layer (everything below the pybind11 boundary)
kiln_py/   Python orchestration layer (API, scheduler, runtime driver)
tests/     cpp/ (GoogleTest), py/ (pytest), parity/ (oracle-diff suites)
tools/     oracle.py (HF reference dumps), quantize_ref.py (Phase 9)
docs/      adr/, walkthrough.md, defense.md, correctness.md, learning/
```

See `docs/walkthrough.md` for a literate tour, `docs/adr/` for design
decisions (ADR-006 is the language boundary, ADR-007 the kernel strategy),
`docs/defense.md` for interview-facing explanations, and `docs/learning/`
for the derivations and notes behind each phase.

## Build

```
pip install -e .              # builds csrc -> kiln_py/_C via scikit-build-core
cmake -B build -G Ninja        # or build/test the C++ side directly
cmake --build build
ctest --test-dir build --output-on-failure
```
