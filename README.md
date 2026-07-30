# Kiln

A vLLM-class LLM serving engine, built hybrid the way real engines are
actually built: **Python orchestrates** (API, scheduler, control plane),
**C++/CUDA computes** (forward pass, kernels, KV memory) — verified at every
step against a HuggingFace reference by a numerical parity harness. Full
plan: [`../KILN PLAN.md`](../KILN%20PLAN.md).

Status: **Phase 0 in progress** (foundations — repo, CI, memory arena,
pybind11 boundary, parity oracle).

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
