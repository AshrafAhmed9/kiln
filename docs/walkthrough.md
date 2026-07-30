# Walkthrough

A literate, top-to-bottom read of one full forward pass through Kiln,
updated after every phase. If a section is hard to write plainly, the
underlying code is too clever and should be simplified first (ADR-010).

## Phase 0 — foundations

There is no forward pass yet. What exists:

- `csrc/memory/arena.{h,cpp}` — a bump allocator: one preallocated block,
  sequential slices handed out, reclaimed all at once via `Reset()`. Every
  later allocator in Kiln (including the Phase 8 paged KV cache) is a
  variant of this same idea — fixed capacity, explicit accounting, no
  per-object free.
- `csrc/bindings.cpp` — the pybind11 boundary (constitution §6). Phase 0
  exposes a single `ping()` returning `"pong"`, proving the build/packaging
  path (`pip install -e .` -> `csrc/` compiles -> `kiln_py._C` importable)
  works end-to-end before any real interface depends on it.
- `tools/oracle.py` — runs the reference HF model in FP32, hooks every
  decoder layer to record its output, and saves input ids + activations +
  logits to a `.pt` fixture. Phase 1–2 C++ code will load this fixture and
  diff against it layer by layer.
