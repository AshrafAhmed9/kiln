# ADR-001: C++ as the engine language

**Status:** Accepted

**Decision:** The engine (everything under `src/`) is C++17. Python is
tooling-only (`tools/oracle.py`).

**Why:** C++ is a named or implied qualification for nearly every ML-infra /
inference-serving role this project targets, and it's the language the
production engines it's modeled on (vLLM's kernels, TensorRT-LLM) are written
in. Writing the scheduler and memory management in Python would hide the
exact machinery (allocation, layout, dispatch) the project exists to teach.

**Alternatives considered:** Rust (stronger memory-safety story, but far less
of a hiring signal for this target and a smaller CUDA ecosystem); pure Python
+ PyTorch (rejected outright — see the constitution's no-PyTorch rule, ADR to
follow on that specifically).
