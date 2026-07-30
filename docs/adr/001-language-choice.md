# ADR-001: C++ as the compute language

**Status:** Superseded in part by ADR-006 (hybrid architecture) — kept for
history.

**Original decision (Phase 0):** the engine is C++17.

**Amendment (Revision 1.1, hybrid edition):** C++/CUDA remains the compute
language (forward pass, kernels, KV memory, quantizer) — that part of this
ADR stands. Orchestration (API, scheduler, control plane) moved to Python;
see ADR-006 for the full reasoning and the boundary contract.

**Why C++/CUDA for compute specifically:** a preferred qualification for
ML-infra roles, and it's the language the production engines this project is
modeled on (vLLM's kernels, TensorRT-LLM) are actually written in. A tensor
framework or fully-managed runtime would hide exactly the memory management
and kernel dispatch the project exists to teach.
