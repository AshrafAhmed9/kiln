# ADR-006: The language-boundary rule (hybrid by design)

**Status:** Accepted (Revision 1.1, hybrid edition)

**Decision:** Python orchestrates (API, scheduler, admission/preemption,
control plane); C++/CUDA computes (model executor/forward pass, KV-cache
manager, memory arenas, all kernels). The boundary is one narrow interface:
Python hands the executor a *batch descriptor* (sequence IDs, block tables,
token counts, sampling params) via `pybind11`; C++ returns logits into a
preallocated buffer. One crossing per decode iteration — never per token per
sequence.

**Why:**
1. PyTorch, vLLM, and TensorRT-LLM all put Python at the front and C++/CUDA
   underneath — this is architecturally correct, not a concession.
2. ~99% of wall-clock is inside the kernels regardless of orchestration
   language. Phase 4 measures the actual per-iteration Python overhead and
   commits the number to `BENCHMARKS.md`, so "hybrid costs nothing that
   matters" is a measured claim, not an assertion. Honest limit: at very high
   request rates on very small models this overhead stops being noise — the
   crossover point gets measured and documented, not hidden.
3. Defensibility: hybrid concentrates the must-defend-cold surface on
   kernels, memory, and scheduling policy — the parts worth being asked
   about — instead of a hand-rolled C++ HTTP stack that teaches nothing but
   still has to be explained under questioning.

**Cost, stated plainly:** the resume claim narrows from "100% C++" to "the
compute layer — kernels, executor, memory manager — in C++/CUDA." Narrower,
but sharper and backed by a measured number instead of a shrug.

**Contract:** fixed at Phase 2, never widened casually — widening the
Python↔C++ interface requires a new ADR. Full detail in `KILN PLAN.md`
constitution §6.
