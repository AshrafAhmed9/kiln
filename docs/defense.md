# Defense — interview-facing explanations, per phase

Written from memory before re-reading the code (ADR-011). One page per
phase: what the component is, why it works, what it cost.

## Phase 0 — Foundations & the oracle

**What:** an `Arena`, a bump allocator over one `std::vector<std::byte>`.
`Allocate(n)` hands back a pointer into the block and advances an offset;
`Reset()` sets the offset back to zero. `csrc/bindings.cpp` is a `pybind11`
module exposing one function, `ping()`, that proves the Python↔C++ boundary
(constitution §6) actually builds and imports before anything real crosses
it. `tools/oracle.py` loads the reference model, hooks each decoder layer
with `register_forward_hook`, runs one forward pass, and saves input ids,
every layer's output tensor, and the final logits to disk.

**Why it works:** the arena works because allocation lifetime in this
project is phase-shaped, not object-shaped — a batch of work is done and then
thrown away as a unit, so tracking "how far into the block have I gotten" is
enough; there's no need for a general allocator that supports arbitrary
free(). `pybind11` works by generating C++ glue that wraps a C++ function
pointer as a CPython callable, handling argument marshalling and reference
counting at the boundary — that glue is exactly why constitution §6 insists
on a single, narrow, audited boundary file rather than binding scattered
throughout the codebase. The oracle works because forward hooks are the
standard PyTorch mechanism for observing intermediate values without
modifying the model's source — they run after a module's forward() and
receive its output.

**What it cost:** a debug-only assert on arena overflow (checked separately
from the release-path nullptr return) — deliberate: a debug build should
crash loudly at the call site of a bug, but a release build shouldn't behave
differently under `NDEBUG`, so the public contract is "returns nullptr on
overflow" and the assert is a bonus diagnostic, not the contract itself. The
pybind11 dependency itself is the one boundary-crossing cost accepted by
ADR-006 — everything on either side of it stays hand-built.
