# ADR-010: Minimal code as a primary design objective

**Status:** Accepted (Revision 1.1)

**Decision:** Total compute-layer code (`csrc/`) targets ~6–10k lines of
C++/CUDA; the Python orchestration layer (`kiln_py/`) gets its own, smaller
budget since policy code is naturally more compact — tracked separately per
part in `BENCHMARKS.md`. Exceeding either budget triggers a deletion pass
before merge, not a raised budget. When a feature and the budget conflict,
the feature is cut. When performance and explainability conflict, the more
legible implementation wins and its measured cost is published. C++ style
rules: C++17 only, no template metaprogramming/SFINAE/CRTP, one concept per
file, ~400-line file ceiling, plain structs/free functions over class
hierarchies, every CUDA kernel keeps an adjacent pseudocode-like CPU
reference. Python style: no cleverness that needs a comment to parse —
scheduler/API code is read by the same "one sitting" standard.

**Why:** the whole project must be defended live in interviews — every line
is a line to defend and to have actually understood. Volume is a risk metric,
same status as the memory high-water mark. See project memory
`kiln-minimal-defensible-code`.
