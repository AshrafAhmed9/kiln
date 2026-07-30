# ADR-007: Minimal code as a primary design objective

**Status:** Accepted (Revision 1.1)

**Decision:** Total engine code (`src/`) targets ~6–10k lines of C++, tracked
per-part in `BENCHMARKS.md`. Exceeding budget triggers a deletion pass before
merge, not a raised budget. When a feature and the budget conflict, the
feature is cut. When performance and explainability conflict, the more
legible implementation wins and its measured cost is published. Style rules:
C++17 only, no template metaprogramming/SFINAE/CRTP, one concept per file,
~400-line file ceiling, plain structs/free functions over class hierarchies,
every CUDA kernel keeps an adjacent pseudocode-like CPU reference.

**Why:** the whole project must be defended live in interviews — every line
is a line to defend and to have actually understood. Volume is a risk metric,
same status as the memory high-water mark. See project memory
`kiln-minimal-defensible-code`.
