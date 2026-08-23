# ADR-013: JSON parsing on the C++ side

**Status:** Accepted (Phase 1)

**Gap found:** the plan's hybrid-edition dependency table (constitution §2)
lists no JSON library for C++, having moved the API layer (which used
`nlohmann/json` for payloads in plan v1.0) to Python/FastAPI. But two Phase 1
inputs are JSON regardless of API language: the safetensors header (a JSON
blob prefixed by an 8-byte length) and `tokenizer.json` (vocab + merges).
Something in `csrc/loader` and `csrc/tokenizer` has to parse them.

**Decision:** add `nlohmann/json` (single header) back to the C++ allowlist,
scoped to `csrc/loader` and `csrc/tokenizer` only.

**Why not hand-built:** identical reasoning to why HTTP parsing isn't
hand-built — a JSON parser teaches nothing about ML systems and is pure
defense-surface cost for zero learning payoff (ADR-010's own test).
