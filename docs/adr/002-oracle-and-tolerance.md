# ADR-002: The parity oracle and tolerance policy

**Status:** Accepted (tolerances to be finalized empirically in Phase 2)

**Decision:** HuggingFace `transformers`, run in FP32, is the ground truth.
`tools/oracle.py` dumps per-layer activations and final logits for a fixed
fixture set of prompts to a `.pt` file; the C++ engine loads and diffs
against it.

- **Exact paths** (FP32 forward pass, KV cache, batching, seeded sampling):
  parity means both `max_abs_diff` under a documented per-dtype threshold
  *and* top-k token-order equality. Thresholds are set per phase once real
  numbers are observed — not guessed in advance — and recorded here as an ADR
  amendment, never as an inline magic constant in test code.
- **Lossy paths** (quantization, speculative-decode drafts): parity is
  replaced by a measured budget (WikiText-2 perplexity delta, per-token KL
  divergence vs FP16), committed to `BENCHMARKS.md`. A scheme exceeding its
  budget fails CI.
- **Determinism:** given (weights, prompt, seed, config), output must be
  byte-identical across runs.

**Why:** "does it feel right" is not falsifiable; a saved reference tensor and
a documented tolerance is. This is the project's entire differentiator and
cannot be retrofitted — the oracle's activation-dump shape drives the
runtime's internal interfaces from Phase 1 onward.
