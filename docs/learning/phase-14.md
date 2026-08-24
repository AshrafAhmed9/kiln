# Phase 14 — derivation notes (adapter-aware serving; the fine-tune loop itself is out of scope here)

## What a LoRA adapter actually is

Fully retraining every number in a large weight matrix is expensive. LoRA
("low-rank adaptation") instead trains two much smaller matrices, A and B,
and represents the *change* to a weight matrix as their product: `delta =
scale * (B @ A)`. If the original weight is `[out_features, in_features]`,
A is `[rank, in_features]` and B is `[out_features, rank]`, where `rank` is
a small number (far smaller than either dimension) -- so the two small
matrices together have vastly fewer numbers to train than the original
weight matrix would. The adapted weight the model actually uses is simply
`original_weight + delta`.

## Why merging happens once, at load time, rather than every forward pass

There are two ways to serve an adapted model: keep the adapter's small
matrices separate and add their contribution in every forward pass, or
compute `original_weight + delta` once, up front, and serve that combined
matrix as if it always looked that way. This project does the second: it's
simpler code, and it means the served model runs through the exact same,
already-tested forward pass as an unmodified model -- there's no separate
"adapter-aware" code path in the executor to build and verify all over
again. The cost is that switching adapters means recomputing the merge (a
cheap, one-time matrix multiply), rather than swapping a small matrix on
the fly -- a reasonable tradeoff for a serving engine that doesn't expect
to hot-swap adapters every request.

## Honest, explicit scope boundary

This phase implements and tests **adapter merging and serving only** --
given an already-trained LoRA adapter's A and B matrices, correctly fold
them into a served model's weights. It does **not** implement:

- the actual training loop that produces A and B from real data (needs
  PyTorch and a real training setup -- explicitly permitted as tooling by
  the constitution, but not built in this session),
- the data pipeline (dedup, filtering, tokenization throughput
  measurement),
- the multi-GPU DDP→FSDP scaling study the plan's Phase 14 centers on
  (needs multiple real GPUs, which this offline, single-machine session
  doesn't have -- Phase 12's tensor-parallel math is the closest CPU-provable
  analog already built, but it isn't a substitute for measuring real
  training-time scaling),
- gating an adapter's promotion through Phase 13's eval infrastructure
  with a real trained adapter (the wiring exists -- Phase 13's regression
  gate takes any two paired score lists -- but no real adapter exists yet
  to actually promote).

These are named here as real, deferred work, not implied to be finished.

## Reproducible training route

`tools/train_lora.py` now provides the training-tooling half: it trains a
standard PEFT `q_proj` LoRA adapter and writes a `kiln-export.json` manifest
whose A/B orientation and scale match Kiln's `wq` merge contract. A 20-step
SmolLM2 run over the tiny included fixture produced a 1.1 MB safetensors
adapter. `tools/merge_lora_adapter.py` then loaded that adapter through the
C++ binding into a real Kiln SmolLM2 model; its final logits changed by a
maximum of 0.1098 for a three-token probe. That proves the
training/export/serving route, not model quality: the four authored records
are intentionally too small for any usefulness claim.
