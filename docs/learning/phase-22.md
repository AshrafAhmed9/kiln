# Phase 22 — first real reference comparison

## Why the model choice matters

`HuggingFaceTB/SmolLM2-135M-Instruct` is small enough to run on this CPU but
has the Llama architecture Kiln implements: RMSNorm, RoPE, grouped-query
attention, SwiGLU, and Llama-style tensor names. That makes a comparison
meaningful. Choosing an arbitrary small model with a different architecture
would only measure incompatibility, not whether Kiln's implementation agrees
with its intended reference.

## The measurement

Hugging Face ran the checkpoint in FP32. Kiln read its BF16 safetensors file,
converted each tensor to FP32, and received the same eight input token IDs for
`Kiln checks its own math.`. The final next-token logit vectors differed by
7.44×10⁻⁵ at their largest element and 1.23×10⁻⁵ on average. Their highest
scoring token was the same.

This is a strong smoke test because an incorrect projection, RoPE position,
attention mask, or tied output head usually changes thousands of logits and
often changes the best token. It is not a parity proof: one final vector can
miss a layer-local error that later happens to cancel, and one prompt cannot
cover sequence lengths, characters, or model states broadly.

The reproducible fixture now contains ten prompts spanning ordinary prose,
numbers, punctuation, Unicode, and a longer 16-token sequence. All ten match
the Hugging Face top token; their largest final-logit difference is
7.44×10⁻⁵. `tools/hf_parity.py --prompts-file tools/fixtures/hf_parity_prompts.txt`
loads the reference once and emits one result per prompt.

Kiln now exposes a debug-only post-layer hidden-state capture path. On the
original eight-token prompt, all 30 layers were compared with Hugging Face;
the largest hidden-state absolute difference was 0.02246. That is materially
larger than the final-logit difference because both implementations start from
BF16 checkpoint tensors but execute independently in FP32; it is recorded as
the observed result, not presented as exact layer parity.

## The tokenizer result was useful precisely because it failed

The first 10,000-string real fixture found a leading-space bug: the generic
whitespace branch ran before the GPT-2-style "space plus word" branch, making
that branch unreachable. Fixing the order increased exact matches from 4,439
to 4,749. The complete fix uses ICU's Unicode regex engine for the same
letter, number, punctuation, and whitespace boundaries as ByteLevel
pre-tokenization. The fixture now matches **10,000/10,000** strings. The BPE
implementation remains Kiln's; ICU only chooses the boundaries (ADR-014).
