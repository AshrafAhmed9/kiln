# Phase 28 — the tooling was already right; nobody had pointed it at more than one thing

## What was actually missing

`tools/hf_parity.py` already supported a `--prompts-file` flag and per-layer
hidden-state diffing. `tools/fixtures/hf_parity_prompts.txt` already had 10
real prompts covering short sequences, a 16-token sequence, Unicode text
(Latin, CJK, Arabic, emoji), and punctuation. None of that was new code. What
had never happened was actually running the fixture -- every number in the
repo's docs up to this point traced back to one hardcoded prompt against one
checkpoint.

## Why one checkpoint isn't a claim about checkpoints in general

A model loader that works for one specific set of `hidden_size`,
`n_layers`, `n_heads`, `n_kv_heads` values could still be silently wrong for
a different shape -- an off-by-one in how grouped-query attention repeats KV
heads across query heads wouldn't necessarily show up if every test run so
far used the same head ratio. SmolLM2-135M-Instruct and
SmolLM2-360M-Instruct are both real Llama-family checkpoints but have
different GQA ratios (135M's config vs. 360M's 15 query / 5 KV heads) and a
different depth (fewer vs. 32 layers). Running the identical fixture against
both, with zero changes to `Model::load_from_safetensors` or the executor,
is a stronger claim than running it once: the loader and forward pass
generalize across at least two real, differently-shaped checkpoints, not
just the one that happened to get tested first.

## The one real snag: `sys.path`, not the model code

Running `python3 tools/hf_parity.py` from the repo root fails to import
`kiln_py` -- Python puts the *script's own directory* (`tools/`) at the front
of `sys.path`, not the current working directory, so the repo-root `kiln_py`
package isn't visible. `PYTHONPATH=.` fixes it. This isn't a bug in the tool;
it's a standard Python footgun worth naming so the next person running this
script doesn't waste time on a `ModuleNotFoundError` that has nothing to do
with the actual code being tested.

## The result

Both checkpoints: 10/10 fixture prompts top-1 match, final-logit max absolute
differences in the 10⁻⁵ range on both (consistent with FP32-vs-BF16-loaded
rounding, not a real numerical divergence), and 10,000/10,000 seeded
tokenizer matches on the 360M checkpoint's own tokenizer JSON (the 135M
checkpoint's tokenizer was already verified in Phase 22). No bug found --
this phase's contribution is evidence, not a fix.
