# Walkthrough

A literate, top-to-bottom read of one full forward pass through Kiln,
updated after every phase. If a section is hard to write plainly, the
underlying code is too clever and should be simplified first (ADR-010).

## The whole path, end to end (through Phase 6)

1. **A sentence comes in as text.** `csrc/tokenizer/bpe.cpp` turns it into
   a list of numbers (token ids), using the merge rules from a
   `tokenizer.json` file.
2. **Those numbers become vectors.** `Model::Forward`
   (`csrc/executor/model.cpp`) looks up each token's row in the embedding
   table.
3. **Each of the model's layers runs in turn:**
   - `RmsNorm` (`csrc/executor/rmsnorm.cpp`) rescales the numbers.
   - `ApplyRope` (`csrc/executor/rope.cpp`) rotates the query and key
     numbers so the model can tell how far apart two words are.
   - `Attention` (`csrc/executor/attention.cpp`) lets each word look back
     at every earlier word (never a later one) and blend their information
     together.
   - The attention result is added back onto the running total (a
     residual connection), then normalized again and passed through
     `SwiGlu` (`csrc/executor/swiglu.cpp`), a small "think about it"
     network, whose result is also added back on.
4. **After every layer, one final normalize and one last matrix multiply**
   (against the `lm_head` weights) produces one score per possible next
   word (the logits).
5. **A word is chosen** from those scores by `Sample` or `GreedyArgmax`
   (`csrc/executor/sampler.cpp`).
6. **If there's a `KVCache`** (`csrc/kv/kv_cache.cpp`), every layer's key
   and value numbers for the new word get written into it, so the *next*
   word only has to repeat steps 2-6 for that one new word, reusing
   everything already cached from before.
7. **The chosen word's number is turned back into text** by the tokenizer,
   and either returned all at once (`kiln_py/runtime/generate.py`) or sent
   to the client immediately, one word at a time
   (`kiln_py/api/app.py`'s streaming endpoint).
8. **Above all of this, in Python**, `kiln_py/scheduler/scheduler.py`
   decides which requests are allowed to be running at all, admitting new
   ones the moment there's enough reserved room and never letting more
   requests run at once than the memory budget allows.

## What exists so far, file by file

- `csrc/memory/arena.{h,cpp}` — a bump allocator: one preallocated block,
  sequential slices handed out, reclaimed all at once via `Reset()`.
- `csrc/bindings.cpp` — the one file allowed to know about both Python and
  C++ (constitution §6): it wraps `Model`, `KVCache`, the sampler, and the
  tokenizer so Python can call them.
- `csrc/loader/safetensors.{h,cpp}` and `dtype.h` — opens a weight file via
  memory-mapping (no copying) and converts on-disk BF16/FP16 numbers to
  FP32 for the CPU path.
- `csrc/tokenizer/bpe.{h,cpp}` — byte-level BPE, loaded from a
  `tokenizer.json` vocabulary and merge list.
- `csrc/executor/` — `gemm`, `rmsnorm`, `rope`, `attention`, `swiglu`,
  `sampler`, `batch` (the padding helper), and `model.cpp` (the class that
  wires all of them into one forward pass).
- `csrc/kv/kv_cache.{h,cpp}` — the contiguous, single-sequence KV cache.
- `kiln_py/runtime/generate.py` — the token-by-token generation loop.
- `kiln_py/runtime/byte_tokenizer.py` — builds a plain byte-level
  vocabulary file (used by the demo API, since no real trained vocabulary
  is available in this offline environment).
- `kiln_py/scheduler/scheduler.py` — the continuous-batching scheduler.
- `kiln_py/api/app.py` — the OpenAI-compatible FastAPI server.
- `tools/oracle.py` — runs the reference HF model and dumps per-layer
  activations and logits for later parity comparison (not yet exercised
  against a real download in this offline session -- see ADR-009).

## Honest state of Part I, as of this session

Every C++ unit test (29) and every Python test (13) passes, including
under AddressSanitizer/UndefinedBehaviorSanitizer. The pybind11 boundary
was verified end to end (Python calling the real C++ model, not a stub).
What has **not** been verified in this session: numerical parity against a
real HuggingFace checkpoint (needs a real `transformers` install; deferred
to Kaggle/local per ADR-009), tokenizer conformance against real text on a
large fixture set, and the scheduler wired to the API for genuine
concurrent multi-request continuous batching (the API currently drives the
single-sequence generation loop directly). These are named here rather than
implied to be done.
