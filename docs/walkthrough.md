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

## Part II additions

- `csrc/kv/paged_kv_cache.{h,cpp}` and `csrc/executor/paged_attention.{h,cpp}`
  — the block-table KV allocator with copy-on-write prefix sharing, and a
  CPU attention variant that reads through it; proven to match contiguous
  attention exactly.
- `csrc/quant/quantize.{h,cpp}` — INT8 per-channel and INT4 group-wise
  (packed) quantization, cross-checked against `tools/quantize_ref.py`
  (an independent Python reimplementation) through the pybind11 boundary.
- `kiln_py/runtime/speculative_decode.py` — the rejection-sampling
  speculative decoding loop; proven token-for-token exact against direct
  greedy decoding in `tests/py/test_speculative_decode.py`.
- `kiln_py/runtime/tensor_parallel_sim.py` — column-parallel and
  row-parallel matrix multiplication, simulated across N ranks in numpy,
  proven exact against the unsharded computation.
- `csrc/kernels/cuda/*.cu` and `csrc/kernels/triton/rope.py` — hand-written
  CUDA (attention, RMSNorm, greedy-argmax sampling) and Triton (RoPE, both
  ways for the ADR-007 comparison) kernels. Raw CUDA compiled and passed
  four CPU-vs-GPU checks on Kaggle P100 and T4 hardware; Triton RoPE matches
  its CPU reference, and the T4 run includes a narrow event-timed comparison.

## Honest state, as of this session

**Part I:** every C++ unit test and every Python test passes, including
under AddressSanitizer/UndefinedBehaviorSanitizer (60/60 total across both
parts as of Phase 12 -- see `BENCHMARKS.md` for the exact count per part).
The pybind11 boundary was verified end to end (Python calling the real
C++ model, not a stub). Normal and streaming API completions now go through
the scheduler-backed continuous-batching worker; concurrent API tests cover
that path. Prompt prefill is batched without padding, while each sequence
still owns a contiguous KV cache rather than sharing paged prefix blocks.

**Part II:** this machine has no NVIDIA GPU, but the raw CUDA kernels have
been compiled and run remotely on Kaggle P100 and T4 GPUs, matching small CPU
references in four tests. Triton RoPE is also verified and has one narrow T4
comparison; device-resident CUDA prefill and cached decode also passed
full-logit CPU parity on a T4, but they remain unprofiled and are not the API's default
executor. Everything else in Part II (paged KV cache,
quantization, speculative decoding, the parity-harness self-test, and
tensor-parallel sharding math) *is* real, CPU-testable, and tested for
real, but each has a named gap versus the plan's full GPU-based
deliverable: no real perplexity/KL table (needs a real checkpoint and
WikiText-2), no real decode-speedup measurement, no real multi-GPU scaling
efficiency. All of these are deferred to Kaggle T4 time, per ADR-009 --
named explicitly here and in `BENCHMARKS.md`, not implied to be done.

## Part III additions

- `kiln_py/eval/` — exact-match task scoring, a perplexity calculator, a
  bootstrap-confidence-interval regression gate (paired, not just an
  average comparison), and canary-replay diffing. Tested against small,
  hand-constructed inputs rather than the real (untrained) model, since
  that's the more meaningful test available offline.
- `csrc/executor/lora.{h,cpp}` and `Model::MergeLoraIntoLayer` — folds an
  already-trained LoRA adapter into a served model's weights. Proven both
  by a hand-computed matrix check and by confirming the merge changes a
  real forward pass's output. Real LoRA *training*, the data pipeline, and
  the multi-GPU scaling study are explicitly out of scope (no PyTorch
  training setup or GPUs available) -- see docs/defense.md.
- `demo.sh` -- the scripted five-minute demo, **actually run end to end in
  this session**, not just written and assumed to work.
- `docs/writeups/` -- three longer explanations (the paged allocator, the
  parity-harness methodology, the quantization tradeoff study), drafted
  from the phase learning notes.

## Part IV additions

- `kiln_py/control_plane/` -- API keys (SHA-256 hashed, never stored raw),
  daily token quotas and per-second rate limits (both enforced *before* a
  request runs), and usage metering, as its own FastAPI service per
  constitution §6. Both of the plan's named Phase 16 tests pass:
  tenant-isolation-under-hammering, and a leaked-key revocation drill.
  In-memory store, not Postgres -- the enforcement logic is proven, the
  persistence layer isn't built.
- `kiln_py/api/playground.html` -- a self-contained static page (a
  deliberate scope call instead of the plan's Next.js app; Node was
  actually available -- see docs/learning/phase-17.md), live side-by-side
  temperature comparison with real measured latency, manually verified
  against a real running server.
- `kiln_py/metrics/` and the `/metrics` endpoint -- real Prometheus
  counters and a histogram, verified to actually increment on a real
  request.
- `deploy/` -- a Dockerfile and docker-compose stack (engine + Prometheus
  + Grafana), **actually built and run** in this session on real Docker,
  which caught two real bugs along the way (see docs/correctness.md): a
  missing `-fPIC` flag that only breaks the build on Linux, and a missing
  copy of installed console scripts that broke the container's entrypoint.
  Both fixed and re-verified by rebuilding and re-running the container.
- **What is deliberately, explicitly not done:** any real public
  deployment, any real users, any real incident, any real retention
  number. None of these are fabricated -- see docs/learning/phase-18.md
  for why that boundary matters more here than anywhere else in this
  project. `docs/postmortems/TEMPLATE.md` is a template with no incident
  in it.

## Final honest tally

94 automated tests pass as of this session (49 C++, checked under
AddressSanitizer/UndefinedBehaviorSanitizer; 45 Python, mypy clean), plus
a real, manually-verified Docker deployment. Everything genuinely GPU-
dependent (Phase 7's kernels, real quantization/speculative-decoding
numbers, real multi-GPU scaling) is written to spec and deferred to
Kaggle T4 time. Real LoRA training and a real public launch are deferred
for reasons independent of GPU budget, named in their own phase notes.
Nothing in this project claims a result that wasn't actually produced and
checked in this session.
