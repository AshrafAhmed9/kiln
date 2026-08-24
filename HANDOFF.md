# Handoff to Codex

This project was built by Claude Code across several sessions. Ashraf is
switching tools; this document is the full state transfer. Read this
before touching anything else. `KILN PLAN.md` (one directory up, at
`../KILN PLAN.md`) is the original 18-phase plan this project follows —
read that second, for the full intent behind each phase.

## What Kiln is

An LLM inference engine built from scratch: Python orchestrates (API,
scheduler, control plane), C++ computes (forward pass, kernels, memory).
See `README.md` for the plain-language tour and architecture diagram —
it's accurate and current as of this handoff.

## Rules you must follow — these are not optional style preferences

1. **Never fabricate a result.** This is the single most important rule
   in this repo, enforced more strictly than anywhere else in Ashraf's
   other projects. If something isn't measured, say so — don't estimate,
   don't imply, don't round up. Every number in this repo that looks
   measured *was* actually measured, by running real code. Search
   `docs/correctness.md` for examples of bugs caught specifically because
   something was actually run rather than assumed to work.
2. **No real users, no real deployment, no real traffic exist.**
   Everything in this repo ran locally, in a sandboxed dev environment,
   against a small, untrained, randomly-initialized model. Do not build a
   "status page with real production metrics," do not claim a "hit rate
   on real traffic," do not write a postmortem for an incident that
   didn't happen. `docs/postmortems/TEMPLATE.md` has no incident in it on
   purpose. If Ashraf asks for a public launch, that is a real decision
   for him to make outside of a coding session, not something to simulate.
3. **This machine (and likely yours) has no NVIDIA GPU.** Raw CUDA was
   nevertheless compiled and checked remotely on a Kaggle Tesla P100
   (sm_60, NVCC 12.8): four CPU-vs-GPU tests and the full 57-test CTest run
   passed at revision `dc792e1`. Triton RoPE was also run successfully after
   pinning a Pascal-compatible PyTorch wheel. No kernel is integrated into
   model execution. Do not inflate these correctness results
   into a GPU performance claim. See the backlog below.
4. **Comments are plain, human, jargon-free language** — someone should
   be able to answer an interview question about a piece of code just by
   reading its comment. This is a standing instruction from Ashraf; don't
   make him repeat it.
5. **Minimum code, maximum clarity, is the actual design objective** —
   ranked above performance. `csrc/` targets 6–10k lines total (currently
   well under budget — check with `wc -l`). When a design choice is
   between "faster" and "easier to explain," legible wins, and the
   tradeoff gets written down (see the pattern in `docs/defense.md`).
6. **Every phase gets a defense-doc entry** (`docs/defense.md`: what it
   is, why it works, what it cost) and, for anything with real derivation
   behind it, a learning note (`docs/learning/phase-NN.md`). This is how
   Ashraf actually studies what gets built — don't skip it to move faster.
7. **Test before claiming done.** Every fix in this repo's history came
   with a regression test. `docs/correctness.md` is the running log of
   real bugs found — several were caught only by *actually running*
   something (a Docker build, a benchmark script), not by code review.
   Follow that pattern: build it, run it, then say it works.

## How to verify the current state (do this first)

```bash
cd kiln  # you're likely already here
cmake -B build -G Ninja -DKILN_SANITIZE=ON -DKILN_BUILD_PYBIND=OFF
cmake --build build
ctest --test-dir build --output-on-failure     # expect 49/49 passing

cmake -B build-py -G Ninja -DKILN_BUILD_PYBIND=ON \
  -Dpybind11_DIR="$(python3 -c 'import pybind11; print(pybind11.get_cmake_dir())')"
cmake --build build-py --target _C
PYTHONPATH=. python3 -m pytest tests/py -q                 # expect 59/59 passing
python3 -m mypy kiln_py                                     # expect clean

bash demo.sh                                                 # the scripted end-to-end tour
```

If any of these don't pass, something regressed since this handoff — fix
that before adding anything new.

## What's genuinely done (all 18 plan phases attempted; see BENCHMARKS.md for the itemized per-phase list)

- **Part I (0–6):** full CPU forward pass, safetensors loader, BPE
  tokenizer, KV cache, padded batching, an Orca-style continuous-batching
  scheduler, an OpenAI-compatible API with SSE streaming.
- **Part II (7–12):** paged KV cache with copy-on-write prefix sharing
  (real, tested); INT8/INT4 quantization cross-checked against an
  independent Python reference (real, tested); speculative decoding
  **proven token-for-token exact** against greedy decoding (the strongest
  correctness result in the repo); a parity-harness self-test; CPU-
  simulated tensor-parallel sharding math. Phase 7's CUDA/Triton kernels
  are the one part of Part II that's unverified (no GPU).
- **Part III (13–15):** eval infrastructure (exact-match, perplexity,
  bootstrap-CI regression gating, canary replay); LoRA adapter merging
  (real training is out of scope — no PyTorch training setup); a demo
  script actually run end-to-end; three writeups.
- **Part IV (16–18):** a separate multi-tenant control-plane service
  (hashed API keys, quotas/rate-limits enforced before generation, usage
  metering) with both of the plan's named tests passing; a static
  playground page (a deliberate scope call vs. the plan's Next.js app);
  real Prometheus metrics; a Dockerfile + docker-compose stack **actually
  built and run** in a real container (caught two real bugs doing so —
  see `docs/correctness.md`). No real deployment, users, or incidents —
  see rule #2 above.
- **Extras added after the 18 phases, at Ashraf's request:**
  - A real delta-table benchmark harness (`bench/run_benchmarks.py`) and
    `BENCHMARK.md`'s roofline/arithmetic-intensity analysis.
  - `docs/writeups/04-why-vllm-beats-kiln.md` — an honest gap analysis.
  - Chunked prefill + a 3-policy scheduler simulation with a real Pareto
    frontier plot (`kiln_py/scheduler/chunked_prefill_sim.py`,
    `bench/scheduler_policy_comparison.py`) — went through three rounds
    of self-caught modeling bugs before the frontier was actually honest;
    read `docs/learning/phase-19.md` before touching this code, so you
    don't reintroduce one of those three mistakes.
  - JSON-schema-constrained decoding via logit masking
    (`kiln_py/runtime/constrained_decode.py`), reusing the existing
    sampler. Tested via `json.loads()` actually succeeding, not by
    eyeballing output.

## What's requested but NOT yet built — pick up here

Ashraf asked for four things in the same message; two are built (above),
two are not yet started:

1. **Prefix-cache hit-rate measurement.** Ashraf's exact framing
   ("measured on your real playground traffic") was factually wrong — no
   real traffic exists (rule #2) — and this was corrected to him
   directly. What to build instead: a hit-rate measurement using
   `csrc/kv/paged_kv_cache.h`'s existing copy-on-write block sharing,
   driven by a **synthetic, seeded workload** (e.g., many simulated
   conversations sharing a common system-prompt prefix, some diverging
   partway through). Report real hit-rate numbers from that synthetic
   run, labeled honestly as synthetic. Follow the pattern in
   `tests/cpp/kv/paged_kv_cache_test.cpp` for how blocks/ref-counts are
   exercised.
2. **A metrics/status page.** Same correction applies — no real
   production traffic exists. Build a page that reads the **real**
   Prometheus counters already wired in `kiln_py/metrics/__init__.py` and
   exposed at `/metrics` (see `kiln_py/api/app.py`), labeled explicitly as
   this session's local test data, not production. Do not write "p99",
   "QPS," or "cost-per-million-tokens" as if they're real unless you have
   actually deployed this somewhere with real traffic to measure — if you
   don't, say so on the page itself.

## The honest backlog — real, deferred work, not forgotten

Everything below is a genuine gap, not an oversight. Each one is named in
its own `docs/defense.md` / `docs/learning/phase-NN.md` entry with the
specific reason it wasn't done — read the relevant one before starting.

- **Remaining GPU work (Phase 7, and the GPU half of Phases 9/10/12):**
  raw CUDA has now compiled and passed small reference tests on a Kaggle
  P100, and Triton RoPE has run against a CPU reference. Profile the raw
  kernels with Nsight Compute and get real tokens/s and
  bandwidth-utilization numbers for `BENCHMARK.md`'s roofline section.
  The zero-budget plan's answer remains Kaggle GPU notebooks (ADR-009);
  these measurements unlock real numbers for
  several other rows in `BENCHMARKS.md` too (real quantization
  speed/accuracy tradeoffs, real speculative-decoding speedup with two
  actually-related models instead of two independently random ones, real
  multi-GPU tensor-parallel scaling).
- **Broader internal numerical parity.** A downloaded SmolLM2 checkpoint
  now passes final-logit comparison across ten diverse prompts, but Kiln
  does not yet expose per-layer activations for direct comparison with
  `tools/oracle.py`.
- **Tokenizer conformance** is complete for the named seeded 10,000-string
  SmolLM2 fixture (10,000 exact matches); other tokenizer configurations
  remain outside that evidence.
- **LoRA quality/evaluation and multi-GPU scaling** (Phase 14): a real
  small PEFT training/export/merge route is now present, but it needs a
  licensed task dataset, held-out evaluation, and multiple GPUs for the
  DDP→FSDP study. PyTorch remains training tooling only.
- **Wiring the scheduler into the API for genuine concurrent multi-
  request continuous batching.** The API currently drives the Phase 3
  single-sequence generation loop directly; `kiln_py/scheduler/` exists
  and is tested in isolation (against a mock executor) but isn't
  connected to real HTTP request handling yet. This is real, valuable,
  buildable-without-a-GPU work.
- **A real public launch (Phase 18).** Not a coding task — a decision
  Ashraf makes. If/when he says to actually deploy this somewhere real,
  that's when `deploy/Dockerfile` and `deploy/docker-compose.yml` (both
  real and verified) get used for real, and the metrics/status page
  above stops being "local test data" and starts being true.

## Where things live

See `README.md`'s "Where things live" section — it's accurate. The short
version: `csrc/` is C++/CUDA, `kiln_py/` is Python, `tests/` mirrors both,
`docs/` has everything explaining *why* (read `docs/adr/`, `docs/
learning/`, `docs/defense.md`, `docs/correctness.md` before making a
design decision that might already have a recorded answer), `bench/` has
the real, runnable benchmark scripts, `deploy/` has the verified Docker
setup.

## One last thing

Ashraf has a memory system (outside this repo) tracking standing
preferences for this project — comment style, the zero-budget constraint,
the learning-loop process, the full-scope decision, and this handoff's
existence. You won't have access to that. Everything load-bearing from it
has been folded into this document and into the repo's own `docs/`. If
Ashraf gives you an instruction that seems to reference something you
don't have context for, ask him rather than guessing.
