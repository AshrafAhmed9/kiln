# Interview prep — the questions you'll want to be asked

`docs/defense.md` answers "what did you build, why does it work, what did
it cost" per phase. This file is different: it's the prepared answers to
the *specific questions* an interviewer is likely to actually ask, framed
as a script to have ready rather than material to derive live. Pulled out
of the original project plan (since deleted as outdated) because this
part of it never stopped being useful.

## The two questions the hybrid architecture invites

These aren't weaknesses to explain away — they're the two questions worth
being asked, because the answer is a measured number, not a shrug.

**"Why isn't the whole thing in C++?"**
→ Because it was measured, not assumed. Roughly 99% of wall-clock time is
inside the kernels, which are C++/CUDA either way — the orchestration
overhead is a per-*iteration* scheduling cost, not a per-token compute
cost, and Phase 4's benchmark row commits the actual split. Putting the
HTTP layer and scheduler in C++ would buy a fraction of a percent and cost
real weeks. It's also the same split PyTorch, vLLM, and TensorRT-LLM
actually use — Python front end, C++/CUDA underneath — so it's the
architecturally faithful choice, not a shortcut. Where the answer would
change: at very high request rates on very small models, the
per-iteration overhead stops being noise — that crossover point is where
real engines move schedulers into C++, and it's worth naming even though
Kiln hasn't needed to cross it.

**"Which parts did you actually write in CUDA?"**
→ Name them precisely, don't gesture at "the kernels": attention, RMSNorm,
and the greedy-argmax sampler are hand-written raw CUDA, each using the
warp-shuffle reduction pattern (many threads combine partial results
without a shared-memory bottleneck). RoPE is written twice — once raw CUDA,
once Triton — specifically to have a real, measured basis for "why not
Triton for everything?" instead of an assertion. GEMM goes through cuBLAS
deliberately (beating cuBLAS by hand is a multi-year compiler project, not
a learning opportunity). Precision here is what makes the claim
bulletproof; vagueness is what makes an interviewer start probing for
exaggeration.

## Per-topic: the question, and where the answer lives

- **"How does attention actually work?"** — implemented four times (CPU,
  paged CPU, raw CUDA, and compared against Triton for RoPE), parity-tested
  against each other and against a real Hugging Face checkpoint
  (`docs/learning/phase-22.md`).
- **"How would you design an inference server?"** — the Orca-style
  continuous-batching scheduler (`kiln_py/scheduler/scheduler.py`),
  measured against static batching under a mixed workload
  (`docs/defense.md`, Phase 5).
- **"Why is vLLM fast, and where does it beat you?"** — implemented paged
  attention's actual core idea (block-table allocator, copy-on-write
  prefix sharing) and have the max-concurrency numbers; named the gap
  honestly rather than implying parity (`docs/writeups/04-why-vllm-beats-kiln.md`).
- **"What does quantization cost?"** — the accuracy/latency/memory
  tradeoff table, per scheme, cross-checked against an independent Python
  reference quantizer (`docs/writeups/03-quantization-tradeoff-study.md`).
- **"Does speculative decoding change outputs?"** — provably not: seeded
  spec-decode output is token-for-token identical to seeded greedy decode,
  by construction of the rejection-sampling rule, not by coincidence
  (`docs/defense.md`, Phase 10). Explain the acceptance rule from memory,
  not from the code.
- **"Hardest bug you've debugged?"** — pick any entry in
  `docs/correctness.md`; every one is a real numerics or build detective
  story, including the nlohmann_json regression caught re-running the
  CUDA suite on a fresh Kaggle session, and the two Docker bugs caught only
  by actually building and running the image.
- **"How do you know it works?"** — the parity harness: every exact-path
  component is diffed token-by-token against a reference (Hugging Face for
  the model itself, an independent Python quantizer for INT8/INT4); every
  lossy path has a measured accuracy budget instead of a hand-wave.

## Why the project doesn't have a launch/users story

The original plan included a public launch, real users, and retention
metrics as its own phase. That was deliberately not simulated — no real
users, no real deployment, no fabricated incident exists anywhere in this
repo (see `docs/defense.md`'s Phase 18 entry and `docs/postmortems/TEMPLATE.md`,
which is empty on purpose). If asked "have people used this," the honest
answer is no, and the reasoning for not faking one is itself a legitimate
answer about engineering integrity under a portfolio-project's incentive
to embellish.
