# Kiln

Kiln is an LLM serving engine built from scratch — the same kind of
system that runs behind a chat product, but written by hand instead of
imported. Python handles the traffic (requests, scheduling, accounts);
C++ does the actual thinking (reading the model's weights, running the
math, remembering the conversation). Every piece is checked against a
reference implementation before it's trusted, and every shortcut this
project takes is written down plainly, not hidden.

**One sentence:** point Kiln at a model's raw weights, and it serves
chat-style requests — many at once, remembering what it's already read,
with the option to shrink the model down for speed — the way a real,
production LLM server works, minus the parts that need a rented GPU or a
real audience to build honestly.

## The delta table

Every row below is a real, reproducible measurement from
`bench/run_benchmarks.py` — run it yourself with `PYTHONPATH=.
python3 bench/run_benchmarks.py`. There is no GPU in this environment, so
this all runs on CPU against a small, **randomly-initialized (untrained)**
model — real numbers, honest hardware, not the production claim. The
metric used per row is whichever one that optimization actually changes;
forcing every row into the same column would be less honest, not more
rigorous.

| Optimization | What it actually changes | Measured result |
|---|---|---|
| Naive (recompute every step) | decode throughput | **254.7 tok/s** |
| + KV cache | decode throughput | **2,843 tok/s** (**11.2×**) |
| + KV cache (representative path) | TTFT / TPOT, p50 / p99, n=20 trials | TTFT 2.14 / 2.47 ms · TPOT 0.28 / 0.31 ms |
| + continuous batching | wall-clock time, mixed-length workload (6 requests) | **1.42×** faster than static batching |
| + paged KV cache | max concurrent sequences at fixed memory | contiguous **4** → paged **21** → paged+shared-prefix **62** |
| + INT8 quantization | memory footprint · reconstruction error | **3.76×** smaller · MSE **3.5×10⁻⁵** (CPU speed *not* faster — no INT8 kernel; see below) |
| + speculative decoding | target-model calls per token | **1.0×** (no reduction) — both models are independently untrained, so the draft's guesses essentially never match the target's; see below |

The paged-cache sharing path also has a separate seeded workload:
`cmake --build build --target kiln_prefix_cache_benchmark &&
./build/kiln_prefix_cache_benchmark`. It ran 80 synthetic conversations
with a shared system prompt and four prompt variants, and measured **248
hits in 252 prefix-block lookups (98.41%)**. This is a local, synthetic
cache-mechanism measurement -- not a rate from users or playground
traffic.

**Two rows that need explaining, not hiding:**

- **INT8 shows no CPU speedup.** The memory reduction and the accuracy
  cost are real, measured numbers. The *speed* win real INT8 quantization
  provides comes from an INT8 GPU kernel — this project's Phase 7 kernels
  are written but unverified (no GPU here), so there is genuinely nothing
  to measure yet on the speed axis. Reporting a CPU number as if it
  demonstrated the GPU win would be the fabrication this project
  specifically refuses to do.
- **Speculative decoding shows a 1.0× (zero) reduction here, and that's
  the correct, honest result for this setup.** Its entire payoff depends
  on the draft model's guesses actually landing — and two independently,
  randomly-initialized models have essentially no reason to agree (about
  a 1-in-1000 chance per token, at this toy vocabulary size). The
  mechanism is proven correct elsewhere (see the speculative-decoding
  section below: exact, token-for-token, against greedy decoding) — this
  row is a measurement of *this session's* draft/target pairing, not a
  claim that speculative decoding doesn't work.

See `BENCHMARKS.md` for the full per-phase history and `BENCHMARK.md` for
the roofline / arithmetic-intensity analysis of why decode and prefill
behave so differently in the first place.

## What actually works right now

Kiln can load a model, turn text into the numbers it understands and
back, generate new text one word at a time while remembering everything
it's already read, serve several people at once fairly, and answer over
a normal web API that other tools (like the `openai` Python package)
already know how to talk to. On top of that: a memory system that shares
identical prompts between conversations instead of storing them twice; a
way to shrink the model's numbers down for less memory use; a proven
(not just claimed) trick for generating text faster using a second,
smaller model as a scout; API keys and usage limits for multiple
separate users; and a small web page to try all of it.

**What doesn't work yet, and why, in one line each:** the GPU-only pieces
(the hand-written fast kernels) are written but never compiled, because
this was built on a machine with no NVIDIA GPU. Nothing has been checked
against a real, trained model's answers, because running one needs an
internet-heavy install this environment didn't have. And nobody has
actually used this — there's no live website, no real users, no incident
that ever happened — because that would take an actual public launch,
which is a decision for later, not something to fake. Every one of these
is written down in detail, not glossed over — see **Honest status**
below.

## How it fits together

```mermaid
flowchart TD
    client["A client<br/>(curl, the openai package, a browser)"]

    subgraph py ["Python — decides what happens"]
        API["Web API<br/>(OpenAI-shaped requests)"]
        SCHED["Scheduler<br/>(who gets to run right now)"]
        CP["Control plane<br/>(API keys, usage limits)"]
    end

    subgraph cpp ["C++ — does the actual thinking"]
        MODEL["The model itself<br/>(reads weights, runs the math)"]
        MEM["Memory<br/>(remembers the conversation so far)"]
    end

    client --> API --> SCHED --> MODEL
    CP -.->|checks the key, checks the limit| API
    MODEL <--> MEM
```

**Why split it this way?** Deciding *who gets to talk to the model next*
is a policy question — it doesn't need to be fast, it needs to be easy to
get right and easy to change. Actually *running* the model is a raw-speed
question. Keeping those two concerns in two different languages, talking
through one narrow, well-defined bridge, is exactly how real production
engines (the ones this project is modeled on) are built — it's not a
compromise, it's the standard shape of the thing.

## The big picture (what each piece is for)

- **Reading a model's weights and turning text into numbers.** A model
  file is opened without copying it into memory twice, and a sentence is
  turned into a list of numbers (and back) the same way real tokenizers
  do it.
- **The forward pass.** The actual "thinking" — read the conversation so
  far, decide what matters, produce a guess at the next word. Written
  from scratch, checked piece by piece.
- **Remembering the conversation (the cache).** Redoing all that thinking
  for every single new word would be wasteful — the cache remembers what
  was already computed, and a second, more advanced version shares
  identical prompts between separate conversations instead of storing
  them twice.
- **Serving many people at once.** New conversations join in and
  finished ones leave immediately, instead of everyone waiting for the
  slowest conversation in the batch to finish before anyone new can
  start.
- **Shrinking the model down (quantization).** Trading a little precision
  in the model's numbers for a lot less memory — with the accuracy cost
  measured, not assumed.
- **A faster way to generate text (speculative decoding).** A smaller,
  faster model guesses several words ahead; the real model checks all the
  guesses in one pass. Proven — not just claimed — to produce the exact
  same answer as not using the shortcut at all.
- **Accounts and limits.** Separate users, each with their own key and
  their own usage limit, that can't interfere with each other.
- **Checking whether a change made things better or worse.** Infrastructure
  for comparing two versions honestly, using statistics that can tell a
  real improvement apart from random noise.

## What each piece is actually made of (for anyone reading the code)

| Piece | Where | What it's built from |
|---|---|---|
| Reading model weights | `csrc/loader/` | opens the file without copying it, reads a small directory describing where each number lives |
| Turning text into numbers | `csrc/tokenizer/` | the standard "merge common pairs of letters together" approach real tokenizers use |
| The forward pass | `csrc/executor/` | matrix multiplication, a normalizing step, the "attention" mechanism, and a small decision-making network — chained together |
| Remembering the conversation | `csrc/kv/` | a straightforward version, and an advanced version that shares memory between conversations and only copies it the moment two conversations actually diverge |
| Choosing the next word | `csrc/executor/sampler.*` | always picking the best guess, or picking with some controlled randomness |
| Shrinking the model | `csrc/quant/` | representing many numbers with one shared "how big are these, roughly" value plus small individual differences |
| Deciding who runs next | `kiln_py/scheduler/` | a waiting line and a running list, moving people between them as room opens up |
| The faster generation trick | `kiln_py/runtime/speculative_decode.py` | a small model guesses, a big model checks all the guesses in one go |
| The web API | `kiln_py/api/` | matches the shape of OpenAI's own API, so existing tools work against it unmodified |
| Accounts and limits | `kiln_py/control_plane/` | keys that are never stored in a readable form, and limits checked *before* letting a request run |

## Honest status

Every claim above has a receipt. `docs/walkthrough.md` has the full,
plain-language tour; `docs/defense.md` explains, phase by phase, what was
built and exactly what it cost; `docs/correctness.md` is a running list
of real bugs this project found in itself (and how); `BENCHMARKS.md` has
the itemized, nothing-hidden list of what's actually been measured versus
what's honestly still missing. Nothing here claims more than what was
actually run and checked.

As of this writing: the checked-in suites include **51 C++ tests**
(including memory-safety tooling) and the Python API/control-plane tests.
The whole thing has also been built and run inside a real Docker container.

## Try it

```
pip install -e .                                          # builds the C++ side
cmake -B build -G Ninja && cmake --build build             # or, to just run the C++ tests directly
ctest --test-dir build --output-on-failure
PYTHONPATH=. python3 -m pytest tests/py                     # the Python side

bash demo.sh                                                # the full five-minute tour
```

Then open `http://localhost:8420/` for the playground and
`http://localhost:8420/status` for this local process's counters, or read
`docs/writeups/` for three longer explanations of the most interesting
parts (the shared-memory conversation cache, the "how do we know it's
still right" testing philosophy, and what shrinking a model actually
costs).

## Where things live

```
csrc/            the C++/CUDA side — everything that does the actual math
kiln_py/         the Python side — requests, scheduling, accounts, the web API
tests/           cpp/ (GoogleTest) and py/ (pytest)
tools/           the reference-model comparison script, a quantizer cross-check
docs/            adr/ (why each big decision was made), learning/ (derivations,
                 phase by phase), writeups/ (the longer explanations),
                 walkthrough.md, defense.md, correctness.md, postmortems/
deploy/          Dockerfile, docker-compose (engine + Prometheus + Grafana)
demo.sh          the scripted end-to-end tour
```

The full 18-phase plan this project follows lives in
[`../KILN PLAN.md`](../KILN%20PLAN.md).
