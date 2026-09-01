# Correctness — the "how do you know" document

This is the running catalog of what the parity harness checks, what it
can't, and every real bug it (or a test written in this same spirit) has
caught. Each entry: what broke, how it was caught, the fix, and what I
misunderstood -- the misconception, not just the bug.

## Phase 0 — the debug-only assert that made two builds disagree

**What broke:** `Arena::Allocate()` had a debug-only assert on top of its
documented "returns a null pointer when full" contract. In a normal debug
build (assertions on), running out of room crashed the program instead of
returning null.

**How it was caught:** the test `Arena.ReturnsNullWhenExhausted` expected a
null pointer and instead the whole test process aborted.

**The fix:** removed the assert. One rule, in every build: run out of
room, get a null pointer back, always.

**What I misunderstood:** I assumed "crash loudly in debug, degrade
gracefully in release" was a reasonable default without checking that it
actually matched what the class's own documented contract promised. Two
different behaviors for one function is complexity, even when each half
seems reasonable on its own -- the fix that mattered wasn't fixing the
assert's wording, it was noticing the design had two contracts instead of
one.

## Phase 5 — admission control that checked the wrong number

**What broke:** the scheduler let a new request in in as long as its
*current* size fit inside the remaining budget. Since every running
request grows by one word every step, a batch that fit perfectly the
moment a request was admitted could still overflow its budget a few steps
later, once everyone already running had grown some more.

**How it was caught:** not by a failing test at first -- by re-reading the
scheduler's own logic while writing its randomized invariant test
(`test_memory_budget_is_never_exceeded_under_random_arrivals`) and
realizing the check being written (`tokens_reserved() <= max_batch_tokens`
after every step) would only be meaningful if admission itself reserved the
right number up front.

**The fix:** admission now checks a request's worst-case final size (its
prompt length plus its full allowance of new words), not its size right
now -- matching how a cache that can't be resized after the fact actually
has to be managed.

**What I misunderstood:** I was thinking about "does it fit right now,"
which is the natural first question, without carrying through the
follow-up question: does it still fit after it's allowed to grow? Any
system where one thing's size increases over time needs its admission
decisions to be based on where that thing will end up, not where it starts.

## Phase 6 — raw bytes assumed to be text

**What broke:** the streaming API endpoint crashed with a Unicode decoding
error the first time a generated token happened to be a raw byte that
wasn't valid text on its own.

**How it was caught:** the very first real run of the streaming endpoint's
test, immediately.

**The fix:** the C++ tokenizer's decode function now hands back raw Python
bytes instead of an auto-converted string, and the Python side explicitly
decodes those bytes with a "replace anything broken with a placeholder
character" policy.

**What I misunderstood:** I treated "decode these tokens" as if it always
produces displayable text, when a byte-level tokenizer's real contract is
narrower than that -- it produces bytes, and turning bytes into text is a
separate, sometimes-lossy step that has to be done deliberately, especially
one token (one possibly-incomplete character) at a time during streaming.

## Phase 8 — copy-on-write that never actually copied

**What broke:** the first version of `PagedSequence::PrepareWriteSlot`
allocated a new, private block whenever a shared block needed to be
written to, but never actually copied the old block's numbers into the new
one -- so a sequence that diverged from a shared prefix would have started
writing into (and, worse, reading stale garbage out of) an uninitialized
block.

**How it was caught:** before any test was even run, while re-reading the
function against its own documentation comment and noticing the comment
claimed a copy that the code never performed.

**The fix:** added `PagedKVCache::CopyBlockContents`, which copies both K
and V numbers, at every layer, from the old block into the new one, and
call it at the exact moment a shared block needs to be written to.

**What I misunderstood:** I wrote the allocate-a-new-block half of
copy-on-write and treated that as if it were the whole mechanism, when
allocating room and actually copying the data into it are two separate
steps -- skipping the second one leaves the "write" correct but the
"copy" fictional. The lesson generalizes: a function's own comment
describing what it does is worth treating as a claim to verify against the
code, not just documentation to trust.

## Phase 18 — two real bugs, caught only by actually building and running the Docker image

**What broke (bug 1):** the C++ compute library (`kiln_cpp`) was built as
a plain static library with no position-independent code flag. Linking a
static library like that into a *shared* object (the `_C` pybind11
extension) works by coincidence on macOS, but fails outright on Linux.

**How it was caught:** actually running `docker build` against the real
Dockerfile, on this machine, rather than writing the Dockerfile and
assuming it would work. The build failed with a wall of linker errors
("dangerous relocation... recompile with -fPIC").

**The fix:** `set_target_properties(kiln_cpp PROPERTIES
POSITION_INDEPENDENT_CODE ON)` in `CMakeLists.txt`.

**What I misunderstood:** I had verified this build extensively on one
platform (macOS, this development machine) and let that stand in for "the
build works," without noticing that a real, meaningful difference in how
the two platforms link shared objects meant local verification here
didn't actually prove anything about Linux, which is where this almost
always actually gets deployed.

**What broke (bug 2):** after fixing bug 1, the built image's `CMD`
(`uvicorn ...`) failed with "executable file not found in $PATH." The
multi-stage Dockerfile's final stage copied installed Python *packages*
(`site-packages`) from the build stage, but not the installed console
*scripts* (`/usr/local/bin/uvicorn` and friends) that `pip install` also
creates.

**How it was caught:** the same way — running the container after fixing
bug 1 and watching it fail immediately, rather than assuming a
successful `docker build` meant a working container.

**The fix:** also copy `/usr/local/bin` from the build stage into the
final stage.

**What I misunderstood:** I treated "the Python packages are installed"
as equivalent to "the tools built from those packages are available,"
when `pip install` actually produces two separate things in two separate
places — and a multi-stage Docker build that copies files explicitly, by
path, will only include exactly what's named, not "everything pip did."

## Phase 7 (re-verification) — a version pin broke a previously-working CUDA build

**What broke:** re-running the CUDA build on a fresh Kaggle GPU session
failed at `cmake` configure with `Could not find a configuration file for
package "nlohmann_json" that is compatible with requested version "3.11"`.
Ubuntu 22.04's `nlohmann-json3-dev` apt package ships 3.10.5; `CMakeLists.txt`
required 3.11.

**How it was caught:** actually running the CUDA parity suite again from a
clean clone via a scripted Kaggle kernel, rather than assuming a previously
successful GPU run (revision `dc792e1` and later) meant the build still
worked on a brand new session with a brand new apt cache.

**The fix:** lowered `find_package(nlohmann_json 3.11 REQUIRED)` to `3.10`.
No 3.11-only feature is used anywhere in this codebase, so this is a real
compatibility fix, not a version-pin workaround.

**What I misunderstood — or rather, what the project as a whole
under-verified:** a build succeeding once, on one session, with one apt
cache, isn't the same claim as "the build works from a clean checkout" —
the same class of mistake as the Phase 18 Docker bugs above (macOS
working "by coincidence" while Linux failed), just one layer further out:
this time it was the *toolchain's own dependency resolution*, not the
compiled code, that silently depended on whatever happened to already be
installed. After the fix: 61/61 CTest checks passed on a real Kaggle
P100 (compute capability 6.0, detected and passed to `CUDA_ARCHITECTURES`
at build time rather than assumed), and the RoPE kernel measured
**216.5 GB/s effective bandwidth** (512 tokens, 9 heads, head_dim 64,
median of 1000 iterations). Nsight Compute profiling was attempted in the
same run and failed with `ERR_NVGPUCTRPERM` — Kaggle's container does not
grant GPU performance-counter access — a real, named platform limitation,
not a build bug.

## Phase 25 (attempted) — Nsight profiling blocked by GCP capacity, not by anything in this repo

**What happened:** `tools/run_gcp_gpu_validation.sh` (the intended path around
Kaggle's perf-counter restriction, since a real GCE VM has root) was run five
times against `kiln-gpu-profiling`, across `nvidia-tesla-t4` in
`us-central1-a`, `us-east1-c`, `us-west1-b`, and `europe-west4-a`, and
`nvidia-tesla-p100` in `us-central1-a` and `us-central1-c`. Every attempt
failed at `gcloud compute instances create` with `ZONE_RESOURCE_POOL_EXHAUSTED`
(one also hit "accelerator type not found" for P100 in a zone that doesn't
offer it) — the project has real GPU quota (verified via `gcloud compute
regions describe`) and billing is enabled, but GCP had no on-demand GPU
capacity to hand out in any zone tried, at the time this was run.

**Why this is recorded here instead of silently retried forever:** none of
these attempts created a billable resource — the script's `trap cleanup`
never fires because there's nothing to clean up when `instances create`
itself fails — so this cost nothing, but it's still a real, reproducible
result worth keeping rather than quietly giving up. Nsight profiling stays a
named, honest gap until either GCP capacity frees up in some zone, a
different cloud GPU provider is used, or physical GPU access becomes
available.

## Phase 27 — three real bugs, found only by actually running the LoRA pipeline for the first time

`tools/prepare_banking77.py`, `tools/train_lora.py`, and
`tools/eval_lora_intent.py` were all written and committed but never
actually run against the real BANKING77 dataset before this pass. Running
them for real on Kaggle surfaced three separate real bugs, one per attempt:

**Bug 1 — `datasets` refuses PolyAI/banking77's legacy loading script.**
Recent `datasets` versions refuse to execute any dataset loading script at
all, as a security policy. A first attempted fix (point at HF's
auto-converted Parquet mirror via `revision="refs/convert/parquet"`) was
itself wrong -- checked via the Hub's `/refs` API, no such mirror exists
for this specific dataset (empty `"converts"` list). The real fix: the
loading script itself does nothing but download two CSV files from
PolyAI's own GitHub repo and parse them with the standard `csv` module, so
`prepare_banking77.py` now fetches those same CSVs directly -- identical
data, identical CC-BY-4.0 license, one fewer dependency on `datasets`'
internal script-execution machinery entirely.

**Bug 2 — `torch.cuda.is_available()` doesn't mean CUDA actually works.**
Kaggle assigned a Tesla P100 (compute capability 6.0). The preinstalled
PyTorch build's own warning states plainly it only ships kernels for
compute capability 7.0 and up -- `is_available()` still returns `True`,
and the first real GPU op fails with `no kernel image is available for
execution on the device`. Fixed with `tools/_torch_device.py`, which
actually attempts a trivial CUDA op and falls back to CPU with a printed
reason on failure, rather than trusting the weaker guarantee
`is_available()` provides. Real LoRA training then ran on CPU for 1000
steps and produced a real result: loss fell from ~4.5-4.9 (first five
steps) to ~1.4-2.2 (last five steps) -- genuine learning, not noise.

**Bug 3 — `eval_lora_intent.py` had no `if __name__ == "__main__":` block
at all.** `main()` was fully defined and never called anywhere in the
file. Running `python3 eval_lora_intent.py ...` therefore did *nothing*:
no output, no error, no written file, a clean exit code 0 -- the single
hardest kind of bug to notice, because there is no wrong answer to catch,
only a silently missing one. Caught by noticing the script's expected
output file simply didn't exist afterward, not by any error message.
Fixed by adding the missing entry point, wrapped in a top-level
try/except that writes any real failure's traceback to stderr and exits
non-zero, so a future real failure can't go silent the same way again.

**What I misunderstood, once per bug:** (1) that a fix for "the library
refuses to run this" generalizes across different datasets on the same
platform, when each dataset's actual mirror situation has to be checked,
not assumed; (2) that `is_available()` answers "does a CUDA device exist"
rather than the question that actually matters, "can this build of PyTorch
run a kernel on it"; (3) that a script defining `main()` and being
runnable at all implies it's also being *called* -- syntax validity and
having an entry point are two different things, and only one of them is
checked by `python -c "import ast; ast.parse(...)"` or even a successful
`import`.
