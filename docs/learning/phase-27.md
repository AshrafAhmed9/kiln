# Phase 27 — running the LoRA pipeline for the first time, and what "written" doesn't mean

## Three tools, all committed, none ever actually run

`tools/prepare_banking77.py`, `tools/train_lora.py`, and
`tools/eval_lora_intent.py` existed before this phase but had never been run
end to end against a real dataset on real hardware. Each one had a real bug,
and each bug was invisible from reading the code alone.

## Bug 1 — a dataset loader that the library refuses to run

Recent `datasets` releases refuse to execute any dataset's Python loading
script at all, as a security policy -- `PolyAI/banking77` ships exactly that
kind of script, so `load_dataset("PolyAI/banking77")` fails immediately. The
first fix attempt (point at HF's auto-converted Parquet mirror via
`revision="refs/convert/parquet"`) assumed every dataset has one; checking the
Hub's own `/refs` API showed this one doesn't (`"converts": []`). Reading the
rejected script's own source showed it does nothing exotic -- it downloads two
CSVs from PolyAI's GitHub repo and parses them with the standard `csv` module.
`prepare_banking77.py` now does exactly that directly: same data, same
CC-BY-4.0 license, zero dependency on `datasets`' script-execution machinery.

## Bug 2 — `is_available()` answers the wrong question

`torch.cuda.is_available()` checks that a CUDA device exists. It does not
check that the *installed PyTorch build* shipped compiled kernels for that
device's specific compute capability. Kaggle assigned a Tesla P100 (compute
capability 6.0); the preinstalled PyTorch build's own startup warning says it
only ships kernels for 7.0+. `is_available()` still returned `True`, and the
first real op failed with "no kernel image is available for execution on the
device." `tools/_torch_device.py` now actually attempts a trivial op
(`torch.zeros(1, device="cuda") + 1`) and falls back to CPU with a printed
reason, rather than trusting a check that answers a narrower question than
the one that matters.

## Bug 3 — the quiet kind of bug, worth naming precisely

`eval_lora_intent.py` had no `if __name__ == "__main__":` block. `main()` was
fully defined, syntactically valid, importable, and never called by anything
in the file. Running it produced no output, no error, no file, exit code 0 --
indistinguishable from "ran and did nothing meaningful" unless you already
know to check for the output file's existence. This is the hardest class of
bug to catch precisely because there's no wrong answer to notice, only a
missing one; `ast.parse` and even a clean `import` both pass a file like
this. The fix adds the entry point plus a top-level `try/except` that prints
any real failure's traceback and exits non-zero, so a genuine future failure
can't go silent the same way again.

## The result, once all three were fixed

1000 real optimizer steps, one `q_proj` LoRA matrix, on the real BANKING77
training split: loss fell from ~4.5-5.0 (first five steps) to ~1.4-2.2 (last
five). Greedy exact-match accuracy on 300 held-out validation examples: base
model 0/300 (0.0%), adapted model 18/300 (6.0%). The honest reading: 6% on a
77-way classification task from one under-tuned matrix and zero
hyperparameter search is a small, real, above-nothing number -- not evidence
of a strong classifier. The more informative signal is qualitative: the
adapted model's wrong answers are still BANKING77-shaped label strings
(`lost_card`, `cash_fee`, `new_passcode_requested`), while the base model's
wrong answers are empty strings or single stray words (`""`, `"i"`) -- the
adapter learned the task's output format even on the majority of examples
where it didn't land on the exact right label.

## The pattern across all three bugs

Every one of them was invisible to static review and only surfaced by
actually running the code against real data on real hardware -- the same
lesson as Phase 7's re-verification and Phase 18's Docker bugs, just in a new
subsystem. "It's written and it imports cleanly" and "it does the thing it
claims to do" are different claims, and only running it settles the second
one.
