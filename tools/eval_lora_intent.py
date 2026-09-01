"""Compare a base causal LM and its LoRA adapter on intent exact match."""
from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from _torch_device import usable_device


def load_records(path: Path, limit: int) -> list[dict[str, str]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
               if line]
    if limit > 0:
        records = records[:limit]
    if not records:
        raise ValueError("evaluation data must contain at least one record")
    return records


def predict(model, tokenizer, prompt: str, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                       max_length=128).to(device)
    with torch.inference_mode():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=8,
                                pad_token_id=tokenizer.eos_token_id)
    suffix = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip().lower()
    match = re.match(r"[a-z_]+", suffix)
    return match.group(0) if match else ""


def score(model, tokenizer, records: list[dict[str, str]], device: str,
          label: str) -> dict:
    correct = 0
    samples = []
    for index, record in enumerate(records):
        prediction = predict(model, tokenizer, record["prompt"], device)
        correct += prediction == record["label"]
        if len(samples) < 20:
            samples.append({"label": record["label"], "prediction": prediction})
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(f"[{label}] scored {index + 1}/{len(records)}", flush=True)
    return {"correct": correct, "total": len(records),
            "exact_match": correct / len(records), "samples": samples}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-examples", type=int, default=512)
    args = parser.parse_args()
    device = usable_device()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = load_records(args.data, args.max_examples)
    print(f"Loaded {len(records)} validation records; device={device}", flush=True)

    print("Loading base model...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    print("Scoring baseline...", flush=True)
    baseline = score(base, tokenizer, records, device, "baseline")

    print("Loading LoRA adapter...", flush=True)
    adapter = PeftModel.from_pretrained(base, args.adapter).to(device).eval()
    print("Scoring adapter...", flush=True)
    tuned = score(adapter, tokenizer, records, device, "adapter")

    args.output.write_text(json.dumps({"metric": "greedy exact-match accuracy",
                                       "baseline": baseline, "adapter": tuned},
                                      indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"baseline": baseline["exact_match"],
                      "adapter": tuned["exact_match"]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # This script was previously missing this entry point entirely --
        # `python3 eval_lora_intent.py ...` defined main() but never called
        # it, so it silently did nothing and exited 0. Writing any real
        # failure to a file (not just stdout) means a future run can't go
        # silent again the same way, regardless of what happens to a
        # process's captured log output.
        traceback.print_exc()
        sys.exit(1)
