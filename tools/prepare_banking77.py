"""Prepare a licensed BANKING77 intent-classification LoRA dataset.

BANKING77 is published by PolyAI under CC-BY-4.0. This tool turns its English
utterances into causal-LM instruction records with an exact-match intent label,
preserving a manifest so a later adapter result is reproducible.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


DATASET_ID = "PolyAI/banking77"
DATASET_LICENSE = "CC-BY-4.0"


def make_record(utterance: str, intent: str) -> dict[str, str]:
    prompt = (
        "Classify this user's request using exactly one banking intent label.\n"
        f"Request: {utterance}\n"
        "Intent:"
    )
    return {"text": f"{prompt} {intent}\n", "prompt": prompt, "label": intent}


def write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8")


def records_from_split(split: Any, intent_names: list[str], limit: int,
                       seed: int) -> list[dict[str, str]]:
    indices = list(range(len(split)))
    random.Random(seed).shuffle(indices)
    records = []
    for index in indices[:limit]:
        row = split[index]
        records.append(make_record(row["text"], intent_names[row["label"]]))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-examples", type=int, default=10000)
    parser.add_argument("--validation-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    if args.train_examples <= 0 or args.validation_examples <= 0:
        raise ValueError("example counts must be positive")

    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID)
    intent_names = list(dataset["train"].features["label"].names)
    train = records_from_split(dataset["train"], intent_names,
                               args.train_examples, args.seed)
    validation = records_from_split(dataset["test"], intent_names,
                                    args.validation_examples, args.seed + 1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train.jsonl", train)
    write_jsonl(args.output_dir / "validation.jsonl", validation)
    (args.output_dir / "manifest.json").write_text(json.dumps({
        "dataset": DATASET_ID,
        "license": DATASET_LICENSE,
        "seed": args.seed,
        "train_examples": len(train),
        "validation_examples": len(validation),
        "task": "banking intent classification",
        "metric": "greedy exact-match accuracy on validation.jsonl",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
