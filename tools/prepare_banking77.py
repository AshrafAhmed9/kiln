"""Prepare a licensed BANKING77 intent-classification LoRA dataset.

BANKING77 is published by PolyAI under CC-BY-4.0. This tool turns its English
utterances into causal-LM instruction records with an exact-match intent label,
preserving a manifest so a later adapter result is reproducible.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import urllib.request
from pathlib import Path


DATASET_ID = "PolyAI/banking77"
DATASET_LICENSE = "CC-BY-4.0"

# PolyAI/banking77 on the Hugging Face Hub ships only a legacy loading
# script (banking77.py) -- no Parquet mirror exists for it (checked via
# the Hub's /refs API: an empty "converts" list), and current `datasets`
# versions refuse to execute any loading script at all, as a security
# policy, not a version this project can pin around. The script itself
# does nothing but download these same two CSV files from PolyAI's
# original GitHub repo and parse them with the standard library `csv`
# module -- so fetching them directly, the same way the script already
# does internally, is the real fix, not a workaround: identical data,
# identical CC-BY-4.0 license, one fewer moving dependency. Found only by
# actually running this tool against the real dataset; see
# docs/correctness.md.
_TRAIN_URL = ("https://raw.githubusercontent.com/PolyAI-LDN/"
             "task-specific-datasets/master/banking_data/train.csv")
_TEST_URL = ("https://raw.githubusercontent.com/PolyAI-LDN/"
            "task-specific-datasets/master/banking_data/test.csv")


def fetch_csv_rows(url: str) -> list[tuple[str, str]]:
    with urllib.request.urlopen(url) as response:
        text = response.read().decode("utf-8")
    reader = csv.reader(io.StringIO(text), quotechar='"', delimiter=",",
                        quoting=csv.QUOTE_ALL, skipinitialspace=True)
    next(reader)  # header row
    return [(row[0], row[1]) for row in reader]


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


def records_from_rows(rows: list[tuple[str, str]], limit: int,
                      seed: int) -> list[dict[str, str]]:
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [make_record(rows[index][0], rows[index][1]) for index in indices[:limit]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-examples", type=int, default=10000)
    parser.add_argument("--validation-examples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    if args.train_examples <= 0 or args.validation_examples <= 0:
        raise ValueError("example counts must be positive")

    train_rows = fetch_csv_rows(_TRAIN_URL)
    test_rows = fetch_csv_rows(_TEST_URL)
    train = records_from_rows(train_rows, args.train_examples, args.seed)
    validation = records_from_rows(test_rows, args.validation_examples, args.seed + 1)
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
