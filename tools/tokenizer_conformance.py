"""Measure Kiln BPE agreement with a downloaded Hugging Face tokenizer.

The corpus is deterministic and includes ASCII, whitespace, punctuation, and
Unicode. The result is evidence about this exact tokenizer and fixture set,
not a claim of universal tokenizer conformance.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from kiln_py import _C


def fixture_strings(seed: int, count: int) -> list[str]:
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    punctuation = "!?,.:;()[]{}+-=/\\\"'"
    unicode_words = ["café", "東京", "مرحبا", "🙂", "naïve"]
    fixtures = ["", "hello", " hello", "hello world", "\n\t", "café", "東京", "🙂"]
    while len(fixtures) < count:
        parts = []
        for _ in range(rng.randint(1, 8)):
            kind = rng.randrange(5)
            if kind == 0:
                parts.append("".join(rng.choice(alphabet) for _ in range(rng.randint(1, 12))))
            elif kind == 1:
                parts.append("".join(rng.choice(punctuation) for _ in range(rng.randint(1, 5))))
            elif kind == 2:
                parts.append(rng.choice([" ", "  ", "\n", "\t"]))
            elif kind == 3:
                parts.append(rng.choice(unicode_words))
            else:
                parts.append(str(rng.randrange(1_000_000)))
        fixtures.append("".join(parts))
    return fixtures[:count]


def measure(model_dir: Path, seed: int, count: int) -> dict[str, object]:
    reference = AutoTokenizer.from_pretrained(model_dir)
    kiln = _C.BpeTokenizer.load(str(model_dir / "tokenizer.json"))
    mismatches = []
    matching = 0
    for text in fixture_strings(seed, count):
        expected = reference.encode(text, add_special_tokens=False)
        actual = kiln.encode(text)
        if actual == expected:
            matching += 1
        elif len(mismatches) < 10:
            mismatches.append({"text": text, "expected": expected, "actual": actual})
    return {
        "seed": seed,
        "fixtures": count,
        "matching_fixtures": matching,
        "example_mismatches": mismatches,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--count", type=int, default=10_000)
    arguments = parser.parse_args()
    print(json.dumps(measure(arguments.model_dir, arguments.seed, arguments.count), ensure_ascii=False, indent=2))
