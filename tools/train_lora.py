"""Train a small, reproducible LoRA adapter outside Kiln's serving path.

This is deliberately tooling: PyTorch/PEFT create an adapter, while Kiln
continues to serve through its C++ merge primitive. The JSONL file must have a
``text`` field and should be replaced with a licensed task dataset for any
quality claim.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from _torch_device import usable_device


def load_texts(path: Path) -> list[str]:
    texts = [json.loads(line)["text"] for line in path.read_text().splitlines() if line]
    if not texts:
        raise ValueError("training data must contain at least one JSONL record")
    return texts


def train(model_name: str, data_path: Path, output: Path, steps: int,
          learning_rate: float, rank: int, seed: int) -> None:
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=rank,
        target_modules=["q_proj"], lora_dropout=0.0, bias="none",
    ))
    device = usable_device()
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    texts = load_texts(data_path)
    for step in range(steps):
        encoded = tokenizer(texts[step % len(texts)], return_tensors="pt",
                            truncation=True, max_length=128)
        ids = encoded["input_ids"].to(device)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(json.dumps({"step": step + 1, "loss": float(loss.detach())}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    (output / "kiln-export.json").write_text(json.dumps({
        "base_model": model_name, "target": "q_proj", "kiln_matrix": "wq",
        "rank": rank, "scale": 1.0, "steps": steps, "data": str(data_path),
    }, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()
    train(args.model, args.data, args.output, args.steps, args.learning_rate, args.rank, args.seed)
