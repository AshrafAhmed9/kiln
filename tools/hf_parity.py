"""Compare Kiln's final logits with a downloaded Hugging Face checkpoint.

This is tooling only: transformers and torch never enter Kiln's served path.
The checkpoint must be a single-file Llama-family safetensors model whose
weight names match Model.load_from_safetensors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kiln_py import _C


def config_from_hf(config_path: Path) -> object:
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "llama":
        raise ValueError(f"expected a Llama checkpoint, got {config.get('model_type')!r}")
    kiln_config = _C.ModelConfig()
    kiln_config.vocab_size = config["vocab_size"]
    kiln_config.hidden_size = config["hidden_size"]
    kiln_config.n_layers = config["num_hidden_layers"]
    kiln_config.n_heads = config["num_attention_heads"]
    kiln_config.n_kv_heads = config["num_key_value_heads"]
    kiln_config.head_dim = config.get("head_dim") or (
        kiln_config.hidden_size // kiln_config.n_heads
    )
    kiln_config.ffn_hidden = config["intermediate_size"]
    kiln_config.max_seq_len = config["max_position_embeddings"]
    kiln_config.rms_eps = config["rms_norm_eps"]
    kiln_config.rope_theta = config["rope_theta"]
    return kiln_config


def compare_prompts(model_dir: Path, prompts: list[str]) -> list[dict[str, float | bool | int | str]]:
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    reference = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float32
    ).eval()
    kiln_model = _C.Model.load_from_safetensors(
        config_from_hf(model_dir / "config.json"),
        str(model_dir / "model.safetensors"),
    )
    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        if not input_ids:
            raise ValueError("prompt must produce at least one token")
        with torch.no_grad():
            reference_logits = reference(
                torch.tensor([input_ids], dtype=torch.long)
            ).logits[0, -1].float().numpy()
        kiln_logits = kiln_model.forward(
            np.asarray(input_ids, dtype=np.int32), 1, len(input_ids), None, 0, None
        )[-1]
        difference = np.abs(kiln_logits - reference_logits)
        results.append({
            "prompt": prompt,
            "prompt_tokens": len(input_ids),
            "vocab_size": len(kiln_logits),
            "max_abs_diff": float(difference.max()),
            "mean_abs_diff": float(difference.mean()),
            "top1_matches": bool(int(kiln_logits.argmax()) == int(reference_logits.argmax())),
        })
    return results


def compare(model_dir: Path, prompt: str) -> dict[str, float | bool | int | str]:
    return compare_prompts(model_dir, [prompt])[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="Kiln checks its own math.")
    parser.add_argument("--prompts-file", type=Path)
    arguments = parser.parse_args()
    prompts = ([line for line in arguments.prompts_file.read_text().splitlines() if line]
               if arguments.prompts_file else [arguments.prompt])
    print(json.dumps(compare_prompts(arguments.model_dir, prompts), indent=2))
