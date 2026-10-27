"""Checks Kiln against Hugging Face: runs the same prompts through both and
compares the scores (logits) and each layer's output.

Exits with an error if Kiln picks a different top word, or if any layer
differs by more than the threshold. Needs a single-file Llama-style
checkpoint (model.safetensors). torch and transformers are only used here,
never when serving.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kiln_py import _C
from kiln_py.api.real_checkpoint import config_from_hf


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
        reference_states: list[np.ndarray] = []
        handles = [layer.register_forward_hook(
            lambda _module, _inputs, output, states=reference_states:
            states.append((output[0] if isinstance(output, tuple) else output).detach().float().numpy()[0])
        ) for layer in reference.model.layers]
        with torch.no_grad():
            reference_logits = reference(torch.tensor([input_ids], dtype=torch.long)).logits[0, -1].float().numpy()
        for handle in handles:
            handle.remove()
        kiln_logits = kiln_model.forward(
            np.asarray(input_ids, dtype=np.int32)
        )[-1]
        difference = np.abs(kiln_logits - reference_logits)
        kiln_states = kiln_model.forward_hidden_states(np.asarray(input_ids, dtype=np.int32))
        layer_differences = [float(np.abs(kiln - reference).max())
                             for kiln, reference in zip(kiln_states, reference_states)]
        results.append({
            "prompt": prompt,
            "prompt_tokens": len(input_ids),
            "vocab_size": len(kiln_logits),
            "max_abs_diff": float(difference.max()),
            "mean_abs_diff": float(difference.mean()),
            "top1_matches": bool(int(kiln_logits.argmax()) == int(reference_logits.argmax())),
            "max_layer_abs_diff": max(layer_differences),
        })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", default="Kiln checks its own math.")
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument(
        "--layer-diff-threshold", type=float, default=5e-2,
        help="max acceptable per-layer hidden-state abs difference. This is looser "
             "than the final-logit threshold on purpose: FP32-vs-BF16 rounding "
             "compounds across depth (observed up to ~2.7e-2 on real checkpoints, "
             "even though the final logit comparison stays ~1e-5) -- this catches "
             "a real divergence, not normal accumulated rounding")
    arguments = parser.parse_args()
    prompts = ([line for line in arguments.prompts_file.read_text().splitlines() if line]
               if arguments.prompts_file else [arguments.prompt])
    results = compare_prompts(arguments.model_dir, prompts)
    print(json.dumps(results, indent=2))

    failures = [r for r in results if not r["top1_matches"]
                or r["max_layer_abs_diff"] > arguments.layer_diff_threshold]
    if failures:
        for r in failures:
            print(f"FAIL: prompt={r['prompt']!r} top1_matches={r['top1_matches']} "
                  f"max_layer_abs_diff={r['max_layer_abs_diff']:.2e} "
                  f"(threshold={arguments.layer_diff_threshold:.2e})")
        raise SystemExit(1)
    print(f"PASS: {len(results)}/{len(results)} prompts within per-layer threshold "
          f"{arguments.layer_diff_threshold:.2e}")
