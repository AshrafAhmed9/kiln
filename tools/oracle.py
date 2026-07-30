"""The parity oracle: dumps tokenizer output, per-layer activations, and
final logits from the reference HuggingFace model so the C++ engine has
something ground-truth to diff against. Test/tooling only -- the engine
itself never imports this or links Python (constitution: dependency
allowlist, ADR-003).
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def dump_fixture(model_name: str, prompt: str, out_dir: Path) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    ids = tokenizer(prompt, return_tensors="pt")["input_ids"]

    activations = {}

    def record(name):
        def hook(_module, _inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            activations[name] = tensor.detach().to(torch.float32)
        return hook

    handles = [
        layer.register_forward_hook(record(f"layer_{i}"))
        for i, layer in enumerate(model.model.layers)
    ]

    with torch.no_grad():
        logits = model(ids).logits[0, -1].to(torch.float32)

    for h in handles:
        h.remove()

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"input_ids": ids, "activations": activations, "logits": logits},
        out_dir / "fixture.pt",
    )
    (out_dir / "meta.json").write_text(
        json.dumps({"model": model_name, "prompt": prompt}, indent=2)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    dump_fixture(args.model, args.prompt, args.out)
