"""Load a PEFT q_proj adapter into a live Kiln model through the C++ merge API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

from kiln_py import _C
from tools.hf_parity import config_from_hf


def merge(model_dir: Path, adapter_dir: Path) -> _C.Model:
    manifest = json.loads((adapter_dir / "kiln-export.json").read_text())
    if manifest["target"] != "q_proj" or manifest["kiln_matrix"] != "wq":
        raise ValueError("this minimal bridge accepts only q_proj -> wq adapters")
    tensors = load_file(adapter_dir / "adapter_model.safetensors")
    model = _C.Model.load_from_safetensors(
        config_from_hf(model_dir / "config.json"), str(model_dir / "model.safetensors")
    )
    scale = float(manifest["scale"])
    for layer in range(model.config.n_layers):
        prefix = f"base_model.model.model.layers.{layer}.self_attn.q_proj"
        a = np.ascontiguousarray(tensors[prefix + ".lora_A.weight"], dtype=np.float32)
        b = np.ascontiguousarray(tensors[prefix + ".lora_B.weight"], dtype=np.float32)
        model.merge_lora_into_layer(layer, "wq", a, b, scale)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--tokens", default="1,2,3")
    args = parser.parse_args()
    tokens = np.asarray([int(token) for token in args.tokens.split(",")], dtype=np.int32)
    base = _C.Model.load_from_safetensors(
        config_from_hf(args.model_dir / "config.json"), str(args.model_dir / "model.safetensors")
    )
    before = base.forward(tokens, 1, len(tokens), None, 0, None)[-1]
    model = merge(args.model_dir, args.adapter_dir)
    logits = model.forward(tokens, 1, len(tokens), None, 0, None)[-1]
    print(json.dumps({"vocab_size": len(logits), "top_token": int(logits.argmax()),
                      "max_logit_change": float(np.abs(logits - before).max())}))
