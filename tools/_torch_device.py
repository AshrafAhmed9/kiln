"""Picks a real, usable torch device for the LoRA training/eval tools.

Backend availability alone does not guarantee that an installed PyTorch
build can execute kernels on that backend. This was found on a Kaggle P100:
`torch.cuda.is_available()` was true although the build had dropped its
compute capability. Probe CUDA first, then Apple MPS, with one small real
operation before selecting either backend; see docs/correctness.md.
"""
from __future__ import annotations

import torch


def _can_run(device: str) -> bool:
    try:
        (torch.zeros(1, device=device) + 1).item()
        return True
    except RuntimeError as error:
        print(f"{device} device present but unusable with this PyTorch build "
              f"({error}); falling back.", flush=True)
        return False


def usable_device() -> str:
    if torch.cuda.is_available() and _can_run("cuda"):
        return "cuda"
    if torch.backends.mps.is_available() and _can_run("mps"):
        return "mps"
    return "cpu"
