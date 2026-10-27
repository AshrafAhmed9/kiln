"""Picks a torch device that really works: CUDA, then Apple MPS, then CPU.

`torch.cuda.is_available()` can say True on a GPU that this PyTorch build
cannot actually run on (this happened on a Kaggle P100), so each device is
tried with one tiny operation before it is chosen.
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
