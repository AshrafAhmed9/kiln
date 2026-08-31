"""Picks a real, usable torch device for the LoRA training/eval tools.

`torch.cuda.is_available()` only checks that a CUDA device exists -- it
says nothing about whether the *installed PyTorch build* actually ships
compiled kernels for that device's compute capability. PyTorch releases
regularly drop support for older GPU generations (this was found running
training for real on a Kaggle session that had a Pascal-generation P100,
compute capability 6.0, under a PyTorch build that only ships kernels for
7.0 and up) -- `is_available()` still returns True in that case, and the
first real GPU op fails with `no kernel image is available for execution
on the device`. Actually attempting one small op is the only way to know
for sure; see docs/correctness.md.
"""
from __future__ import annotations

import torch


def usable_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        (torch.zeros(1, device="cuda") + 1).item()
        return "cuda"
    except RuntimeError as error:
        print(f"CUDA device present but unusable with this PyTorch build "
              f"({error}); falling back to CPU.", flush=True)
        return "cpu"
