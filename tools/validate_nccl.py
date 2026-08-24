"""Confirm that a CUDA host can run Kiln's future two-rank NCCL path.

This is deliberately an environment probe, not a tensor-parallel claim. The
CPU simulation remains the implementation evidence until model tensors stay
on device and use this collective for the row-parallel combine step.

Usage:
  torchrun --standalone --nproc_per_node=2 tools/validate_nccl.py
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        value = torch.tensor([local_rank + 1], device=f"cuda:{local_rank}",
                             dtype=torch.int64)
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        expected = world_size * (world_size + 1) // 2
        if value.item() != expected:
            raise RuntimeError(
                f"NCCL all-reduce returned {value.item()}, expected {expected}"
            )
        if local_rank == 0:
            names = ", ".join(torch.cuda.get_device_name(rank)
                              for rank in range(world_size))
            print(f"NCCL all-reduce passed across {world_size} ranks: {names}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
