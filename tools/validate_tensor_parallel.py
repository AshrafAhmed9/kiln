"""Validate real two-rank tensor-parallel GEMM math with NCCL.

This is intentionally a focused primitive check, not a claim that Kiln's
whole transformer is tensor-parallel yet. It proves the two sharding patterns
the executor will use: output-column shards gathered into a full activation,
and input-row shards summed with NCCL.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist


def require_divisible(value: int, world_size: int, name: str) -> None:
    if value % world_size:
        raise ValueError(f"{name}={value} is not divisible by world_size={world_size}")


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    try:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.manual_seed(47)

        tokens, hidden, ffn = 5, 16, 32
        require_divisible(ffn, world_size, "ffn")
        require_divisible(hidden, world_size, "hidden")

        # Column parallelism: every rank multiplies the shared input by only
        # its own output rows, then all ranks collect the independent slices.
        x = torch.randn(tokens, hidden, device=device)
        column_weight = torch.randn(ffn, hidden, device=device)
        column_shard = column_weight.chunk(world_size, dim=0)[local_rank]
        local_column = x @ column_shard.T
        gathered_column = [torch.empty_like(local_column) for _ in range(world_size)]
        dist.all_gather(gathered_column, local_column)
        actual_column = torch.cat(gathered_column, dim=1)
        expected_column = x @ column_weight.T
        torch.testing.assert_close(actual_column, expected_column, rtol=1e-5, atol=1e-5)

        # Row parallelism: each rank owns matching input columns and can only
        # form a partial output. NCCL all-reduce is the required sync point.
        ffn_input = torch.randn(tokens, ffn, device=device)
        row_weight = torch.randn(hidden, ffn, device=device)
        input_shard = ffn_input.chunk(world_size, dim=1)[local_rank]
        weight_shard = row_weight.chunk(world_size, dim=1)[local_rank]
        actual_row = input_shard @ weight_shard.T
        dist.all_reduce(actual_row, op=dist.ReduceOp.SUM)
        expected_row = ffn_input @ row_weight.T
        torch.testing.assert_close(actual_row, expected_row, rtol=1e-5, atol=1e-5)

        if local_rank == 0:
            names = [torch.cuda.get_device_name(rank) for rank in range(world_size)]
            print(
                "Tensor-parallel GEMM validation passed: "
                f"column gather and row all-reduce across {world_size} ranks: "
                f"{', '.join(names)}"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
