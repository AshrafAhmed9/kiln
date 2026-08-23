"""A CPU-only simulation of tensor-parallel sharding: proves the SPLITTING
AND RECOMBINING MATH is correct, without needing real GPUs or a real NCCL
network connection. See docs/learning/phase-12.md for why this is what can
honestly be verified in this offline, single-machine environment, and what
still can't be (real inter-GPU communication).

Uses plain numpy matmul as a stand-in for the GEMM operation, since what's
being tested here is the sharding algorithm, not matrix multiplication
itself (already tested elsewhere, in csrc/executor/gemm.cpp).
"""
from __future__ import annotations

import numpy as np


def column_parallel_matmul(x: np.ndarray, weight: np.ndarray,
                            num_ranks: int) -> np.ndarray:
    """weight is [out_features, in_features] (matching the layout the
    real model's weights use). Splits the OUTPUT side across `num_ranks`
    simulated ranks -- each rank only ever needs the full input and its
    own slice of the weight rows, and no communication between ranks is
    needed until their output slices are simply placed side by side.
    """
    out_features = weight.shape[0]
    shard_size = out_features // num_ranks

    rank_outputs = []
    for rank in range(num_ranks):
        weight_shard = weight[rank * shard_size:(rank + 1) * shard_size]
        rank_outputs.append(x @ weight_shard.T)

    return np.concatenate(rank_outputs, axis=-1)


def row_parallel_matmul(x: np.ndarray, weight: np.ndarray,
                         num_ranks: int) -> np.ndarray:
    """Splits the INPUT (contraction) side across ranks instead -- each
    rank only holds a slice of the input's columns and a matching slice of
    the weight's columns, so each rank can only produce a PARTIAL sum
    toward the true answer. Every rank's partial result has to be added
    together (an "all-reduce") to get the real output; that sum is
    simulated here as a plain local addition, standing in for what would
    be a real network operation across real GPUs.
    """
    in_features = weight.shape[1]
    shard_size = in_features // num_ranks

    partials = []
    for rank in range(num_ranks):
        x_shard = x[..., rank * shard_size:(rank + 1) * shard_size]
        weight_shard = weight[:, rank * shard_size:(rank + 1) * shard_size]
        partials.append(x_shard @ weight_shard.T)

    return np.sum(partials, axis=0)
