"""Proves the tensor-parallel sharding math (column-parallel and
row-parallel matrix multiplication, split across a simulated number of
ranks) always produces exactly the same numbers as doing the whole
computation on one "rank" at once. See docs/learning/phase-12.md for what
this does and does not prove.
"""
import numpy as np

from kiln_py.runtime.tensor_parallel_sim import (column_parallel_matmul,
                                                  row_parallel_matmul)


def test_column_parallel_matches_unsharded_for_various_rank_counts():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 16)).astype(np.float32)
    weight = rng.normal(size=(8, 16)).astype(np.float32)  # [out_features, in_features]

    expected = x @ weight.T

    for num_ranks in (1, 2, 4):
        sharded = column_parallel_matmul(x, weight, num_ranks)
        np.testing.assert_allclose(sharded, expected, rtol=1e-5, atol=1e-5)


def test_row_parallel_matches_unsharded_for_various_rank_counts():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(3, 16)).astype(np.float32)
    weight = rng.normal(size=(8, 16)).astype(np.float32)

    expected = x @ weight.T

    for num_ranks in (1, 2, 4):
        sharded = row_parallel_matmul(x, weight, num_ranks)
        np.testing.assert_allclose(sharded, expected, rtol=1e-5, atol=1e-5)


def test_column_then_row_parallel_matches_two_unsharded_layers():
    """The real pattern real models use: one column-parallel layer feeding
    into one row-parallel layer, needing exactly one combine step (the
    row-parallel step's own all-reduce) for the whole two-layer block --
    not one combine step per layer. This checks that pairing produces the
    same numbers as just running both layers normally, unsharded.
    """
    rng = np.random.default_rng(3)
    x = rng.normal(size=(2, 16)).astype(np.float32)
    w1 = rng.normal(size=(32, 16)).astype(np.float32)  # expands 16 -> 32
    w2 = rng.normal(size=(16, 32)).astype(np.float32)  # projects 32 -> 16 again

    expected_hidden = x @ w1.T
    expected_out = expected_hidden @ w2.T

    for num_ranks in (2, 4):
        hidden = column_parallel_matmul(x, w1, num_ranks)
        out = row_parallel_matmul(hidden, w2, num_ranks)
        np.testing.assert_allclose(out, expected_out, rtol=1e-4, atol=1e-4)
