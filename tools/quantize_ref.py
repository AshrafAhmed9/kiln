"""A plain Python re-implementation of the INT8 per-channel quantizer in
csrc/quant/quantize.cpp, kept intentionally separate from the C++ code
(not calling into it at all) so it can be used as an independent
cross-check -- the same "don't just trust your own implementation, compare
it against something written independently" idea the parity harness
applies to the model's numerics, applied here to the quantizer itself.
"""
from __future__ import annotations

import numpy as np


def quantize_int8_per_channel(weights: np.ndarray):
    """weights is [rows, cols]. Returns (quantized_int8, scales) exactly
    the way csrc/quant/quantize.cpp's QuantizeInt8PerChannel does: one
    scale per row, chosen so the row's single largest weight maps to the
    largest representable integer step (127).
    """
    rows, cols = weights.shape
    max_abs = np.max(np.abs(weights), axis=1)
    scales = np.where(max_abs > 0, max_abs / 127.0, 1.0)

    quantized = np.zeros((rows, cols), dtype=np.int8)
    for r in range(rows):
        steps = np.round(weights[r] / scales[r])
        steps = np.clip(steps, -127, 127)
        quantized[r] = steps.astype(np.int8)

    return quantized, scales.astype(np.float32)


def dequantize_int8_per_channel(quantized: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return quantized.astype(np.float32) * scales[:, None]
