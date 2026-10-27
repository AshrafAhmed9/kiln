"""A separate Python version of the INT8 quantizer in csrc/quant/quantize.cpp.
It does not call the C++ code, so the tests can check the C++ quantizer
against an independent implementation.
"""
from __future__ import annotations

import numpy as np


def quantize_int8_per_channel(weights: np.ndarray):
    """weights is [rows, cols]. Returns (int8 values, one scale per row).
    Each row's scale is picked so its largest weight becomes 127.
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
