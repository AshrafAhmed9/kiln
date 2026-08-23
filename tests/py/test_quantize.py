"""Cross-checks the C++ INT8 quantizer (csrc/quant/quantize.cpp, called
through the pybind11 boundary) against the independent Python
reimplementation in tools/quantize_ref.py -- the parity principle applied
to the quantizer itself, not just to the model's forward pass.
"""
import numpy as np

from kiln_py import _C
from tools.quantize_ref import quantize_int8_per_channel


def test_cpp_quantizer_matches_python_reference():
    rng = np.random.default_rng(3)
    weights = rng.normal(size=(5, 16)).astype(np.float32)

    cpp_quantized, cpp_scales = _C.quantize_int8_per_channel(weights)
    ref_quantized, ref_scales = quantize_int8_per_channel(weights)

    np.testing.assert_array_equal(cpp_quantized, ref_quantized)
    np.testing.assert_allclose(cpp_scales, ref_scales, rtol=1e-6)
