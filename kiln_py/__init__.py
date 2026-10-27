"""Kiln's Python orchestration layer -- API and runtime driver.

This package never holds a tensor. All compute lives in the `_C` extension
built from `csrc/` and crossed via the single pybind11 boundary in
`csrc/bindings.cpp`.
"""
