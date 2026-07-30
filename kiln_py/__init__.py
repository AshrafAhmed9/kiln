"""Kiln's Python orchestration layer -- API, scheduler, runtime driver.

Constitution §6: this package holds request objects, block tables, and
schedules. It never holds a tensor. All compute lives in the `_C` extension
built from `csrc/` and crossed via the single pybind11 boundary in
`csrc/bindings.cpp`.
"""
