"""Typed stub for the kiln._C pybind11 extension -- the §6 boundary contract,
written down. Every symbol the C++ compute layer exposes to Python must be
listed here; if it isn't, it isn't part of the contract.
"""

def ping() -> str: ...
