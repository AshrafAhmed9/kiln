"""Sanity check for the pybind11 boundary (constitution §6): the extension
builds, imports, and the one Phase 0 symbol round-trips correctly.
"""
from kiln_py import _C


def test_ping_pong():
    assert _C.ping() == "pong"
