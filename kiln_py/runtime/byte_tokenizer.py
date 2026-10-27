"""Builds a tokenizer file where every raw byte (0-255) is its own token and
there are no merges. Any text can be turned into tokens and back exactly.

The API uses this with the random-weight toy model, which has no real
vocabulary file. Real checkpoints use their own tokenizer.json.
"""
from __future__ import annotations

import json
from pathlib import Path


def _byte_to_unicode_table() -> dict[int, str]:
    """Gives each byte a visible character to stand for it (the GPT-2
    approach): printable bytes map to themselves, and the rest borrow unused
    characters. csrc/tokenizer/bpe.cpp builds the same table, and the two
    must match.
    """
    printable = (list(range(ord("!"), ord("~") + 1)) +
                 list(range(0xA1, 0xAC + 1)) +
                 list(range(0xAE, 0xFF + 1)))
    mapping: dict[int, str] = {}
    next_extra = 256
    for byte in range(256):
        if byte in printable:
            mapping[byte] = chr(byte)
        else:
            mapping[byte] = chr(next_extra)
            next_extra += 1
    return mapping


def write_byte_level_tokenizer_json(path: Path) -> None:
    """Writes the tokenizer.json. Token id N always means the raw byte N."""
    byte_to_char = _byte_to_unicode_table()
    vocab = {byte_to_char[byte]: byte for byte in range(256)}
    data = {"model": {"vocab": vocab, "merges": []}}
    path.write_text(json.dumps(data))
