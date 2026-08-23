"""Builds a plain byte-level vocabulary file for our own BPE tokenizer code
(csrc/tokenizer/bpe.cpp), with an empty merge list. That might sound like a
strange thing to want, but it's a genuinely useful special case: with no
merges at all, every single raw byte just becomes its own token, so any
text -- any language, any punctuation, even things that aren't quite valid
text -- can always be turned into tokens and back, perfectly. This is what
we use for the API demo, since we don't have a real trained vocabulary file
in this offline environment.
"""
from __future__ import annotations

import json
from pathlib import Path


def _byte_to_unicode_table() -> dict[int, str]:
    """This must build the exact same mapping as
    csrc/tokenizer/bpe.cpp's BpeTokenizer::Load -- both sides have to agree
    on which character stands for which raw byte, or encoding on one side
    and decoding on the other would silently produce garbage. It's the
    well-known GPT-2 approach: printable characters map to themselves, and
    every other byte value borrows some unused, otherwise-unused character
    to stand in for it.
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
    """Writes a tokenizer.json with one vocabulary entry per possible byte
    value (0 through 255) and no merges. Token id N always means "the raw
    byte N" -- simple, and always round-trips exactly.
    """
    byte_to_char = _byte_to_unicode_table()
    vocab = {byte_to_char[byte]: byte for byte in range(256)}
    data = {"model": {"vocab": vocab, "merges": []}}
    path.write_text(json.dumps(data))
