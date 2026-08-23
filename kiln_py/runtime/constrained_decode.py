"""JSON-schema-constrained decoding: at every step, compute which raw
bytes would keep the output on a path to valid JSON matching a fixed
schema, and force every other byte's score to negative infinity before
sampling. See docs/learning/phase-20.md for why this reuses the existing
sampler rather than needing a new one.

The schema is a simple, fixed sequence of segments -- literal text the
schema dictates outright (key names, punctuation) and typed "slots" the
model actually gets to choose the content of (a string value, an integer
value). This is simpler than a full JSON grammar, and it's what
"schema-constrained" (as opposed to merely "syntax-constrained") usually
means in practice: the shape is fixed in advance, only the values vary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from kiln_py import _C

QUOTE = ord('"')
BACKSLASH = ord("\\")
# JSON strings require a backslash to start a real escape sequence -- a
# bare backslash on its own is invalid JSON. Simplest correct fix: never
# allow the model to emit one at all, rather than implementing the full
# escape-sequence grammar for a demo constrained-decoding example. This
# was a real bug, caught by actually running json.loads() on the output
# rather than eyeballing it for "looks like JSON."
PRINTABLE_NON_QUOTE = [b for b in range(0x20, 0x7F) if b not in (QUOTE, BACKSLASH)]
DIGITS = [ord(str(d)) for d in range(10)]


@dataclass
class Literal:
    text: bytes


@dataclass
class StringSlot:
    max_length: int = 40


@dataclass
class IntegerSlot:
    max_length: int = 8


Segment = Literal | StringSlot | IntegerSlot


def string_schema(*key_names: str) -> list[Segment]:
    """Builds a schema for a flat JSON object whose every field is a
    string, e.g. string_schema("name", "city") constrains output to
    exactly `{"name": "<any string>", "city": "<any string>"}`.
    """
    segments: list[Segment] = [Literal(b"{")]
    for i, key in enumerate(key_names):
        if i > 0:
            segments.append(Literal(b", "))
        segments.append(Literal(f'"{key}": "'.encode()))
        segments.append(StringSlot())
        segments.append(Literal(b'"'))
    segments.append(Literal(b"}"))
    return segments


def _allowed_next_bytes(segment: Segment, bytes_in_segment: int) -> set[int] | None:
    """Returns the set of byte values legal right now, or None to mean
    "this segment is a literal, there's no model choice at all here."
    """
    if isinstance(segment, Literal):
        return None
    if isinstance(segment, StringSlot):
        allowed = set(PRINTABLE_NON_QUOTE)
        if bytes_in_segment > 0:
            allowed.add(QUOTE)  # the model may choose to end the string here
        return allowed
    if isinstance(segment, IntegerSlot):
        allowed = set(DIGITS)
        return allowed
    raise TypeError(f"unknown segment type: {segment}")


def generate_constrained(model, schema: list[Segment], temperature: float = 1.0,
                         seed: int = 0) -> str:
    """Walks the schema segment by segment, using the real model and the
    real sampler for every byte the schema leaves up to the model, and
    emitting literal bytes directly (with no model call at all) for
    everything the schema fixes outright.
    """
    output = bytearray()
    sampler_config = _C.SamplerConfig()
    sampler_config.temperature = temperature

    for segment_index, segment in enumerate(schema):
        if isinstance(segment, Literal):
            output.extend(segment.text)
            continue

        bytes_in_segment = 0
        max_length = segment.max_length
        while True:
            tokens = np.array(list(output), dtype=np.int32)
            logits = model.forward(tokens, 1, len(tokens), None, 0, None)[-1]

            allowed = _allowed_next_bytes(segment, bytes_in_segment)
            if allowed is None:
                # _allowed_next_bytes only returns None for a Literal, and
                # this loop never runs for one (handled above, via
                # `continue`) -- this is here so the type checker doesn't
                # have to take that on faith.
                raise AssertionError("a non-literal segment produced no allowed-byte set")
            if bytes_in_segment >= max_length:
                # Force termination rather than run unboundedly -- a real
                # safety valve, not just a convenience: an unconstrained
                # "free" slot with no length cap could otherwise never
                # choose to stop.
                allowed = {QUOTE} if isinstance(segment, StringSlot) else set()

            masked_logits = np.full_like(logits, -np.inf)
            for byte_value in allowed:
                masked_logits[byte_value] = logits[byte_value]

            if isinstance(segment, IntegerSlot) and bytes_in_segment > 0:
                # An integer slot may also choose to stop once it has at
                # least one digit -- "stopping" here means simply not
                # emitting another digit, so the loop for this slot ends
                # without emitting anything further, moving on to the
                # next literal. Modeled as a synthetic extra choice.
                stop_score = logits[DIGITS].max() - 1.0  # a real, comparable score, not a magic sentinel
                masked_logits = np.append(masked_logits, stop_score)
                next_byte = _C.sample(masked_logits.astype(np.float32), sampler_config,
                                      list(output), seed + segment_index * 1000 + bytes_in_segment)
                if next_byte == len(masked_logits) - 1:
                    break
            else:
                next_byte = _C.sample(masked_logits.astype(np.float32), sampler_config,
                                      list(output), seed + segment_index * 1000 + bytes_in_segment)

            if isinstance(segment, StringSlot) and next_byte == QUOTE and bytes_in_segment > 0:
                break

            output.append(next_byte)
            bytes_in_segment += 1

    return bytes(output).decode("utf-8", errors="replace")


def is_valid_json(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False
