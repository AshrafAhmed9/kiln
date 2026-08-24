# ADR-014: ICU for Unicode-equivalent ByteLevel pre-tokenization

**Status:** Accepted

## Context

Kiln's byte-level BPE merge loop is hand-written, but Hugging Face's
ByteLevel tokenizer decides BPE boundaries with Unicode character classes:
letters, numbers, punctuation, and whitespace. The earlier ASCII-only
approximation matched 4,749 of a 10,000-string reference fixture and split
non-ASCII text incorrectly.

## Decision

Use ICU's C++ regular-expression engine only for pre-token boundary
selection. The BPE byte mapping, vocabulary loading, merge ranking, and
decode path remain Kiln code. ICU is linked through CMake as `ICU::i18n` and
`ICU::uc`.

## Consequences

The same seeded SmolLM2 fixture now matches 10,000/10,000 strings. This adds
a native dependency, but hand-implementing Unicode property tables and a
regex engine would not teach an inference-engine concept and would be less
reliable than the standard Unicode implementation. It does not add a tensor
framework or change the served-model boundary.
