"""API key generation and hashing. The raw key is shown to a tenant exactly
once, at creation time, and never stored -- only its hash is kept, so a
leaked key database can't be used directly (see docs/learning/phase-16.md).
"""
from __future__ import annotations

import hashlib
import secrets


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Returns (raw_key, hashed_key). The caller must show raw_key to the
    tenant now and store only hashed_key -- there is no way to recover
    raw_key from hashed_key later, by design.
    """
    raw_key = "kiln_" + secrets.token_urlsafe(32)
    return raw_key, hash_api_key(raw_key)
