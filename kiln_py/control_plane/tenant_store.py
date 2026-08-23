"""In-memory tenant storage: one record per tenant, keyed by their key's
hash (never the raw key -- see keys.py). A real deployment would back this
with Postgres, per the plan; what's here is the same logic, tested on its
own, with persistence swapped out for something reproducible in tests.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from kiln_py.control_plane.keys import generate_api_key, hash_api_key


@dataclass
class Tenant:
    name: str
    hashed_key: str
    tokens_per_day: int
    requests_per_second: float
    tokens_used_today: int = 0
    day_started_at: float = field(default_factory=time.time)
    request_timestamps: list = field(default_factory=list)
    revoked: bool = False

    # Usage metering (Phase 16's other requirement): totals across this
    # tenant's whole lifetime, not just the current day's quota window --
    # a usage dashboard reads these, quota enforcement reads the fields
    # above.
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_requests: int = 0


class TenantStore:
    def __init__(self):
        self._tenants_by_hash: dict[str, Tenant] = {}

    def create_tenant(self, name: str, tokens_per_day: int,
                       requests_per_second: float) -> str:
        """Returns the raw API key -- the only time it's ever available."""
        raw_key, hashed = generate_api_key()
        self._tenants_by_hash[hashed] = Tenant(
            name=name, hashed_key=hashed, tokens_per_day=tokens_per_day,
            requests_per_second=requests_per_second)
        return raw_key

    def authenticate(self, raw_key: str) -> Tenant | None:
        """Returns the tenant for a valid, non-revoked key, or None."""
        hashed = hash_api_key(raw_key)
        tenant = self._tenants_by_hash.get(hashed)
        if tenant is None or tenant.revoked:
            return None
        return tenant

    def revoke(self, raw_key: str) -> bool:
        """Returns True if a real, not-already-revoked key was revoked."""
        hashed = hash_api_key(raw_key)
        tenant = self._tenants_by_hash.get(hashed)
        if tenant is None or tenant.revoked:
            return False
        tenant.revoked = True
        return True
