"""Quota and rate-limit enforcement, checked BEFORE a request is allowed to
run -- see docs/learning/phase-16.md for why after-the-fact checking
wouldn't actually stop anything.
"""
from __future__ import annotations

import time

from kiln_py.control_plane.tenant_store import Tenant


class QuotaExceeded(Exception):
    pass


class RateLimitExceeded(Exception):
    pass


def consume_daily_token_budget(tenant: Tenant, tokens_requested: int,
                                now: float | None = None) -> None:
    """Raises QuotaExceeded and consumes nothing if this request would push
    the tenant over their daily budget; otherwise reserves the tokens.
    The daily window resets on its own the first time a check happens
    after 24 hours have passed -- no separate background job needed.
    """
    now = now if now is not None else time.time()
    if now - tenant.day_started_at >= 86400:
        tenant.tokens_used_today = 0
        tenant.day_started_at = now

    if tenant.tokens_used_today + tokens_requested > tenant.tokens_per_day:
        raise QuotaExceeded(
            f"tenant '{tenant.name}' would exceed its daily token budget "
            f"({tenant.tokens_used_today}/{tenant.tokens_per_day} used, "
            f"{tokens_requested} requested)")

    tenant.tokens_used_today += tokens_requested


def check_rate_limit(tenant: Tenant, now: float | None = None,
                      window_seconds: float = 1.0) -> None:
    """Raises RateLimitExceeded if the tenant has already made
    requests_per_second-worth of requests within the trailing window;
    otherwise records this request's timestamp.
    """
    now = now if now is not None else time.time()
    tenant.request_timestamps = [
        t for t in tenant.request_timestamps if now - t < window_seconds
    ]

    limit = tenant.requests_per_second * window_seconds
    if len(tenant.request_timestamps) >= limit:
        raise RateLimitExceeded(
            f"tenant '{tenant.name}' exceeded its rate limit "
            f"({tenant.requests_per_second} req/s)")

    tenant.request_timestamps.append(now)
