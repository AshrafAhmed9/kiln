"""Tests for the multi-tenant control plane (Phase 16). The two tests the
plan specifically calls for as this phase's Definition of Done: two
tenants hammering the API cannot affect each other's quota or data, and a
leaked-key revocation drill actually works.
"""
from fastapi.testclient import TestClient

from kiln_py.control_plane.app import app
from kiln_py.control_plane.keys import hash_api_key
from kiln_py.control_plane.quota import (QuotaExceeded, RateLimitExceeded,
                                          check_rate_limit,
                                          consume_daily_token_budget)
from kiln_py.control_plane.tenant_store import Tenant, TenantStore

client = TestClient(app)


def _create_tenant(name: str, tokens_per_day: int = 100_000,
                    requests_per_second: float = 100.0) -> str:
    response = client.post("/admin/tenants", json={
        "name": name,
        "tokens_per_day": tokens_per_day,
        "requests_per_second": requests_per_second,
    })
    assert response.status_code == 200
    return response.json()["api_key"]


def test_api_key_is_never_returned_or_stored_raw_after_creation():
    store = TenantStore()
    raw_key = store.create_tenant("acme", 1000, 10)
    # The store must only be holding the HASH, not the raw key -- looking
    # up by the raw key's hash is the only supported path back to it.
    tenant = store.authenticate(raw_key)
    assert tenant is not None
    assert tenant.hashed_key == hash_api_key(raw_key)
    assert tenant.hashed_key != raw_key


def test_request_without_a_key_is_rejected():
    response = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 2})
    assert response.status_code == 401


def test_request_with_an_unknown_key_is_rejected():
    response = client.post("/v1/completions",
                           json={"prompt": "hi", "max_tokens": 2},
                           headers={"Authorization": "Bearer not-a-real-key"})
    assert response.status_code == 401


def test_valid_key_can_complete_and_meter_usage():
    api_key = _create_tenant("tenant-a")
    response = client.post("/v1/completions",
                           json={"prompt": "hello", "max_tokens": 3},
                           headers={"Authorization": f"Bearer {api_key}"})
    assert response.status_code == 200

    usage = client.get("/admin/usage",
                       headers={"Authorization": f"Bearer {api_key}"}).json()
    assert usage["total_requests"] == 1
    assert usage["total_tokens_out"] == 3


def test_two_tenants_hammering_the_api_cannot_affect_each_others_quota_or_usage():
    """The Phase 16 Definition of Done, stated directly: nothing tenant A
    does can move tenant B's numbers, even under real, repeated load.
    """
    key_a = _create_tenant("tenant-hammer-a", tokens_per_day=50)
    key_b = _create_tenant("tenant-hammer-b", tokens_per_day=50)

    # Hammer tenant A until ITS OWN quota is exhausted.
    for _ in range(20):
        response = client.post("/v1/completions",
                               json={"prompt": "x", "max_tokens": 2},
                               headers={"Authorization": f"Bearer {key_a}"})
        if response.status_code == 429:
            break

    usage_a = client.get("/admin/usage",
                         headers={"Authorization": f"Bearer {key_a}"}).json()
    usage_b = client.get("/admin/usage",
                         headers={"Authorization": f"Bearer {key_b}"}).json()

    # Tenant A's own quota got consumed by the hammering...
    assert usage_a["tokens_used_today"] > 0
    # ...but tenant B, who made zero requests, must show exactly zero
    # usage -- not a fraction of A's, not corrupted, exactly zero.
    assert usage_b["total_requests"] == 0
    assert usage_b["tokens_used_today"] == 0


def test_leaked_key_revocation_drill():
    """A tenant's key is compromised; it gets revoked; the compromised key
    must stop working immediately, on the very next request.
    """
    api_key = _create_tenant("tenant-to-revoke")

    working_response = client.post("/v1/completions",
                                   json={"prompt": "hi", "max_tokens": 1},
                                   headers={"Authorization": f"Bearer {api_key}"})
    assert working_response.status_code == 200

    revoke_response = client.post("/admin/tenants/revoke",
                                  json={"api_key": api_key})
    assert revoke_response.status_code == 200

    blocked_response = client.post("/v1/completions",
                                   json={"prompt": "hi", "max_tokens": 1},
                                   headers={"Authorization": f"Bearer {api_key}"})
    assert blocked_response.status_code == 401


def test_revoking_an_unknown_key_reports_not_found_rather_than_pretending_success():
    response = client.post("/admin/tenants/revoke",
                           json={"api_key": "kiln_never-existed"})
    assert response.status_code == 404


def test_daily_quota_is_enforced_and_fails_closed():
    tenant = Tenant(name="t", hashed_key="h", tokens_per_day=10,
                    requests_per_second=100)
    consume_daily_token_budget(tenant, 6)
    assert tenant.tokens_used_today == 6

    try:
        consume_daily_token_budget(tenant, 5)  # would push to 11 > 10
        assert False, "expected QuotaExceeded"
    except QuotaExceeded:
        pass
    # A rejected request must not have consumed anything -- fail-closed,
    # not fail-and-partially-charge.
    assert tenant.tokens_used_today == 6


def test_quota_window_resets_after_a_day_passes():
    tenant = Tenant(name="t", hashed_key="h", tokens_per_day=10,
                    requests_per_second=100, tokens_used_today=10,
                    day_started_at=0.0)
    # "now" is more than a day after day_started_at -- the window should
    # reset instead of treating this tenant as permanently exhausted.
    consume_daily_token_budget(tenant, 5, now=100_000.0)
    assert tenant.tokens_used_today == 5


def test_rate_limit_is_enforced_and_fails_closed():
    tenant = Tenant(name="t", hashed_key="h", tokens_per_day=1000,
                    requests_per_second=2)
    check_rate_limit(tenant, now=1.0)
    check_rate_limit(tenant, now=1.1)
    try:
        check_rate_limit(tenant, now=1.2)  # a 3rd request within the same second
        assert False, "expected RateLimitExceeded"
    except RateLimitExceeded:
        pass
