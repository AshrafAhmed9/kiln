"""The multi-tenant control plane's HTTP surface: create/revoke API keys,
serve completions gated by authentication and quota, and report usage.
Kept as its own FastAPI app, separate from kiln_py/api/app.py's plain demo
API, per constitution §6 -- but backed by the exact same engine instance,
since a control plane sits in front of an engine rather than replacing it.
"""
from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from kiln_py import _C
from kiln_py.api.app import _model, _tokenizer
from kiln_py.control_plane.quota import (QuotaExceeded, RateLimitExceeded,
                                          check_rate_limit,
                                          consume_daily_token_budget)
from kiln_py.control_plane.tenant_store import Tenant, TenantStore
from kiln_py.runtime.generate import generate

app = FastAPI(title="Kiln Control Plane")
_store = TenantStore()


class CreateTenantRequest(BaseModel):
    name: str
    tokens_per_day: int = 100_000
    requests_per_second: float = 2.0


@app.post("/admin/tenants")
def create_tenant(request: CreateTenantRequest):
    raw_key = _store.create_tenant(request.name, request.tokens_per_day,
                                    request.requests_per_second)
    # This is the ONLY time the raw key is ever returned -- only its hash
    # is kept from here on (see docs/learning/phase-16.md).
    return {"api_key": raw_key}


class RevokeRequest(BaseModel):
    api_key: str


@app.post("/admin/tenants/revoke")
def revoke_tenant(request: RevokeRequest):
    revoked = _store.revoke(request.api_key)
    if not revoked:
        raise HTTPException(status_code=404,
                            detail="unknown or already-revoked API key")
    return {"revoked": True}


def _authenticate(authorization: str | None) -> Tenant:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401,
                            detail="missing or malformed Authorization header")
    raw_key = authorization[len("Bearer "):]
    tenant = _store.authenticate(raw_key)
    if tenant is None:
        raise HTTPException(status_code=401, detail="invalid or revoked API key")
    return tenant


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = 16
    temperature: float = 1.0
    seed: int = 0


@app.post("/v1/completions")
def create_completion(request: CompletionRequest,
                       authorization: str | None = Header(default=None)):
    tenant = _authenticate(authorization)

    try:
        check_rate_limit(tenant)
    except RateLimitExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))

    prompt_tokens = _tokenizer.encode(request.prompt)
    # Reserve for the worst case (prompt plus every word this request
    # might still generate) before doing any work -- the same "reserve
    # worst-case, not current size" rule the Phase 5 scheduler uses for
    # memory, applied here to a tenant's token budget instead.
    try:
        consume_daily_token_budget(tenant, len(prompt_tokens) + request.max_tokens)
    except QuotaExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))

    sampler_config = _C.SamplerConfig()
    sampler_config.temperature = request.temperature

    text = generate(_model, _tokenizer, request.prompt, request.max_tokens,
                     sampler_config, seed=request.seed)

    tenant.total_requests += 1
    tenant.total_tokens_in += len(prompt_tokens)
    tenant.total_tokens_out += request.max_tokens

    return {
        "id": f"cmpl-{tenant.name}",
        "object": "text_completion",
        "choices": [{"text": text, "index": 0, "finish_reason": "length"}],
        "usage": {
            "prompt_tokens": len(prompt_tokens),
            "completion_tokens": request.max_tokens,
        },
    }


@app.get("/admin/usage")
def get_usage(authorization: str | None = Header(default=None)):
    tenant = _authenticate(authorization)
    return {
        "name": tenant.name,
        "total_requests": tenant.total_requests,
        "total_tokens_in": tenant.total_tokens_in,
        "total_tokens_out": tenant.total_tokens_out,
        "tokens_used_today": tenant.tokens_used_today,
        "tokens_per_day": tenant.tokens_per_day,
    }
