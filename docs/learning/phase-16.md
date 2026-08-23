# Phase 16 — derivation notes (multi-tenancy and the control plane)

## Why API keys are hashed, never stored raw

If a database of API keys ever leaks (a backup exposed, a misconfigured
bucket, an insider), storing the raw keys would hand every one of them,
immediately usable, to whoever got the leak. Storing only a one-way hash
of each key means a leaked database is useless on its own -- an attacker
would need to reverse the hash (computationally infeasible for a
well-chosen hash) to get a usable key back. This is the exact same
reasoning already applied once in this portfolio's Go+Java auth project
for refresh tokens, reapplied here for API keys: the raw key is shown to
the tenant exactly once, at creation time, and never again.

## Why the daily-token check happens before generation, not after

If a request were allowed to run first and only checked against the quota
afterward, a tenant could always get "one more" request through regardless
of how far over budget they already are -- the check would never actually
stop anything, just record it too late to matter. Checking (and reserving)
quota *before* doing the expensive work is the same "admission control,
not an afterthought" principle the Phase 5 scheduler already uses for
memory: reject or defer *before* spending the resource, not after.

## Why isolation is tested explicitly, not just assumed

Two tenants sharing one server is only safe if one tenant's usage can
never affect another's quota, and one tenant's data never leaks into
another's response. Because these two tenants run through the exact same
code path (the same store, the same functions), it would be easy for a
shared piece of mutable state to accidentally leak between them without
anyone noticing during normal development -- which is exactly why this is
tested directly (hammer tenant A's quota, confirm tenant B's is
completely untouched), rather than left as "should be fine because the
code looks like it keeps them separate."

## Honest scope

This is a real, working implementation of API keys, quotas, rate limits,
and usage metering, matching the plan's constitution §6 split (a separate
Python service, not woven into the compute engine). What it doesn't
include: a persistent database (Postgres, as the plan specifies) --
tenant and usage state lives in memory for this session, which is
sufficient to prove the *logic* is correct but would need a real database
layer for actual production use; and real abuse-control hooks
(content-policy filtering) beyond basic input-length caps, since building
a real content classifier is its own project.
