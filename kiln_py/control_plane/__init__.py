"""The multi-tenant control plane: API keys, quotas, rate limits, and usage
metering. Kept as its own service, separate from the engine, per
constitution §6 -- this is product surface, not compute. See
docs/learning/phase-16.md.
"""
