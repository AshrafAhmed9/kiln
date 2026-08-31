"""Prometheus metrics for the engine: request counts and latency, exposed
at /metrics for Prometheus to scrape (see deploy/prometheus.yml). Kept
separate from the request-handling code in api/app.py so the metrics
definitions have one home, matching this project's own "one concept per
file" rule.
"""
from prometheus_client import Counter, Histogram

completions_total = Counter(
    "kiln_completions_total", "Total number of /v1/completions requests served")

completion_latency_seconds = Histogram(
    "kiln_completion_latency_seconds",
    "Time to serve one /v1/completions request, start to finish")

tokens_generated_total = Counter(
    "kiln_tokens_generated_total", "Total number of tokens generated across all requests")
