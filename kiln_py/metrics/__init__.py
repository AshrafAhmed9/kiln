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


def local_session_snapshot() -> dict[str, float]:
    """Returns only the metrics this process has genuinely recorded so far.

    Counters and histograms live inside prometheus_client objects rather than
    ordinary Python numbers. Reading their collected samples here keeps the
    status page tied to the same source as /metrics, without parsing Prometheus
    text or inventing production-style aggregates such as p99 latency.
    """
    completion_samples = {
        sample.name: sample.value
        for metric in completions_total.collect()
        for sample in metric.samples
    }
    token_samples = {
        sample.name: sample.value
        for metric in tokens_generated_total.collect()
        for sample in metric.samples
    }
    latency_samples = {
        sample.name: sample.value
        for metric in completion_latency_seconds.collect()
        for sample in metric.samples
    }
    return {
        "completions_total": completion_samples["kiln_completions_total"],
        "tokens_generated_total": token_samples["kiln_tokens_generated_total"],
        "completion_latency_count": latency_samples[
            "kiln_completion_latency_seconds_count"
        ],
        "completion_latency_seconds_sum": latency_samples[
            "kiln_completion_latency_seconds_sum"
        ],
    }
