"""Compares three scheduling policies (FCFS/monolithic, shortest-prompt-
priority, and chunked prefill at several chunk sizes) on a real, seeded
synthetic workload -- several requests already streaming tokens, then one
long prompt arrives and needs prefilling. Produces a real
latency/throughput frontier plot from real simulated numbers (see
kiln_py/scheduler/chunked_prefill_sim.py) -- no GPU or real model is
needed for this, since what's being measured is scheduling policy, not
model math.

The metric is the WORST STALL an already-streaming request experiences
(the longest gap between two consecutive tokens it receives), not total
completion time -- an earlier version of this script used completion
time and found it barely changed across chunk sizes, which turned out to
be a real, correct property of this model (every interleave step costs
exactly one step and grants exactly one tick -- a zero-sum reallocation),
not a sign chunking doesn't matter. Stall length is what a user actually
notices; total completion time, in this model, isn't what chunking buys.

Usage: PYTHONPATH=. python3 bench/scheduler_policy_comparison.py
"""
from __future__ import annotations

from kiln_py.scheduler.chunked_prefill_sim import SimRequest, simulate_workload


def _build_workload() -> list[SimRequest]:
    """Five short requests arrive first and are already mid-decode -- then
    one long prompt (a big document, a long system prompt) arrives and
    needs to be prefilled. This is the real-world case chunked prefill
    targets: a long prefill arriving while OTHER requests are already
    generating, not merely queued behind it.

    Each short request's own prefill (prompt_length=2) only occupies the
    single shared prefill slot for 2 steps, so all five cycle through it
    and are genuinely decoding well before the long request arrives at
    step 15 -- an earlier version of this workload gave short requests a
    longer prompt_length, and since only one request can occupy the
    prefill slot at a time, most of them were STILL WAITING for their own
    first turn when the long request arrived, which hid the effect this
    benchmark exists to show.
    """
    requests = []
    for i in range(5):
        requests.append(SimRequest(request_id=i, arrival_step=0,
                                   prompt_length=2, decode_length=60))
    requests.append(SimRequest(request_id=100, arrival_step=15,
                               prompt_length=150, decode_length=5))
    return requests


def _short_request_ids(requests: list[SimRequest]) -> list[int]:
    return [r.request_id for r in requests if r.prompt_length <= 10]


def compare_policies() -> list[dict]:
    workload = _build_workload()
    short_ids = _short_request_ids(workload)
    rows = []

    def add_row(label: str, result: dict) -> None:
        stalls = [result["max_stall"][rid] for rid in short_ids]
        rows.append({
            "policy": label,
            "big_request_ttft": result["ttft"][100],
            "short_requests_worst_stall": max(stalls),
        })

    add_row("monolithic (FCFS)", simulate_workload(workload, policy="monolithic"))
    add_row("priority (shortest prompt first)",
           simulate_workload(workload, policy="priority_shortest_first"))
    for chunk_size in (2, 5, 10, 20, 50, 100):
        add_row(f"chunked (chunk_size={chunk_size})",
               simulate_workload(workload, policy="chunked", chunk_size=chunk_size))

    return rows


def plot_frontier(rows: list[dict], out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")  # no display needed -- this just saves a PNG
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    for row in rows:
        ax.scatter(row["big_request_ttft"], row["short_requests_worst_stall"],
                  s=80)
        ax.annotate(row["policy"],
                   (row["big_request_ttft"], row["short_requests_worst_stall"]),
                   textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Big request's own TTFT (steps)")
    ax.set_ylabel("Worst stall an already-streaming request experiences (steps)")
    ax.set_title("Chunked prefill: the real latency/throughput frontier\n"
                 "(seeded synthetic workload -- see bench/scheduler_policy_comparison.py)",
                pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved frontier plot to {out_path}")


if __name__ == "__main__":
    rows = compare_policies()
    for row in rows:
        print(row)
    plot_frontier(rows, "bench/scheduler_policy_frontier.png")
