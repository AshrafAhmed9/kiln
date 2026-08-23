"""A discrete-step simulation of three scheduling policies, isolating the
real, well-known tradeoff chunked prefill exists to navigate: interleaving
decode into a long prefill costs a little of the long request's own
throughput (every interleave point is a step where prefill makes no
progress), in exchange for a much shorter worst-case stall for everyone
else sharing the scheduler. This is a genuine trade-off, not a free win in
either direction.

This uses the same "seeded synthetic workload, no real model needed"
approach as the Phase 5 scheduler tests -- what's being tested here is
scheduling POLICY, not model math.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimRequest:
    request_id: int
    arrival_step: int
    prompt_length: int  # how many "prefill units" this request's prompt needs
    decode_length: int  # how many new words this request will generate


def simulate_workload(requests: list[SimRequest], policy: str,
                      chunk_size: int | None = None) -> dict:
    """policy is one of:
    - "monolithic": once a request starts prefilling, it consumes
      `prompt_length` consecutive steps with NO interleaving at all --
      nobody else's decode advances during any of them.
    - "chunked": after every `chunk_size` units of prefill progress, ONE
      extra step is spent letting every currently-decoding request
      advance -- an interleave point. That interleave point is real
      overhead: it's a step where the long request's own prefill makes
      NO progress, so a smaller chunk_size means more frequent
      interleaving (shorter stalls for others) at the direct cost of more
      total steps before the long request's own prefill finishes.
    - "priority_shortest_first": like "monolithic," but when more than
      one request is waiting to START its prefill, the shortest prompt
      goes first instead of strict arrival order.

    Returns, by request_id: TTFT (the step its own prefill finished),
    completion step (the step its decode finished), and max_stall -- the
    longest gap this request ever went between two consecutive decode
    ticks. max_stall is the metric that actually matters for a user
    watching text stream in: total completion time turns out to be
    essentially invariant to chunk_size in this model (every interleave
    step costs exactly one step and grants exactly one tick to every
    decoding request -- a zero-sum reallocation, not a net win or loss),
    but WHEN those ticks arrive is exactly what chunking changes. An
    earlier version of this benchmark compared completion time across
    chunk sizes and found it barely moved regardless of chunk_size --
    which, on inspection, is a real, correct property of this model, not
    a bug, and it was the wrong metric to look at for what chunking
    actually buys you.
    """
    if policy == "chunked" and not chunk_size:
        raise ValueError("chunked policy requires a chunk_size")

    pending = sorted(requests, key=lambda r: r.arrival_step)
    prefilling: list[list] = []  # [request, prefill_remaining], queue order
    decoding: list[list] = []    # [request, decode_remaining, last_tick_step]
    active: list | None = None   # [request, remaining, units_since_interleave]
    ttft_step: dict[int, int] = {}
    completion_step: dict[int, int] = {}
    max_stall: dict[int, int] = {}

    step = 0
    max_steps = sum(2 * (r.prompt_length + r.decode_length) for r in requests) + 20
    while (pending or prefilling or decoding or active) and step < max_steps:
        while pending and pending[0].arrival_step <= step:
            request = pending.pop(0)
            prefilling.append([request, request.prompt_length])

        if policy == "priority_shortest_first":
            prefilling.sort(key=lambda entry: entry[0].prompt_length)

        if active is None and prefilling:
            request, remaining = prefilling.pop(0)
            active = [request, remaining, 0]  # [request, prefill_remaining, units_since_interleave]

        if active is not None:
            interleave_every = chunk_size if policy == "chunked" else None

            if interleave_every is not None and active[2] >= interleave_every:
                # An interleave point: spend this whole step letting
                # decode advance instead of the active prefill -- the
                # real, paid-for cost of chunking.
                _advance_decode(decoding, step, completion_step, max_stall)
                active[2] = 0
            else:
                active[1] -= 1
                active[2] += 1
                if active[1] <= 0:
                    ttft_step[active[0].request_id] = step
                    # A request's own "last tick" starts counting from the
                    # moment it enters decode -- its first real tick's gap
                    # is measured from here, not from step 0.
                    decoding.append([active[0], active[0].decode_length, step])
                    active = None
                # Monolithic and priority policies never interleave --
                # decode simply does not advance while prefill is active.
        else:
            _advance_decode(decoding, step, completion_step, max_stall)

        step += 1

    return {"ttft": ttft_step, "completion": completion_step, "max_stall": max_stall}


def _advance_decode(decoding: list[list], step: int,
                    completion_step: dict[int, int],
                    max_stall: dict[int, int]) -> None:
    still_decoding = []
    for entry in decoding:
        request, remaining, last_tick_step = entry
        gap = step - last_tick_step
        max_stall[request.request_id] = max(max_stall.get(request.request_id, 0), gap)

        remaining -= 1
        if remaining <= 0:
            completion_step[request.request_id] = step
        else:
            still_decoding.append([request, remaining, step])
    decoding[:] = still_decoding
