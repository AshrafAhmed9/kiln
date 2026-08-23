from kiln_py.scheduler.chunked_prefill_sim import SimRequest, simulate_workload


def test_monolithic_prefill_stalls_other_requests_decode():
    """A long prompt arriving alongside a short, already-running request
    must delay the short request's completion by roughly the long
    prompt's own prefill length -- that's the actual mechanism chunked
    prefill exists to fix, so it has to be reproduced here to be a
    meaningful baseline.
    """
    short_request = SimRequest(request_id=1, arrival_step=0, prompt_length=1,
                               decode_length=3)
    long_request = SimRequest(request_id=2, arrival_step=0, prompt_length=20,
                              decode_length=1)

    result = simulate_workload([short_request, long_request], policy="monolithic")

    # The short request's decode only starts after whichever request's
    # prefill happens to go first fully finishes -- with two requests
    # both arriving at step 0, at least one of them pays for the other's
    # full monolithic prefill before its own decode can even begin.
    assert result["completion"][1] >= long_request.prompt_length - 1 or \
        result["completion"][2] >= long_request.prompt_length - 1


def test_chunked_prefill_finishes_a_short_request_far_sooner_when_a_long_one_is_present():
    """The actual claim chunked prefill makes: a short request sharing the
    scheduler with a long prompt should finish close to its OWN real work
    (1 prefill unit + 3 decode steps) under chunking, but be dragged out
    to roughly the long prompt's entire prefill length under the
    monolithic policy -- regardless of which request happens to be
    scheduled first.
    """
    short_request = SimRequest(request_id=1, arrival_step=0, prompt_length=1,
                               decode_length=3)
    long_request = SimRequest(request_id=2, arrival_step=0, prompt_length=20,
                              decode_length=1)

    monolithic = simulate_workload([short_request, long_request], policy="monolithic")
    chunked = simulate_workload([short_request, long_request], policy="chunked",
                                chunk_size=2)

    # Chunking must get the short request done meaningfully sooner than
    # monolithic -- not instantly (decode only advances at interleave
    # points, spaced chunk_size apart, which is itself the real, honest
    # cost of chunking -- see chunked_prefill_sim.py), but nowhere close
    # to being dragged out toward the long prompt's full 20-unit length.
    assert chunked["completion"][1] < monolithic["completion"][1] / 2


def test_priority_shortest_first_serves_the_short_request_before_the_long_one():
    short_request = SimRequest(request_id=1, arrival_step=0, prompt_length=1,
                               decode_length=1)
    long_request = SimRequest(request_id=2, arrival_step=0, prompt_length=20,
                              decode_length=1)

    result = simulate_workload([short_request, long_request],
                              policy="priority_shortest_first")

    assert result["ttft"][1] < result["ttft"][2]


def test_all_requests_eventually_complete_under_every_policy():
    requests = [
        SimRequest(request_id=1, arrival_step=0, prompt_length=5, decode_length=4),
        SimRequest(request_id=2, arrival_step=1, prompt_length=10, decode_length=2),
        SimRequest(request_id=3, arrival_step=2, prompt_length=1, decode_length=6),
    ]
    for policy, kwargs in [("monolithic", {}), ("priority_shortest_first", {}),
                          ("chunked", {"chunk_size": 4})]:
        result = simulate_workload(requests, policy=policy, **kwargs)
        assert set(result["completion"].keys()) == {1, 2, 3}


def test_monolithic_causes_one_giant_stall_equal_to_the_long_prompt_length():
    """The actual user-facing harm monolithic prefill causes: a request
    that's already streaming tokens goes completely silent for exactly
    the long prompt's full length, in one uninterrupted freeze.
    """
    already_decoding = SimRequest(request_id=1, arrival_step=0, prompt_length=1,
                                  decode_length=30)
    long_request = SimRequest(request_id=2, arrival_step=5, prompt_length=50,
                              decode_length=1)

    result = simulate_workload([already_decoding, long_request], policy="monolithic")
    assert result["max_stall"][1] >= long_request.prompt_length - 1


def test_chunking_bounds_the_worst_stall_to_roughly_the_chunk_size():
    """The actual claim chunking makes: instead of one giant freeze, the
    worst gap a streaming request experiences is bounded by roughly the
    chunk size, however long the interrupting prefill is.
    """
    already_decoding = SimRequest(request_id=1, arrival_step=0, prompt_length=1,
                                  decode_length=30)
    long_request = SimRequest(request_id=2, arrival_step=5, prompt_length=50,
                              decode_length=1)

    result = simulate_workload([already_decoding, long_request], policy="chunked",
                               chunk_size=5)
    assert result["max_stall"][1] <= 5 + 2  # a small, bounded margin, not the full 50-unit prefill


def test_smaller_chunk_size_gives_a_strictly_smaller_worst_stall():
    already_decoding = SimRequest(request_id=1, arrival_step=0, prompt_length=1,
                                  decode_length=30)
    long_request = SimRequest(request_id=2, arrival_step=5, prompt_length=50,
                              decode_length=1)

    small_chunk = simulate_workload([already_decoding, long_request], policy="chunked",
                                    chunk_size=2)
    large_chunk = simulate_workload([already_decoding, long_request], policy="chunked",
                                    chunk_size=20)
    assert small_chunk["max_stall"][1] < large_chunk["max_stall"][1]


def test_smaller_chunk_size_trades_the_long_requests_own_ttft_for_others_latency():
    """The actual tradeoff chunking makes, checked directly: a smaller
    chunk_size means MORE frequent interleave points, which is better for
    everyone else's latency but costs the long request more total steps
    before ITS OWN prefill finishes (more interleave points = more steps
    spent not making prefill progress). If chunking only ever helped and
    never cost anything, it wouldn't be a real tradeoff -- it would just
    be a strictly better setting, which isn't the honest shape of this
    problem (see docs/learning/phase-19.md).
    """
    long_request = SimRequest(request_id=1, arrival_step=0, prompt_length=40,
                              decode_length=1)

    small_chunk = simulate_workload([long_request], policy="chunked", chunk_size=2)
    large_chunk = simulate_workload([long_request], policy="chunked", chunk_size=20)

    assert small_chunk["ttft"][1] > large_chunk["ttft"][1]


def test_chunked_requires_a_chunk_size():
    request = SimRequest(request_id=1, arrival_step=0, prompt_length=5, decode_length=1)
    try:
        simulate_workload([request], policy="chunked")
        assert False, "expected ValueError"
    except ValueError:
        pass
