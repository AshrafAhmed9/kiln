"""Tests for the continuous-batching scheduler. These don't use the real
model at all -- a fake "executor" stands in for it, since what we're
testing here is the scheduling *policy* (who runs when, does memory ever
overflow), not the math. That's the whole benefit of keeping the scheduler
and the model as two separate, swappable pieces (constitution §6): we can
throw thousands of made-up requests at the scheduler in a fraction of a
second, with no real model and no GPU needed at all.
"""
import random

import pytest

from kiln_py.scheduler.scheduler import Request, RequestState, Scheduler


def fake_executor(request: Request) -> int:
    """A stand-in for the real model: just returns some made-up next word.
    It doesn't need to be realistic -- it only needs to always return an
    answer, so the scheduler has something to attach to each request.
    """
    return (len(request.tokens) * 7 + request.request_id) % 1000


def test_single_request_runs_to_completion():
    scheduler = Scheduler(max_batch_tokens=100, executor=fake_executor)
    request = scheduler.submit(prompt_tokens=[1, 2, 3], max_new_tokens=5)

    scheduler.run_until_all_done(max_steps=20)

    assert request.state == RequestState.DONE
    assert request.new_tokens_generated == 5


def test_request_too_big_to_ever_fit_is_rejected_immediately():
    scheduler = Scheduler(max_batch_tokens=10, executor=fake_executor)
    # This request could never fit even alone (3 prompt words + 20 new
    # words is bigger than the entire 10-word budget) -- it must be turned
    # away immediately, not left waiting forever for room that will never
    # exist.
    request = scheduler.submit(prompt_tokens=[1, 2, 3], max_new_tokens=20)

    assert request.state == RequestState.REJECTED
    assert request not in scheduler.waiting
    assert request not in scheduler.running


@pytest.mark.parametrize("max_new_tokens", [0, -1])
def test_nonpositive_token_budget_is_rejected(max_new_tokens):
    scheduler = Scheduler(max_batch_tokens=10, executor=fake_executor)

    with pytest.raises(ValueError, match="must be positive"):
        scheduler.submit(prompt_tokens=[1], max_new_tokens=max_new_tokens)


def test_a_finishing_request_frees_room_for_a_waiting_one():
    scheduler = Scheduler(max_batch_tokens=6, executor=fake_executor)
    # This first request uses up the whole budget by itself (3 prompt + 3
    # new = 6, exactly the limit), so a second one has to wait.
    first = scheduler.submit(prompt_tokens=[1, 2, 3], max_new_tokens=3)
    second = scheduler.submit(prompt_tokens=[4, 5], max_new_tokens=1)

    scheduler.step()
    assert first.state == RequestState.RUNNING
    assert second.state == RequestState.WAITING  # no room yet

    scheduler.run_until_all_done(max_steps=20)

    # Once the first request finished and gave back its room, the second
    # one should have been let in and finished too -- nobody gets stuck
    # forever just because they arrived second.
    assert first.state == RequestState.DONE
    assert second.state == RequestState.DONE


def test_batch_executor_advances_running_requests_together():
    calls = []

    def batch_executor(requests: list[Request]) -> list[int]:
        calls.append([request.request_id for request in requests])
        return [request.request_id + 10 for request in requests]

    scheduler = Scheduler(max_batch_tokens=20, batch_executor=batch_executor)
    first = scheduler.submit(prompt_tokens=[1], max_new_tokens=1)
    second = scheduler.submit(prompt_tokens=[2], max_new_tokens=1)

    scheduler.step()  # admits both requests
    scheduler.step()  # advances both in one batch-executor call

    assert calls == [[first.request_id, second.request_id]]
    assert first.tokens[-1] == 10
    assert second.tokens[-1] == 11


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_memory_budget_is_never_exceeded_under_random_arrivals(seed):
    """Throws a batch of randomly-sized, randomly-arriving requests at the
    scheduler (a "Poisson arrival" pattern just means requests show up at
    random, somewhat clustered times, which is a realistic stand-in for
    real traffic) and checks, after every single step, that we never
    reserved more room than the budget allows. If this test ever failed,
    it would mean a real server using this scheduler could crash from
    running out of memory -- which is exactly the failure this accounting
    exists to prevent.
    """
    rng = random.Random(seed)
    max_batch_tokens = 50
    scheduler = Scheduler(max_batch_tokens=max_batch_tokens, executor=fake_executor)

    pending_arrivals = []
    current_step = 0
    for _ in range(20):
        current_step += rng.randint(0, 3)  # random gap before the next arrival
        prompt_len = rng.randint(1, 8)
        max_new = rng.randint(1, 8)
        pending_arrivals.append((current_step, prompt_len, max_new))

    for step in range(current_step + 30):
        while pending_arrivals and pending_arrivals[0][0] == step:
            _, prompt_len, max_new = pending_arrivals.pop(0)
            scheduler.submit(prompt_tokens=list(range(prompt_len)),
                              max_new_tokens=max_new)

        if scheduler.running or scheduler.waiting:
            scheduler.step()

        assert scheduler._tokens_reserved() <= max_batch_tokens

    # Every request must have ended up somewhere final -- either it
    # finished, or it was rejected outright. None should be silently lost
    # partway through (still WAITING or RUNNING forever).
    for request in scheduler.done:
        assert request.state in (RequestState.DONE, RequestState.REJECTED)
    assert not scheduler.waiting
    assert not scheduler.running
