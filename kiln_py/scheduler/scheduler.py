"""The continuous-batching scheduler (Orca-style, OSDI '22). This is the
piece that decides, every single step, which requests get to run right now
-- not once per whole batch, but once per generated word. That's the whole
idea behind "continuous" batching: a request that finishes early frees up
its spot immediately, and a new request can start in that spot on the very
next step, instead of everyone waiting for the slowest sentence in the
batch to finish before anyone new can begin.

This file only decides WHO runs when -- it doesn't do any of the actual
math. It calls out to an "executor" (a real model in production, or a fake
stand-in during testing) to actually produce the next word for whichever
requests are currently running.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional


class RequestState(Enum):
    WAITING = auto()   # submitted, but hasn't been given a spot in the running batch yet
    RUNNING = auto()   # currently getting new words generated for it, one per scheduler step
    DONE = auto()       # finished -- either it hit its word limit or produced an end token
    REJECTED = auto()   # could never have fit even alone -- rejected immediately, not queued forever


@dataclass
class Request:
    """One user's request to generate text. `tokens` grows by one every
    step it spends in the RUNNING state.
    """
    request_id: int
    prompt_tokens: List[int]
    max_new_tokens: int
    tokens: List[int] = field(default_factory=list)
    state: RequestState = RequestState.WAITING

    def __post_init__(self):
        if not self.tokens:
            self.tokens = list(self.prompt_tokens)

    @property
    def new_tokens_generated(self) -> int:
        return len(self.tokens) - len(self.prompt_tokens)

    @property
    def is_finished(self) -> bool:
        return self.new_tokens_generated >= self.max_new_tokens

    @property
    def reserved_tokens(self) -> int:
        """The most space this request could ever need: its prompt plus
        every word it's still allowed to generate. A contiguous KV cache
        (the kind this project's Phase 3 uses, before Phase 8's paging)
        can't be grown after the fact, so admission has to reserve this
        worst case up front -- otherwise a request could be let in now
        and then, a few steps later, grow past the budget with nowhere
        left to put its next word.
        """
        return len(self.prompt_tokens) + self.max_new_tokens


# A stand-in for the real model during scheduler testing. Given a request,
# it must return the next word (as a token id) that request should receive.
# In production this would be swapped for something that actually calls the
# C++ model -- the scheduler's logic doesn't change either way, which is
# exactly the point of keeping "who runs when" separate from "how do we
# compute the next word."
ExecutorFn = Callable[[Request], int]
BatchExecutorFn = Callable[[List[Request]], List[int]]


class Scheduler:
    """Decides which requests run on each step, admitting new requests as
    soon as there's room and never letting the total amount of "memory"
    (measured here in token-slots, standing in for the real KV-cache bytes
    a real engine would track) go over budget. This is "admission control by
    KV-memory accounting" from the plan: we'd rather make a request wait
    than let the whole server run out of memory and crash.
    """

    def __init__(self, max_batch_tokens: int, executor: ExecutorFn | None = None,
                 batch_executor: BatchExecutorFn | None = None):
        if executor is None and batch_executor is None:
            raise ValueError("Scheduler needs an executor")
        self.max_batch_tokens = max_batch_tokens
        self.executor = executor
        self.batch_executor = batch_executor
        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.done: List[Request] = []
        self._next_id = itertools.count()

    def _tokens_reserved(self) -> int:
        """The worst-case total space every running request could still
        grow into. Checking against this (not against how much space is
        used right now) is what stops the cache from overflowing a few
        steps later, after requests have grown some more.
        """
        return sum(r.reserved_tokens for r in self.running)

    def submit(self, prompt_tokens: List[int], max_new_tokens: int) -> Request:
        """Adds a new request to the waiting line. If the request could
        never fit even by itself (its prompt plus its full allowance of new
        words is bigger than the entire budget), there's no point making it
        wait forever for room that will never exist -- so it's rejected
        immediately instead, which is what keeps this fail-closed rather
        than silently stuck (the same "reject clearly instead of hanging
        forever" idea used elsewhere in the project's auth work).
        """
        request = Request(
            request_id=next(self._next_id),
            prompt_tokens=list(prompt_tokens),
            max_new_tokens=max_new_tokens,
        )
        if request.reserved_tokens > self.max_batch_tokens:
            request.state = RequestState.REJECTED
            self.done.append(request)
            return request

        self.waiting.append(request)
        return request

    def _admit_waiting_requests(self) -> None:
        """Lets waiting requests in, one at a time, for as long as there's
        room for their full worst-case size -- first come, first served. A
        request only ever moves once it actually fits; nobody jumps the
        line ahead of an earlier request just because they're smaller.
        """
        while self.waiting:
            candidate = self.waiting[0]
            if self._tokens_reserved() + candidate.reserved_tokens > self.max_batch_tokens:
                break  # the next-in-line doesn't fit yet -- try again next step
            self.waiting.pop(0)
            candidate.state = RequestState.RUNNING
            self.running.append(candidate)

    def step(self) -> None:
        """Runs one round: every currently-running request gets exactly
        one new word, finished requests are cleared out (freeing their
        spot), and then as many waiting requests as now fit are let in.
        This "clear finished, then admit waiting" order, repeated every
        single step, is what makes the batch continuously refill itself
        instead of waiting for every request to finish at once.
        """
        if self.batch_executor is not None and self.running:
            next_tokens = self.batch_executor(self.running)
            if len(next_tokens) != len(self.running):
                raise RuntimeError("batch executor returned the wrong number of tokens")
            for request, next_token in zip(self.running, next_tokens):
                request.tokens.append(next_token)
        else:
            for request in self.running:
                # The constructor rejects an executor-less non-batched
                # scheduler, so this assertion only narrows the optional type.
                assert self.executor is not None
                request.tokens.append(self.executor(request))

        still_running = []
        for request in self.running:
            if request.is_finished:
                request.state = RequestState.DONE
                self.done.append(request)
            else:
                still_running.append(request)
        self.running = still_running

        self._admit_waiting_requests()

    def run_until_all_done(self, max_steps: int) -> None:
        """Keeps stepping until every submitted request has finished, or
        until max_steps is hit -- max_steps exists only so a bug that
        somehow leaves a request stuck can't hang a test forever.
        """
        for _ in range(max_steps):
            if not self.waiting and not self.running:
                return
            self.step()
