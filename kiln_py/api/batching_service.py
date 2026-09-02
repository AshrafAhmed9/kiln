"""A small worker that connects HTTP completions to the real scheduler.

Requests wait on a per-request event while one background thread advances the
scheduler. That gives concurrent normal completions a shared decode step
without making FastAPI request handlers own model state or cache lifetime.
"""
from __future__ import annotations

import threading
from queue import Queue
from typing import Iterator

from kiln_py.runtime.continuous_batch import ContinuousBatchExecutor
from kiln_py.scheduler.scheduler import Request, RequestState, Scheduler


class CompletionBatchService:
    def __init__(self, model, tokenizer, max_batch_tokens: int):
        self._tokenizer = tokenizer
        self._executor = ContinuousBatchExecutor(model)
        self._scheduler = Scheduler(
            max_batch_tokens=max_batch_tokens,
            batch_executor=self._executor,
        )
        self._condition = threading.Condition()
        self._finished: dict[int, Request] = {}
        self._failed: dict[int, BaseException] = {}
        self._waiters: dict[int, threading.Event] = {}
        self._streams: dict[int, Queue[int | BaseException | None]] = {}
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def complete(self, prompt: str, max_new_tokens: int, sampler_config,
                 seed: int) -> str:
        waiter = threading.Event()
        request = self._submit(prompt, max_new_tokens, sampler_config, seed,
                               waiter=waiter)
        waiter.wait()
        with self._condition:
            if request.request_id in self._failed:
                raise self._failed.pop(request.request_id)
            finished = self._finished.pop(request.request_id)
            self._waiters.pop(request.request_id, None)
        return self._tokenizer.decode(
            finished.tokens[len(finished.prompt_tokens):]
        ).decode("utf-8", errors="replace")

    def stream(self, prompt: str, max_new_tokens: int, sampler_config,
               seed: int) -> Iterator[int]:
        stream: Queue[int | BaseException | None] = Queue()
        request = self._submit(prompt, max_new_tokens, sampler_config, seed, stream)
        try:
            while True:
                item = stream.get()
                if item is None:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            self.cancel(request.request_id)

    def _submit(self, prompt: str, max_new_tokens: int, sampler_config,
                seed: int, stream: Queue[int | BaseException | None] | None = None,
                waiter: threading.Event | None = None) -> Request:
        prompt_tokens = self._tokenizer.encode(prompt)
        with self._condition:
            request = self._scheduler.submit(prompt_tokens, max_new_tokens)
            if request.state is RequestState.REJECTED:
                raise ValueError("request exceeds the scheduler's KV-cache budget")
            self._executor.register(request, sampler_config, seed)
            if waiter is not None:
                self._waiters[request.request_id] = waiter
            if stream is not None:
                self._streams[request.request_id] = stream
            self._condition.notify()
            return request

    def cancel(self, request_id: int) -> None:
        with self._condition:
            if self._scheduler.cancel(request_id) is not None:
                self._executor.forget(request_id)
            self._streams.pop(request_id, None)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._scheduler.waiting and not self._scheduler.running:
                    self._condition.wait()
                try:
                    active = list(self._scheduler.running)
                    self._scheduler.step()
                    for request in active:
                        stream = self._streams.get(request.request_id)
                        if stream is not None and request.new_tokens_generated:
                            stream.put(request.tokens[-1])
                    completed = list(self._scheduler.done)
                    self._scheduler.done.clear()
                    for request in completed:
                        self._executor.forget(request.request_id)
                        self._finished[request.request_id] = request
                        waiter = self._waiters.get(request.request_id)
                        if waiter is not None:
                            waiter.set()
                        stream = self._streams.get(request.request_id)
                        if stream is not None:
                            stream.put(None)
                except BaseException as error:
                    affected = self._scheduler.running + self._scheduler.waiting
                    self._scheduler.running.clear()
                    self._scheduler.waiting.clear()
                    for request in affected:
                        self._executor.forget(request.request_id)
                        self._failed[request.request_id] = error
                        waiter = self._waiters.get(request.request_id)
                        if waiter is not None:
                            waiter.set()
                        stream = self._streams.get(request.request_id)
                        if stream is not None:
                            stream.put(error)
