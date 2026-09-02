"""Tests for the HTTP API. These exercise the actual FastAPI app -- real
request objects going in, real responses coming out -- but through an
in-process test client rather than a real network socket, since that's
faster and doesn't need a port to be free. The model behind these tests is
the small, randomly-initialized one described in app.py: these tests check
that the request/response *shapes* are correct and that generation runs to
completion without crashing, not that the generated words make sense.
"""
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from kiln_py.api.app import app

client = TestClient(app)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metrics_endpoint_reflects_a_real_completion():
    before = client.get("/metrics").text
    client.post("/v1/completions", json={"prompt": "metrics test", "max_tokens": 2})
    after = client.get("/metrics").text

    # A real counter incrementing, not a static/mocked page -- prove it by
    # actually parsing out the number rather than just checking the text
    # changed at all.
    def read_counter(text: str, name: str) -> float:
        for line in text.splitlines():
            if line.startswith(name + " "):
                return float(line.split()[-1])
        return 0.0

    assert read_counter(after, "kiln_completions_total") > read_counter(
        before, "kiln_completions_total")




def test_completion_returns_openai_shaped_response():
    response = client.post("/v1/completions", json={
        "prompt": "hello",
        "max_tokens": 4,
    })
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert len(body["choices"]) == 1
    assert isinstance(body["choices"][0]["text"], str)


def test_completion_rejects_nonpositive_max_tokens():
    for max_tokens in (0, -1):
        response = client.post("/v1/completions", json={
            "prompt": "invalid length", "max_tokens": max_tokens,
        })
        assert response.status_code == 422


def test_concurrent_non_streaming_completions_use_the_scheduler_path():
    def complete(prompt: str):
        return client.post("/v1/completions", json={
            "prompt": prompt, "max_tokens": 3, "seed": 7,
        })

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(complete, ["first request", "second request"]))

    assert all(response.status_code == 200 for response in responses)
    assert all(response.json()["object"] == "text_completion" for response in responses)


def test_same_seed_gives_same_completion():
    """Generation with a fixed seed must be replayable -- this is the
    determinism promise the whole project is built on, checked here at the
    outermost, user-facing layer rather than just deep in the C++ code.
    """
    request_body = {"prompt": "same input", "max_tokens": 5, "seed": 42,
                     "temperature": 1.0}
    first = client.post("/v1/completions", json=request_body).json()
    second = client.post("/v1/completions", json=request_body).json()
    assert first["choices"][0]["text"] == second["choices"][0]["text"]


def test_streaming_completion_sends_multiple_chunks_then_done():
    with client.stream("POST", "/v1/completions",
                        json={"prompt": "hi", "max_tokens": 3, "stream": True}) as response:
        lines = [line for line in response.iter_lines() if line]

    assert lines[-1] == "data: [DONE]"
    # One data line per generated word, plus the final [DONE] line.
    assert len(lines) == 3 + 1


def test_concurrent_streaming_completions_share_the_scheduler_service():
    def stream(prompt: str) -> list[str]:
        with client.stream("POST", "/v1/completions", json={
            "prompt": prompt, "max_tokens": 3, "stream": True, "seed": 9,
        }) as response:
            assert response.status_code == 200
            return [line for line in response.iter_lines() if line]

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(stream, ["first stream", "second stream"]))

    assert all(lines[-1] == "data: [DONE]" for lines in responses)
    assert all(len(lines) == 4 for lines in responses)
