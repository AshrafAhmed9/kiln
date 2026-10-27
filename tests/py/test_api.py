"""Tests for the web API. They call the real FastAPI app through an
in-process test client (no network port needed). The model behind them is the
small random-weight one from app.py, so these tests check that requests and
responses have the right shape and that generation finishes, not that the
words make sense.
"""
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from kiln_py.api.app import app

client = TestClient(app)


def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


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


def test_concurrent_non_streaming_completions_work():
    def complete(prompt: str):
        return client.post("/v1/completions", json={
            "prompt": prompt, "max_tokens": 3, "seed": 7,
        })

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(complete, ["first request", "second request"]))

    assert all(response.status_code == 200 for response in responses)
    assert all(response.json()["object"] == "text_completion" for response in responses)


def test_same_seed_gives_same_completion():
    """The same seed must give the same text every time."""
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


def test_concurrent_streaming_completions_work():
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
