"""Tests for the HTTP API. These exercise the actual FastAPI app -- real
request objects going in, real responses coming out -- but through an
in-process test client rather than a real network socket, since that's
faster and doesn't need a port to be free. The model behind these tests is
the small, randomly-initialized one described in app.py: these tests check
that the request/response *shapes* are correct and that generation runs to
completion without crashing, not that the generated words make sense.
"""
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
