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


def test_playground_page_is_served_and_points_at_the_real_api():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    # It has to actually call the real endpoint, not a mocked one -- this
    # is a cheap, real check that the page wasn't quietly pointed
    # somewhere else during editing.
    assert "/v1/completions" in response.text


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


def test_local_status_page_reads_the_same_process_metrics():
    before = client.get("/status/data").json()
    client.post("/v1/completions", json={"prompt": "status test", "max_tokens": 3})
    after = client.get("/status/data").json()
    page = client.get("/status")

    assert after["completions_total"] == before["completions_total"] + 1
    assert after["tokens_generated_total"] == before["tokens_generated_total"] + 3
    assert after["completion_latency_count"] == before["completion_latency_count"] + 1
    assert page.status_code == 200
    assert "Local session status" in page.text
    assert "not production traffic" in page.text
    assert 'fetch("/status/data")' in page.text


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
