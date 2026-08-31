"""The OpenAI-compatible HTTP API. This is the "front door" of Kiln: it
speaks the same request/response shape as OpenAI's API, so existing tools
built for OpenAI (like the `openai` Python package) can talk to Kiln without
changes. This file only handles requests coming in and answers going out --
it doesn't do any of the actual math itself; it calls into
kiln_py.runtime.generate, which calls into the real C++ model.

Honest status: the model loaded here is randomly initialized, not trained
on real text (see docs/defense.md) -- there is no real checkpoint available
in this offline environment. So the *shape* of every request and response
below is real and correct, but the actual words the model produces are
meaningless. This is stated here in plain terms rather than hidden, in
keeping with the project's honesty rule.
"""
from __future__ import annotations

import json
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from kiln_py.metrics import (
    completion_latency_seconds,
    completions_total,
    tokens_generated_total,
)

from kiln_py import _C
from kiln_py.api.batching_service import CompletionBatchService
from kiln_py.runtime.byte_tokenizer import write_byte_level_tokenizer_json
from kiln_py.runtime.generate import generate

app = FastAPI(title="Kiln", description="An LLM inference engine, built from scratch.")

_MODEL_NAME = "kiln-toy"


def _build_toy_model_and_tokenizer():
    """Creates the small, randomly-initialized model and the byte-level
    tokenizer this demo API serves. A real deployment would instead load a
    real trained checkpoint from disk with Model.load_from_safetensors.
    """
    config = _C.ModelConfig()
    config.vocab_size = 256  # one token per possible raw byte -- see byte_tokenizer.py
    config.hidden_size = 32
    config.n_layers = 2
    config.n_heads = 4
    config.n_kv_heads = 2
    config.head_dim = 8
    config.ffn_hidden = 64
    config.max_seq_len = 512
    config.rms_eps = 1e-5
    config.rope_theta = 10000.0

    model = _C.Model.load_random(config, 1)

    tokenizer_path = Path(tempfile.gettempdir()) / "kiln_byte_tokenizer.json"
    write_byte_level_tokenizer_json(tokenizer_path)
    tokenizer = _C.BpeTokenizer.load(str(tokenizer_path))

    return model, tokenizer


_model, _tokenizer = _build_toy_model_and_tokenizer()
_completion_batches = CompletionBatchService(
    _model, _tokenizer, max_batch_tokens=_model.config.max_seq_len
)


class CompletionRequest(BaseModel):
    model: str = _MODEL_NAME
    prompt: str
    max_tokens: int = 16
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False
    seed: int = 0


def _sampler_config_from_request(request: CompletionRequest):
    config = _C.SamplerConfig()
    config.temperature = request.temperature
    config.top_p = request.top_p
    config.top_k = request.top_k
    return config


@app.post("/v1/completions")
def create_completion(request: CompletionRequest):
    """The main endpoint: send a prompt, get generated text back. Matches
    the shape of OpenAI's /v1/completions so existing OpenAI-client code
    can be pointed at Kiln with nothing but a different base URL.
    """
    sampler_config = _sampler_config_from_request(request)

    if request.stream:
        return StreamingResponse(
            _stream_completion(request, sampler_config),
            media_type="text/event-stream",
        )

    completions_total.inc()
    with completion_latency_seconds.time():
        text = _completion_batches.complete(
            request.prompt, request.max_tokens, sampler_config, request.seed
        )
    tokens_generated_total.inc(request.max_tokens)

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:16]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{"text": text, "index": 0, "finish_reason": "length"}],
    }


def _stream_completion(request: CompletionRequest, sampler_config):
    """Generates one word at a time and sends each one to the client the
    moment it's ready, instead of making the client wait for the entire
    answer before seeing anything. This is what "streaming" means here --
    each `data: ...` line below is one chunk of the answer, arriving as
    soon as it exists. Real chat apps use exactly this trick so text
    appears to type itself out live, instead of showing up all at once
    after a long pause.
    """
    for next_token in _completion_batches.stream(
        request.prompt, request.max_tokens, sampler_config, request.seed
    ):
        # Same reasoning as in kiln_py/runtime/generate.py: a byte-level
        # tokenizer can hand back a byte that isn't valid text on its own,
        # especially from this untrained demo model -- swap it for the
        # standard placeholder character rather than crash the stream.
        piece = _tokenizer.decode([next_token]).decode("utf-8", errors="replace")

        chunk = {
            "id": f"cmpl-{uuid.uuid4().hex[:16]}",
            "object": "text_completion.chunk",
            "choices": [{"text": piece, "index": 0, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    yield "data: [DONE]\n\n"


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": _MODEL_NAME}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


