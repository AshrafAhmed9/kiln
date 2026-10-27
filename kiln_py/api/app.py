"""The web API: the front door of Kiln. It accepts requests in the same
format as OpenAI's /v1/completions, so tools built for OpenAI (like the
`openai` Python package) work unchanged. This file only takes requests in and
sends answers out; the generating happens in kiln_py.runtime.generate, which
calls the C++ model.

Two backends, picked at startup by environment variable. Both serve one
request at a time:

- Default (no env vars set): a small, randomly-initialized toy model with a
  byte-level placeholder tokenizer, so the API and its tests run without a
  checkpoint. The shape of every request and response is real; the words
  the model produces are not.
- KILN_CHECKPOINT_DIR=<path to a real Llama-family HF checkpoint> (and
  optionally KILN_DEVICE=cuda): loads real trained weights and a real BPE
  tokenizer via kiln_py.api.real_checkpoint, on CPU or GPU depending on
  KILN_DEVICE.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from kiln_py import _C
from kiln_py.api.real_checkpoint import CompletionService, load_real_model_and_tokenizer
from kiln_py.runtime.byte_tokenizer import write_byte_level_tokenizer_json

app = FastAPI(title="Kiln", description="An LLM inference engine, built from scratch.")


def _build_toy_model_and_tokenizer():
    """The small random-weight model and byte-level tokenizer served when no
    checkpoint is set."""
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


_CHECKPOINT_DIR = os.environ.get("KILN_CHECKPOINT_DIR")
_DEVICE = os.environ.get("KILN_DEVICE", "cpu")

if _CHECKPOINT_DIR:
    _model, _tokenizer = load_real_model_and_tokenizer(Path(_CHECKPOINT_DIR), _DEVICE)
    _MODEL_NAME = f"kiln-real:{Path(_CHECKPOINT_DIR).name}:{_DEVICE}"
else:
    _MODEL_NAME = "kiln-toy"
    _model, _tokenizer = _build_toy_model_and_tokenizer()
_service = CompletionService(_model, _tokenizer)


class CompletionRequest(BaseModel):
    model: str = _MODEL_NAME
    prompt: str
    max_tokens: int = Field(default=16, gt=0)
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

    text = _service.complete(
        request.prompt, request.max_tokens, sampler_config, request.seed
    )

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
    for next_token in _service.stream(
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

