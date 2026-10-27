"""The generation loop: ask the model for the next word, add it to the text,
and repeat until the answer is long enough. It lives in Python because it is
just a loop; the heavy math happens inside the C++ model.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from kiln_py import _C


def generate_stream(model, tokenizer, prompt: str, max_new_tokens: int,
                    sampler_config, seed: int = 0) -> Iterator[int]:
    """Yields the new token ids one at a time (not the prompt's own tokens).

    The first call to the model reads the whole prompt. After that, the KV
    cache remembers the earlier words, so each step only passes in the one
    newest word.
    """
    prompt_ids = tokenizer.encode(prompt)
    all_tokens = list(prompt_ids)
    prompt_array = np.array(prompt_ids, dtype=np.int32)

    # The GPU model keeps its own cache inside. The CPU model needs one
    # passed in.
    uses_gpu_cache = hasattr(model, "forward_cached")
    if uses_gpu_cache:
        logits = model.forward_cached(prompt_array, 0)
    else:
        config = model.config
        cache = _C.KVCache(config.n_layers, config.max_seq_len,
                           config.n_kv_heads, config.head_dim)
        logits = model.forward(prompt_array, 0, cache)

    for step in range(max_new_tokens):
        # The last row holds the scores for what comes next.
        next_token = _C.sample(logits[-1], sampler_config, all_tokens, seed + step)
        all_tokens.append(next_token)
        yield next_token

        one_token = np.array([next_token], dtype=np.int32)
        position = len(all_tokens) - 1
        if uses_gpu_cache:
            logits = model.forward_cached(one_token, position)
        else:
            logits = model.forward(one_token, position, cache)


def generate(model, tokenizer, prompt: str, max_new_tokens: int,
             sampler_config, seed: int = 0) -> str:
    """Like generate_stream, but returns the whole answer as one string."""
    new_tokens = list(generate_stream(model, tokenizer, prompt, max_new_tokens,
                                      sampler_config, seed))
    # The tokenizer works on raw bytes, and a model (especially one with
    # random weights) can produce bytes that are not valid text. Replace
    # those with "?" characters instead of crashing the request.
    return tokenizer.decode(new_tokens).decode("utf-8", errors="replace")
