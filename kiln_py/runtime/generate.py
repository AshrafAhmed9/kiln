"""The generation loop: the part that repeatedly asks the model "what's the
next word?", one word at a time, until we have a full answer. This lives in
Python because it's just a loop calling into C++ -- the actual heavy lifting
(the forward pass) still happens on the C++ side, through the pybind11
boundary described in constitution §6.
"""
from __future__ import annotations

import numpy as np

from kiln_py import _C


def generate(model, tokenizer, prompt: str, max_new_tokens: int,
             sampler_config, seed: int = 0) -> str:
    """Reads `prompt`, then keeps asking the model for one more word at a
    time (reusing the KV cache so it never has to reread words it's already
    seen), until it has generated `max_new_tokens` new words or produced
    them all. Returns just the newly generated text, not the prompt.
    """
    prompt_ids = tokenizer.encode(prompt)
    config = model.config

    # The cache remembers everything the model has "read" so far, so each
    # new word only costs one more step of work instead of rereading the
    # whole conversation from scratch every time.
    cache = _C.KVCache(config.n_layers, config.max_seq_len, config.n_kv_heads,
                        config.head_dim)

    all_tokens = list(prompt_ids)

    # First step: read the whole prompt at once (this is called "prefill").
    tokens_array = np.array(prompt_ids, dtype=np.int32)
    logits = model.forward(tokens_array, 1, len(prompt_ids), None, 0, cache)

    for step in range(max_new_tokens):
        last_word_scores = logits[-1]
        next_token = _C.sample(last_word_scores, sampler_config, all_tokens,
                                seed + step)
        all_tokens.append(next_token)

        # Every following step only has to hand the model the ONE new word
        # -- the cache already remembers everything before it. This is why
        # a cached loop is so much cheaper than starting over every time.
        one_token = np.array([next_token], dtype=np.int32)
        position = len(all_tokens) - 1
        logits = model.forward(one_token, 1, 1, None, position, cache)

    new_tokens = all_tokens[len(prompt_ids):]
    raw_bytes = tokenizer.decode(new_tokens)
    # A byte-level tokenizer hands back raw bytes, not guaranteed-valid
    # text -- especially from an untrained model, which can output any byte
    # value at all, not just ones that spell out real characters. Any byte
    # sequence that isn't valid text gets swapped for the standard "�"
    # placeholder character instead of crashing the whole request.
    return raw_bytes.decode("utf-8", errors="replace")
