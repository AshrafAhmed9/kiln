"""Speculative decoding: a small, fast "draft" model guesses several words
ahead, and a single pass through the real, big "target" model checks all of
those guesses at once. See docs/learning/phase-10.md for the full
reasoning behind the acceptance rule below -- in short, it's built so the
final output has exactly the same odds of happening as if the target model
had generated every word by itself, with no shortcuts.

This intentionally does not use a KV cache (see docs/learning/phase-10.md
for why) -- every round recomputes the whole context from scratch through
both models. That's a real, named simplification, not an oversight.
"""
from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits / temperature
    shifted = scaled - np.max(scaled)
    exp = np.exp(shifted)
    return exp / np.sum(exp)


def _forward_all_logits(model, tokens: list[int]) -> np.ndarray:
    """Runs the model over the whole given sequence, with no cache, and
    returns every position's logits (one row per input token).
    """
    tokens_array = np.array(tokens, dtype=np.int32)
    return model.forward(tokens_array, 1, len(tokens), None, 0, None)


def speculative_decode_round(draft_model, target_model, context: list[int],
                              num_draft_tokens: int, temperature: float,
                              rng: np.random.Generator) -> list[int]:
    """Runs one full round: the draft model proposes `num_draft_tokens`
    words, the target model checks all of them in one pass, and this
    returns the words that actually get kept (which may be fewer than
    proposed, if one gets rejected partway through -- or one more than
    proposed, if every guess was accepted and a bonus word comes for free).
    """
    greedy = temperature <= 0.0

    draft_tokens: list[int] = []
    draft_probs: list[np.ndarray] = []
    running_context = list(context)

    for _ in range(num_draft_tokens):
        logits = _forward_all_logits(draft_model, running_context)[-1]
        if greedy:
            token = int(np.argmax(logits))
            # Never actually read back in greedy mode (see the greedy
            # branch below, which returns before touching draft_probs) --
            # kept as a real, empty array rather than None just so every
            # entry in this list has one consistent, simple type.
            probs = np.empty(0, dtype=np.float32)
        else:
            probs = _softmax(logits, temperature)
            token = int(rng.choice(len(probs), p=probs))
        draft_tokens.append(token)
        draft_probs.append(probs)
        running_context.append(token)

    full_sequence = context + draft_tokens
    target_logits_all = _forward_all_logits(target_model, full_sequence)

    accepted: list[int] = []
    context_len = len(context)
    for j in range(num_draft_tokens):
        verify_row = target_logits_all[context_len - 1 + j]

        if greedy:
            target_choice = int(np.argmax(verify_row))
            if draft_tokens[j] == target_choice:
                accepted.append(draft_tokens[j])
                continue
            # A mismatch, in greedy mode, always means "replace with
            # whatever the target model would have picked" -- see
            # docs/learning/phase-10.md for why this is the deterministic
            # collapse of the general accept/residual rule, not a
            # separate special case bolted on.
            accepted.append(target_choice)
            return accepted

        target_probs = _softmax(verify_row, temperature)
        p = target_probs[draft_tokens[j]]
        q = draft_probs[j][draft_tokens[j]]
        accept_probability = min(1.0, p / q)

        if rng.uniform() <= accept_probability:
            accepted.append(draft_tokens[j])
            continue

        residual = np.maximum(0.0, target_probs - draft_probs[j])
        residual_sum = np.sum(residual)
        residual /= residual_sum
        replacement = int(rng.choice(len(residual), p=residual))
        accepted.append(replacement)
        return accepted

    # Every draft token was accepted -- one more word comes for free, since
    # the target model's forward pass already computed its prediction for
    # what comes right after the full accepted sequence.
    bonus_row = target_logits_all[-1]
    if greedy:
        accepted.append(int(np.argmax(bonus_row)))
    else:
        bonus_probs = _softmax(bonus_row, temperature)
        accepted.append(int(rng.choice(len(bonus_probs), p=bonus_probs)))

    return accepted


def speculative_generate(draft_model, target_model, prompt_tokens: list[int],
                          num_new_tokens: int, num_draft_tokens: int,
                          temperature: float, seed: int = 0) -> list[int]:
    """Keeps running speculative rounds until num_new_tokens new words have
    been produced (the last round may overshoot slightly and gets trimmed).
    """
    rng = np.random.default_rng(seed)
    context = list(prompt_tokens)
    new_tokens: list[int] = []

    while len(new_tokens) < num_new_tokens:
        accepted = speculative_decode_round(
            draft_model, target_model, context, num_draft_tokens,
            temperature, rng)
        new_tokens.extend(accepted)
        context.extend(accepted)

    return new_tokens[:num_new_tokens]
