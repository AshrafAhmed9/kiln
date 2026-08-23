"""Tests for speculative decoding. The headline test is the greedy case:
see docs/learning/phase-10.md for why, in greedy mode, speculative
decoding's output must be EXACTLY identical to just running the target
model alone, word for word, every single time -- with no randomness and no
statistics needed to check it.
"""
import numpy as np

from kiln_py import _C
from kiln_py.runtime.speculative_decode import speculative_generate


def _toy_config(vocab_size=32, hidden=16, layers=2, heads=2, kv_heads=1,
                 head_dim=8, ffn=32):
    config = _C.ModelConfig()
    config.vocab_size = vocab_size
    config.hidden_size = hidden
    config.n_layers = layers
    config.n_heads = heads
    config.n_kv_heads = kv_heads
    config.head_dim = head_dim
    config.ffn_hidden = ffn
    config.max_seq_len = 64
    config.rms_eps = 1e-5
    config.rope_theta = 10000.0
    return config


def _greedy_generate_direct(model, prompt_tokens, num_new_tokens):
    """A plain, cache-free, greedy generation loop using only the target
    model -- the baseline speculative decoding's greedy output must match
    exactly. Deliberately independent of kiln_py/runtime/generate.py (which
    uses the KV cache) so this test isolates the acceptance rule itself,
    not whether caching agrees with not caching (already proven in Phase 3).
    """
    context = list(prompt_tokens)
    for _ in range(num_new_tokens):
        tokens_array = np.array(context, dtype=np.int32)
        logits = model.forward(tokens_array, 1, len(context), None, 0, None)
        next_token = int(np.argmax(logits[-1]))
        context.append(next_token)
    return context[len(prompt_tokens):]


def test_greedy_speculative_decoding_exactly_matches_direct_greedy_decoding():
    # Two different models stand in for "small draft" and "big target" --
    # any two models work for testing the algorithm itself, since the
    # guarantee being tested doesn't depend on the draft model being
    # smaller, only on the acceptance rule being applied correctly.
    draft_model = _C.Model.load_random(_toy_config(hidden=8, layers=1), 1)
    target_model = _C.Model.load_random(_toy_config(hidden=16, layers=2), 2)

    prompt = [1, 2, 3]
    num_new_tokens = 6

    direct = _greedy_generate_direct(target_model, prompt, num_new_tokens)
    speculative = speculative_generate(
        draft_model, target_model, prompt, num_new_tokens,
        num_draft_tokens=3, temperature=0.0)

    assert speculative == direct


def test_greedy_speculative_decoding_matches_across_different_draft_lengths():
    """The number of tokens the draft model guesses ahead is purely a
    speed knob -- it must never change the final answer. Trying a few
    different draft lengths against the same direct-greedy baseline is a
    real check of that, not just a repeat of the test above.
    """
    draft_model = _C.Model.load_random(_toy_config(hidden=8, layers=1), 5)
    target_model = _C.Model.load_random(_toy_config(hidden=16, layers=2), 6)
    prompt = [4, 5]
    num_new_tokens = 5
    direct = _greedy_generate_direct(target_model, prompt, num_new_tokens)

    for draft_length in (1, 2, 4):
        speculative = speculative_generate(
            draft_model, target_model, prompt, num_new_tokens,
            num_draft_tokens=draft_length, temperature=0.0)
        assert speculative == direct, f"mismatch at draft_length={draft_length}"


def test_random_sampling_mode_runs_and_produces_the_right_length():
    """The non-greedy path involves real randomness, so it can't be
    checked for an exact match the way greedy mode can -- proving its
    distribution is exactly preserved needs the statistical, large-sample
    comparison this offline CPU session doesn't have room for (see
    docs/learning/phase-10.md). What's checked here is that the mechanism
    runs correctly end to end and produces a valid, right-length sequence
    of real token ids -- a sanity check, not a proof of the distribution
    guarantee.
    """
    draft_model = _C.Model.load_random(_toy_config(vocab_size=16, hidden=8,
                                                    layers=1), 9)
    target_model = _C.Model.load_random(_toy_config(vocab_size=16, hidden=16,
                                                     layers=2), 10)
    result = speculative_generate(draft_model, target_model, [1, 2],
                                   num_new_tokens=8, num_draft_tokens=3,
                                   temperature=1.0, seed=42)
    assert len(result) == 8
    assert all(0 <= token < 16 for token in result)
