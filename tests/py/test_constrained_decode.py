"""Tests for schema-constrained decoding (Phase 20). The real claim being
tested: json.loads() actually succeeds on the output, every time, across
many seeds -- not "looks like JSON," an executable, unforgiving check.
"""
from kiln_py import _C
from kiln_py.runtime.constrained_decode import (IntegerSlot, Literal,
                                                 StringSlot, generate_constrained,
                                                 is_valid_json, string_schema)


def _toy_model(seed=1, vocab_size=256):
    config = _C.ModelConfig()
    config.vocab_size = vocab_size
    config.hidden_size = 16
    config.n_layers = 2
    config.n_heads = 2
    config.n_kv_heads = 1
    config.head_dim = 8
    config.ffn_hidden = 32
    config.max_seq_len = 128
    config.rms_eps = 1e-5
    config.rope_theta = 10000.0
    return _C.Model.load_random(config, seed)


def test_string_schema_is_always_valid_json_across_many_seeds():
    model = _toy_model(seed=1)
    schema = string_schema("name", "city")
    for seed in range(20):
        result = generate_constrained(model, schema, temperature=1.0, seed=seed)
        assert is_valid_json(result), f"invalid JSON at seed {seed}: {result!r}"


def test_string_schema_matches_the_expected_keys():
    import json
    model = _toy_model(seed=2)
    schema = string_schema("name", "city")
    result = generate_constrained(model, schema, temperature=1.0, seed=0)
    parsed = json.loads(result)
    assert set(parsed.keys()) == {"name", "city"}
    assert isinstance(parsed["name"], str)
    assert isinstance(parsed["city"], str)


def test_integer_slot_produces_a_real_integer_field():
    import json
    model = _toy_model(seed=3)
    schema = [Literal(b'{"name": "'), StringSlot(), Literal(b'", "age": '),
             IntegerSlot(), Literal(b"}")]
    for seed in range(10):
        result = generate_constrained(model, schema, temperature=1.0, seed=seed)
        assert is_valid_json(result), f"invalid JSON at seed {seed}: {result!r}"
        parsed = json.loads(result)
        assert isinstance(parsed["age"], int)


def test_greedy_constrained_decoding_is_deterministic():
    model = _toy_model(seed=4)
    schema = string_schema("name")
    first = generate_constrained(model, schema, temperature=0.0, seed=42)
    second = generate_constrained(model, schema, temperature=0.0, seed=42)
    assert first == second


def test_is_valid_json_rejects_actual_garbage():
    assert not is_valid_json("not json at all {")
    assert is_valid_json('{"a": 1}')
