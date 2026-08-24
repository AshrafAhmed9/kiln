import numpy as np

from kiln_py import _C
from kiln_py.runtime.continuous_batch import ContinuousBatchExecutor
from kiln_py.scheduler.scheduler import Request


def toy_model():
    config = _C.ModelConfig()
    config.vocab_size = 32
    config.hidden_size = 8
    config.n_layers = 2
    config.n_heads = 2
    config.n_kv_heads = 1
    config.head_dim = 4
    config.ffn_hidden = 16
    config.max_seq_len = 32
    config.rms_eps = 1e-5
    config.rope_theta = 10000.0
    return _C.Model.load_random(config, 8)


def sampler_config():
    config = _C.SamplerConfig()
    config.temperature = 0.0
    return config


def test_existing_sequences_share_one_cached_decode_call():
    model = toy_model()
    executor = ContinuousBatchExecutor(model)
    first = Request(1, [1, 2], 3)
    second = Request(2, [3, 4, 5], 3)
    executor.register(first, sampler_config(), seed=1)
    executor.register(second, sampler_config(), seed=2)

    first_tokens = executor([first, second])
    first.tokens.append(first_tokens[0])
    second.tokens.append(first_tokens[1])
    second_tokens = executor([first, second])
    first.tokens.append(second_tokens[0])
    second.tokens.append(second_tokens[1])

    # The second call reaches Model.forward_decode_batch with both requests.
    # Its results must still agree with doing that same call independently.
    first_cache = _C.KVCache(2, 32, 1, 4)
    second_cache = _C.KVCache(2, 32, 1, 4)
    model.forward(np.asarray([1, 2], dtype=np.int32), 1, 2, None, 0, first_cache)
    model.forward(np.asarray([3, 4, 5], dtype=np.int32), 1, 3, None, 0, second_cache)
    first_reference = model.forward(
        np.asarray([first_tokens[0]], dtype=np.int32), 1, 1, None, 2, first_cache
    )[-1]
    second_reference = model.forward(
        np.asarray([first_tokens[1]], dtype=np.int32), 1, 1, None, 3, second_cache
    )[-1]

    third_tokens = executor([first, second])

    # The third call samples the logits produced by the preceding shared C++
    # decode call. Those must be the same logits independent cached decoding
    # produced for each sequence.
    assert third_tokens[0] == int(first_reference.argmax())
    assert third_tokens[1] == int(second_reference.argmax())
