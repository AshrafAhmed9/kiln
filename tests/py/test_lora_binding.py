import numpy as np

from kiln_py import _C


def test_python_can_merge_a_lora_adapter_into_the_real_model():
    config = _C.ModelConfig()
    config.vocab_size = 16
    config.hidden_size = 8
    config.n_layers = 1
    config.n_heads = 2
    config.n_kv_heads = 1
    config.head_dim = 4
    config.ffn_hidden = 16
    config.max_seq_len = 16
    config.rms_eps = 1e-5
    config.rope_theta = 10000.0
    model = _C.Model.load_random(config, 3)
    tokens = np.asarray([1, 2], dtype=np.int32)
    before = model.forward(tokens, 1, 2, None, 0, None)

    model.merge_lora_into_layer(
        0, "wq", np.ones((1, 8), dtype=np.float32),
        np.ones((8, 1), dtype=np.float32), 0.1,
    )
    after = model.forward(tokens, 1, 2, None, 0, None)
    assert not np.allclose(before, after)
