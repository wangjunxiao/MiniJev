"""Both encoders must be real, checkpointable and interchangeable.

The transformer was added as an ablation arm. An ablation is only evidence if the
two arms differ in exactly one thing, so these tests pin down what each encoder
owns, that the config that produced a checkpoint travels with it, and that no
parameter is created outside `__init__` (a projection built in `__call__` would be
re-randomised on every batch and silently dropped from the checkpoint).
"""

from __future__ import annotations

import pytest
from tinygrad.nn.state import get_state_dict

from minijev.config import JevConfig
from minijev.model import TinyJev, load_checkpoint, load_checkpoint_config, save_checkpoint
from minijev.tokenizer import ByteTokenizer


def _batch(texts=("你好", "查一下产品手册")):
    config = JevConfig()
    tokenizer = ByteTokenizer(config)
    return tokenizer.batch(list(texts))


def test_pooled_encoder_owns_no_transformer_parameters():
    model = TinyJev(JevConfig(encoder_type="pooled"))
    keys = set(get_state_dict(model))
    assert "encoder.weight" in keys
    assert not any(key.startswith("transformer") for key in keys)
    assert model.position_embedding is None


def test_transformer_encoder_owns_attention_and_position_parameters():
    model = TinyJev(JevConfig(encoder_type="transformer", num_layers=1, num_heads=4, ffn_dim=128))
    keys = set(get_state_dict(model))
    assert "position_embedding.weight" in keys
    assert "transformer.0.qkv.weight" in keys
    assert "transformer.0.out.weight" in keys
    assert model.encoder is None


def test_transformer_projection_is_a_parameter_not_a_local():
    """It was once created inside the forward pass: untrained, uncheckpointed, random."""
    model = TinyJev(JevConfig(encoder_type="transformer"))
    keys = set(get_state_dict(model))
    assert "transformer_projection.weight" in keys
    assert "transformer_projection.bias" in keys


def test_both_encoders_produce_four_logits():
    for encoder in ("pooled", "transformer"):
        model = TinyJev(JevConfig(encoder_type=encoder))
        outputs = model(*_batch())
        assert outputs.route_logits.shape == (2, 4)


def test_attention_masks_padding_out_of_the_read():
    """A padded row must score identically to the same row alone, or the mask is decorative."""
    tokenizer = ByteTokenizer(JevConfig())
    model = TinyJev(JevConfig(encoder_type="transformer"))
    alone = model(*tokenizer.batch(["你好"])).route_logits.numpy().flatten()
    padded = model(*tokenizer.batch(["你好", ""])).route_logits.numpy().flatten()[:4]
    assert padded == pytest.approx(alone, abs=1e-5)


def test_config_rejects_shapes_the_transformer_cannot_use():
    with pytest.raises(ValueError):
        JevConfig(encoder_type="transformer", embed_dim=64, num_heads=5)
    with pytest.raises(ValueError):
        JevConfig(encoder_type="lstm")
    with pytest.raises(ValueError):
        JevConfig(encoder_type="transformer", num_layers=0)


def test_checkpoint_carries_the_encoder_choice(tmp_path):
    """Loading a transformer checkpoint must rebuild a transformer, not the default."""
    config = JevConfig(encoder_type="transformer", num_layers=2, num_heads=4, ffn_dim=64)
    path = tmp_path / "t.safetensors"
    save_checkpoint(TinyJev(config), path)
    assert load_checkpoint_config(path) == config

    rebuilt = TinyJev(load_checkpoint_config(path))
    assert len(rebuilt.transformer) == 2
    assert set(get_state_dict(rebuilt)) == set(get_state_dict(TinyJev(config)))


def test_transformer_checkpoint_round_trips_within_float_tolerance(tmp_path):
    """Tensor values match exactly; the logits differ only by kernel ordering (~1e-7)."""
    config = JevConfig(encoder_type="transformer", num_layers=1, num_heads=4, ffn_dim=64)
    tokenizer = ByteTokenizer(config)
    model = TinyJev(config)
    before = model(*tokenizer.batch(["你好"])).route_logits.numpy().flatten()
    path = tmp_path / "t.safetensors"
    save_checkpoint(model, path)

    reloaded = TinyJev(load_checkpoint_config(path))
    load_checkpoint(reloaded, path)
    after = reloaded(*tokenizer.batch(["你好"])).route_logits.numpy().flatten()
    assert after == pytest.approx(before, abs=1e-5)


def test_repeated_forward_passes_are_identical():
    """Guards the exact bug that was fixed: a per-call random projection."""
    model = TinyJev(JevConfig(encoder_type="transformer"))
    batch = _batch()
    first = model(*batch).route_logits.numpy().flatten().tolist()
    second = model(*batch).route_logits.numpy().flatten().tolist()
    assert first == second


def test_preserve_tail_keeps_the_end_of_a_long_prompt():
    """Literary prompts carry the requested operation at the end.

    The tail window is `max_seq_len - max_seq_len // 2` bytes, so the marker has to
    fit inside it for the last bytes of the encoding to equal the marker.
    """
    max_seq_len = 32
    tail_window = max_seq_len - max_seq_len // 2
    marker = "汉字数量"  # 12 bytes, inside the 16-byte tail window
    assert len(marker.encode("utf-8")) <= tail_window
    query = "甲" * 40 + marker

    keeping = ByteTokenizer(JevConfig(max_seq_len=max_seq_len, preserve_tail=True)).encode(query)
    assert keeping.input_ids[-len(marker.encode("utf-8")) :] == [byte + 1 for byte in marker.encode("utf-8")]

    truncating = ByteTokenizer(JevConfig(max_seq_len=max_seq_len, preserve_tail=False)).encode(query)
    assert [byte + 1 for byte in marker.encode("utf-8")] != truncating.input_ids[-len(marker.encode("utf-8")) :]


def test_a_short_prompt_is_never_truncated_by_either_mode():
    expected = [byte + 1 for byte in "你好".encode("utf-8")]
    for preserve in (True, False):
        encoded = ByteTokenizer(JevConfig(max_seq_len=128, preserve_tail=preserve)).encode("你好")
        assert encoded.input_ids[: len(expected)] == expected
        assert sum(encoded.attention_mask) == len(expected)
