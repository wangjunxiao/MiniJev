"""The operating point must be an input to the policy, not a constant baked into it.

This is the architectural half of "RL-like but not RL": a contextual bandit evaluated against
a *stated* objective. One policy serves a family of quality-versus-cost trade-offs, and the
operator moves along the family without retraining. The scoring side already had
`Preference`; these tests hold the model side to the same definition.
"""

from __future__ import annotations

import pytest
from tinygrad import Tensor, dtypes
from tinygrad.nn.state import get_state_dict

from minijev.config import JevConfig
from minijev.model import TinyJev, load_checkpoint_config, save_checkpoint
from minijev.rewards import PREFERENCE_SWEEP, Preference, RewardWeights
from minijev.tokenizer import ByteTokenizer

QUERIES = ["查一下安装步骤", "算一下 2 加 2"]


def _batch(config: JevConfig):
    tokenizer = ByteTokenizer(config)
    return tokenizer.batch(QUERIES)


def _preference(share: float, rows: int = 2) -> Tensor:
    value = Preference.at_cost_share(share)
    return Tensor([[value.quality, value.cost]] * rows, dtype=dtypes.float32)


def test_a_preference_free_config_is_unchanged():
    """Every earlier checkpoint must keep loading and deciding identically."""
    config = JevConfig()
    assert config.preference_dim == 0
    assert config.uses_preference is False
    model = TinyJev(config)
    ids, mask = _batch(config)
    assert model(ids, mask).route_logits.shape == (2, 4)


def test_the_route_head_accepts_the_operating_point():
    config = JevConfig(preference_dim=2)
    assert config.uses_preference is True
    model = TinyJev(config)
    assert model.route_head.weight.shape == (len(config_route_order()), config.hidden_dim + 2)


def config_route_order():
    from minijev.schema import ROUTE_ORDER

    return ROUTE_ORDER


def test_the_preference_changes_the_logits():
    """If it did not, the conditioning would be decorative."""
    config = JevConfig(preference_dim=2)
    model = TinyJev(config)
    ids, mask = _batch(config)
    cheap = model(ids, mask, _preference(0.9)).route_logits.numpy().flatten().tolist()
    quality = model(ids, mask, _preference(0.1)).route_logits.numpy().flatten().tolist()
    assert cheap != quality


def test_the_same_preference_is_reproducible():
    config = JevConfig(preference_dim=2)
    model = TinyJev(config)
    ids, mask = _batch(config)
    first = model(ids, mask, _preference(0.5)).route_logits.numpy().flatten().tolist()
    second = model(ids, mask, _preference(0.5)).route_logits.numpy().flatten().tolist()
    assert first == second


def test_a_plain_model_refuses_a_preference_rather_than_ignoring_it():
    """Silently dropping it would make an unconditional run look conditional in the logs."""
    config = JevConfig()
    model = TinyJev(config)
    ids, mask = _batch(config)
    with pytest.raises(ValueError):
        model(ids, mask, _preference(0.5))


def test_a_conditioned_model_refuses_to_run_without_one():
    config = JevConfig(preference_dim=2)
    model = TinyJev(config)
    ids, mask = _batch(config)
    with pytest.raises(ValueError):
        model(ids, mask)


def test_a_wrongly_shaped_preference_is_refused():
    config = JevConfig(preference_dim=2)
    model = TinyJev(config)
    ids, mask = _batch(config)
    with pytest.raises(ValueError):
        model(ids, mask, Tensor([[1.0]] * 2, dtype=dtypes.float32))
    with pytest.raises(ValueError):
        model(ids, mask, Tensor([[1.0, 0.0]], dtype=dtypes.float32))


def test_a_negative_preference_width_is_refused():
    with pytest.raises(ValueError):
        JevConfig(preference_dim=-1)


def test_the_scoring_and_the_model_share_one_definition_of_a_preference():
    """`Preference.normalized` must be exactly what the reward uses, or the knob lies."""
    for preference in PREFERENCE_SWEEP:
        weights = RewardWeights().with_preference(preference)
        assert (weights.quality, weights.cost) == pytest.approx(preference.normalized)


def test_the_preference_width_travels_in_the_checkpoint(tmp_path):
    path = tmp_path / "pref.safetensors"
    save_checkpoint(TinyJev(JevConfig(preference_dim=2)), path)
    config = load_checkpoint_config(path)
    assert config.preference_dim == 2
    assert config.uses_preference is True
    # The rebuilt model must have a head the stored weights fit into.
    rebuilt = TinyJev(config)
    assert rebuilt.route_head.weight.shape == (len(config_route_order()), config.hidden_dim + 2)


def test_a_conditioned_checkpoint_carries_the_extra_head_weights(tmp_path):
    path = tmp_path / "pref.safetensors"
    save_checkpoint(TinyJev(JevConfig(preference_dim=2)), path)
    keys = set(get_state_dict(TinyJev(load_checkpoint_config(path))))
    assert "route_head.weight" in keys
    plain_keys = set(get_state_dict(TinyJev(JevConfig())))
    assert plain_keys == keys, "the conditioning must not add or remove parameters, only widen one"


def test_the_extra_parameters_are_negligible():
    """Guards the confound that bit the encoder comparison: conditioning must not be capacity."""
    plain = sum(t.numel() for t in get_state_dict(TinyJev(JevConfig())).values())
    conditioned = sum(t.numel() for t in get_state_dict(TinyJev(JevConfig(preference_dim=2))).values())
    assert conditioned - plain == 2 * len(config_route_order())
    assert (conditioned - plain) / plain < 0.01
