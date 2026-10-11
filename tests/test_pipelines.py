"""End-to-end: training writes weights, inference loads them, RL loss is wired but unused."""

from __future__ import annotations

import pytest
from tinygrad import Tensor

from minijev.config import JevConfig
from minijev.inference import CheckpointMissing, load_router, predict
from minijev.model import TinyJev, load_checkpoint, load_checkpoint_config, save_checkpoint
from minijev.objectives import policy_gradient_loss, supervised_loss
from minijev.schema import Route
from minijev.tokenizer import ByteTokenizer
from minijev.training import TrainingConfig, fit, tensors

TINY = [
    ("你好", Route.LOCAL),
    ("把 2 加 2", Route.LOCAL),
    ("查一下产品手册", Route.RAG),
    ("搜索项目文档里的部署命令", Route.RAG),
    ("设计一个分布式系统", Route.CLOUD),
    ("分析这段代码的性能问题", Route.CLOUD),
    ("这件事有法律风险吗", Route.REVIEW),
    ("我不确定是否安全", Route.REVIEW),
]


def _samples():
    from minijev.schema import JevSample

    return [JevSample(query=query, confidence=1.0, route=route) for query, route in TINY]


def test_tokenizer_pads_and_masks_consistently():
    tokenizer = ByteTokenizer(JevConfig())
    ids, mask = tokenizer.batch(["你好", ""])
    assert ids.shape == mask.shape == (2, JevConfig().max_seq_len)
    assert int(mask[0].sum().item()) == len("你好".encode("utf-8"))
    assert int(mask[1].sum().item()) == 0


def test_tokenizer_truncates_long_input_rather_than_failing():
    tokenizer = ByteTokenizer(JevConfig())
    ids, mask = tokenizer.batch(["字" * 500])
    assert int(mask[0].sum().item()) == JevConfig().max_seq_len
    assert ids.shape[1] == JevConfig().max_seq_len


def test_supervised_loss_weights_rows_by_their_agreement_label():
    """The dataset's confidence label is a weight, never a prediction target."""
    tokenizer = ByteTokenizer(JevConfig())
    model = TinyJev(JevConfig())
    samples = _samples()
    input_ids, mask, routes, weights = tensors(tokenizer, samples)
    outputs = model(input_ids, mask)

    assert supervised_loss(outputs, routes).item() > 0
    assert supervised_loss(outputs, routes, sample_weights=weights).item() > 0

    # A batch of one row reduces to that row's own loss whatever its weight is.
    single_ids, single_mask, single_routes, single_weights = tensors(tokenizer, samples[:1])
    single_outputs = model(single_ids, single_mask)
    one = supervised_loss(single_outputs, single_routes, sample_weights=single_weights)
    plain = supervised_loss(single_outputs, single_routes)
    assert one.item() == pytest.approx(plain.item(), rel=1e-5)


def test_agreement_weights_can_be_switched_off():
    tokenizer = ByteTokenizer(JevConfig())
    _, _, _, weights = tensors(tokenizer, _samples(), weight_by_agreement=False)
    assert weights is None


def test_label_smoothing_changes_the_loss():
    tokenizer = ByteTokenizer(JevConfig())
    model = TinyJev(JevConfig())
    input_ids, mask, routes, _ = tensors(tokenizer, _samples())
    outputs = model(input_ids, mask)
    assert supervised_loss(outputs, routes, label_smoothing=0.1).item() != pytest.approx(supervised_loss(outputs, routes).item())


def test_there_is_no_confidence_head_to_predict_the_label():
    """The removed head was the bug; a second head must not creep back in."""
    model = TinyJev(JevConfig())
    assert not hasattr(model, "confidence_head")
    assert "route_head" in vars(model)


def test_policy_gradient_loss_is_available_but_nothing_calls_it():
    """The reserved RL entry point: implemented, tested, not on the training path."""
    log_probabilities = Tensor([-0.5, -1.5, -0.2])
    rewards = Tensor([1.0, 0.0, 2.0])
    loss = policy_gradient_loss(log_probabilities, rewards)
    assert loss.item() > 0

    with_baseline = policy_gradient_loss(log_probabilities, rewards, baseline=Tensor([1.0, 1.0, 1.0]))
    assert with_baseline.item() != pytest.approx(loss.item())
    advantage = [1.0 - 1.0, 0.0 - 1.0, 2.0 - 1.0]
    expected = -sum(probability * weight for probability, weight in zip([-0.5, -1.5, -0.2], advantage)) / 3
    assert with_baseline.item() == pytest.approx(expected)


def test_policy_gradient_loss_can_be_differentiated():
    from tinygrad import Context

    with Context(TRAINING=1):
        log_probabilities = Tensor([-0.5, -1.5, -0.2])
        loss = policy_gradient_loss(log_probabilities, Tensor([1.0, 0.0, 2.0]))
        loss.backward()
    assert log_probabilities.grad is not None


def test_fit_produces_a_checkpoint_and_a_useful_log(tmp_path):
    samples = _samples()
    model, log = fit(
        samples,
        samples,
        TrainingConfig(epochs=2, batch_size=4, embed_dim=8, hidden_dim=8, patience=2, log_every=1),
        output=tmp_path / "smoke.safetensors",
    )
    assert (tmp_path / "smoke.safetensors").exists()
    assert log["config"]["best_epoch"] >= 1
    assert log["validation"]["samples"] == len(samples)
    # The audit must be present whether or not the head learned anything.
    assert "confidence" in log["validation"]
    assert "spread" in log["validation"]["confidence"]
    assert log["history"], "training history should be recorded"


def test_inference_loads_the_weights_it_was_given(tmp_path):
    samples = _samples()
    checkpoint = tmp_path / "smoke.safetensors"
    fit(samples, samples, TrainingConfig(epochs=1, batch_size=4, embed_dim=8, hidden_dim=8, patience=1), output=checkpoint)

    router = load_router(checkpoint)
    assert router.checkpoint == str(checkpoint)
    decisions = predict(["你好"], router=router)
    assert len(decisions) == 1
    decision = decisions[0]
    assert decision.route in set(Route)
    assert len(decision.route_probs) == 4
    assert sum(decision.route_probs) == pytest.approx(1.0)
    assert decision.confidence > 0


def test_inference_refuses_to_run_on_missing_weights(tmp_path):
    """The previous CLI silently used random weights; that must now be an error."""
    with pytest.raises(CheckpointMissing):
        load_router(tmp_path / "absent.safetensors")
    router = load_router(tmp_path / "absent.safetensors", allow_untrained=True)
    assert router.checkpoint is None


def test_checkpoint_roundtrip_preserves_scores(tmp_path):
    config = JevConfig(embed_dim=8, hidden_dim=8)
    tokenizer = ByteTokenizer(config)
    model = TinyJev(config)
    before = model(*tokenizer.batch(["你好"])).route_logits.numpy().flatten().tolist()
    path = tmp_path / "roundtrip.safetensors"
    save_checkpoint(model, path)
    reloaded = TinyJev(config)
    load_checkpoint(reloaded, path)
    after = reloaded(*tokenizer.batch(["你好"])).route_logits.numpy().flatten().tolist()
    assert after == pytest.approx(before)


def test_checkpoint_records_the_shapes_it_was_trained_with(tmp_path):
    """Without the sidecar, a non-default width produces an unloadable checkpoint."""
    config = JevConfig(embed_dim=8, hidden_dim=16)
    path = tmp_path / "narrow.safetensors"
    save_checkpoint(TinyJev(config), path)

    assert load_checkpoint_config(path) == config
    router = load_router(path)
    assert router.config.embed_dim == 8
    assert router.config.hidden_dim == 16
    assert router.route("你好").route in set(Route)


def test_legacy_checkpoint_without_a_sidecar_still_loads_at_the_default_width(tmp_path):
    path = tmp_path / "legacy.safetensors"
    save_checkpoint(TinyJev(JevConfig()), path)
    sidecar = path.with_suffix(".config.json")
    sidecar.unlink()
    assert load_checkpoint_config(path) is None
    assert load_router(path).config.embed_dim == JevConfig().embed_dim


def test_decision_carries_what_an_rl_step_needs():
    """log-probability of a sampled route, not just an argmax."""
    decision = load_router(None, allow_untrained=True).route("你好")
    assert decision.log_prob() <= 0
    for route in Route:
        assert decision.log_prob(route) <= 0
