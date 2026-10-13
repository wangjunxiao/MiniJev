"""Calibration-aware training must optimise the number the safety branch actually reads.

Reference: Karandikar et al., *Soft Calibration Objectives for Neural Networks* (arXiv
2108.00106). ECE bins confidences and compares each bin's mean confidence with its empirical
accuracy; the bin assignment is a step function, so the quantity is not differentiable.
Replacing the hard bin with a triangular hat makes it smooth enough to train against.

Two ways this could silently do nothing:

1. **The loss could optimise a different statistic than the one reported.** `normalized_entropy`
   is MiniJev's default and what the risk ladder thresholds; penalising `top_probability`
   instead would improve a number nobody uses.
2. **The loss could be constant.** A calibration term that never varies with the model gives no
   gradient and looks like a failed experiment rather than a misconfiguration.
"""

from __future__ import annotations

import math

import pytest
from tinygrad import Tensor, dtypes

from minijev.confidence import (
    CONFIDENCE_MEASURES,
    DIFFERENTIABLE_MEASURES,
    confidence_from,
    differentiable_confidence,
)
from minijev.model import JevOutputs
from minijev.objectives import soft_calibration_loss

LOGITS = [
    [4.0, 0.0, -1.0, 0.5],
    [0.2, 0.3, 0.1, 0.4],
    [0.0, 0.0, 0.0, 0.0],
    [10.0, 9.9, -5.0, -5.0],
    [-1.0, -1.0, -1.0, 20.0],
]


def _probabilities(row: list[float]) -> list[float]:
    exponentials = [math.exp(value) for value in row]
    total = sum(exponentials)
    return [value / total for value in exponentials]


# --- the differentiable measure must equal the reported one ---------------------


@pytest.mark.parametrize("measure", DIFFERENTIABLE_MEASURES)
def test_the_differentiable_measure_equals_the_scalar_one(measure):
    """A training signal for a different statistic is worse than no signal."""
    tensor = Tensor(LOGITS, dtype=dtypes.float32)
    differentiable = differentiable_confidence(tensor, measure).numpy().flatten().tolist()
    scalar = [confidence_from(_probabilities(row), measure) for row in LOGITS]
    for left, right in zip(differentiable, scalar):
        assert left == pytest.approx(right, abs=1e-5)


def test_every_measure_can_be_measured_but_only_smooth_ones_trained():
    assert set(DIFFERENTIABLE_MEASURES) <= set(CONFIDENCE_MEASURES)
    assert set(CONFIDENCE_MEASURES) - set(DIFFERENTIABLE_MEASURES) == {"margin"}


def test_an_unsmooth_measure_is_refused_for_training():
    """`margin` is only piecewise smooth. Refusing beats training against nothing."""
    with pytest.raises(KeyError):
        differentiable_confidence(Tensor(LOGITS, dtype=dtypes.float32), "margin")
    with pytest.raises(KeyError):
        soft_calibration_loss(JevOutputs(route_logits=Tensor(LOGITS, dtype=dtypes.float32)), Tensor([0] * 5, dtype=dtypes.int32), measure="margin")


def test_a_uniform_distribution_scores_zero_confidence_and_a_peaked_one_scores_high():
    peaked = Tensor([[20.0, 0.0, 0.0, 0.0]], dtype=dtypes.float32)
    flat = Tensor([[0.0, 0.0, 0.0, 0.0]], dtype=dtypes.float32)
    assert differentiable_confidence(peaked).numpy().flatten()[0] == pytest.approx(1.0, abs=1e-4)
    assert differentiable_confidence(flat).numpy().flatten()[0] == pytest.approx(0.0, abs=1e-4)


# --- the loss behaves like a calibration error ---------------------------------


def test_confidently_wrong_is_the_maximum_penalty():
    """The exact failure the term exists to punish, and the one MiniJev shows out of domain."""
    wrong = JevOutputs(route_logits=Tensor([[20.0, 0.0, 0.0, 0.0]] * 4, dtype=dtypes.float32))
    mislabelled = Tensor([1, 2, 3, 1], dtype=dtypes.int32)
    assert soft_calibration_loss(wrong, mislabelled).item() == pytest.approx(1.0, abs=1e-3)


def test_confidently_right_is_nearly_free():
    right = JevOutputs(route_logits=Tensor([[20.0, 0.0, 0.0, 0.0]] * 4, dtype=dtypes.float32))
    correct = Tensor([0, 0, 0, 0], dtype=dtypes.int32)
    assert soft_calibration_loss(right, correct).item() == pytest.approx(0.0, abs=1e-3)


def test_an_under_confident_model_is_punished_by_its_accuracy_gap():
    """A uniform model claims confidence 0 yet still guesses one of four right.

    That is under-confidence of exactly the accuracy, so the penalty is the accuracy and not
    zero. Calibration is a *match* between confidence and accuracy, not a preference for low
    confidence -- the distinction the first version of this test got wrong.
    """
    hedged = JevOutputs(route_logits=Tensor([[0.0, 0.0, 0.0, 0.0]] * 4, dtype=dtypes.float32))
    mixed = Tensor([0, 1, 2, 3], dtype=dtypes.int32)  # argmax ties resolve to route 0: accuracy 0.25
    assert soft_calibration_loss(hedged, mixed).item() == pytest.approx(0.25, abs=0.02)


def test_the_penalty_tracks_the_gap_between_confidence_and_accuracy():
    """The loss must be a function of the gap, monotone in how far off the confidence is."""
    labels = Tensor([0, 0, 1, 1], dtype=dtypes.int32)  # accuracy 0.5 under a confident read
    near = JevOutputs(route_logits=Tensor([[0.0, 0.0, 0.0, 0.0]] * 4, dtype=dtypes.float32))
    far = JevOutputs(route_logits=Tensor([[9.0, 0.0, 0.0, 0.0]] * 4, dtype=dtypes.float32))
    # Uniform: confidence 0 against accuracy 0.5 -> gap 0.5. Peaked: confidence ~1 against 0.5 -> gap ~0.5.
    assert soft_calibration_loss(near, labels).item() == pytest.approx(0.5, abs=0.05)
    assert soft_calibration_loss(far, labels).item() == pytest.approx(0.5, abs=0.05)


def test_a_calibrated_model_is_cheaper_than_a_miscalibrated_one_at_equal_accuracy():
    """The property that makes this a calibration loss rather than an accuracy loss."""
    labels = Tensor([0, 1, 0, 1], dtype=dtypes.int32)
    # Both predict the same routes (accuracy 0.5); only the stated confidence differs.
    honest = JevOutputs(route_logits=Tensor([[0.4, 0.4, 0.1, 0.1]] * 4, dtype=dtypes.float32))
    boastful = JevOutputs(route_logits=Tensor([[9.0, 0.0, 0.0, 0.0]] * 4, dtype=dtypes.float32))
    assert soft_calibration_loss(honest, labels).item() < soft_calibration_loss(boastful, labels).item()


def test_the_loss_is_not_constant_in_the_model():
    """A term that never varies gives no gradient and would look like a failed experiment."""
    weak = JevOutputs(route_logits=Tensor([[1.0, 0.0, 0.0, 0.0]] * 4, dtype=dtypes.float32))
    strong = JevOutputs(route_logits=Tensor([[8.0, 0.0, 0.0, 0.0]] * 4, dtype=dtypes.float32))
    labels = Tensor([1, 1, 1, 1], dtype=dtypes.int32)
    assert soft_calibration_loss(weak, labels).item() != pytest.approx(soft_calibration_loss(strong, labels).item(), abs=1e-6)


def test_the_loss_carries_a_nonzero_gradient():
    """The property that matters is that a gradient exists.

    An earlier version asserted the loss *falls* over eight Adam steps. It passed alone and
    failed in the full suite, because the model's random init depends on global RNG state and
    the term is noisy enough to bounce upward from some starts -- a flaky test asserting
    something that is not actually the property under test.
    """
    from tinygrad import Context
    from tinygrad.nn.state import get_parameters

    from minijev.config import JevConfig
    from minijev.model import TinyJev
    from minijev.tokenizer import ByteTokenizer

    config = JevConfig()
    model = TinyJev(config)
    tokenizer = ByteTokenizer(config)
    ids, mask = tokenizer.batch(["查", "算", "分析", "判断"])
    labels = Tensor([1, 0, 2, 3], dtype=dtypes.int32)

    with Context(TRAINING=1):
        loss = soft_calibration_loss(model(ids, mask), labels)
        assert loss.item() > 0.0, "a randomly initialised model is not already calibrated"
        loss.backward()
        magnitudes = [abs(parameter.grad.numpy()).sum() for parameter in get_parameters(model) if parameter.grad is not None]
    assert magnitudes, "backward produced no gradients at all"
    assert sum(magnitudes) > 0.0, "the calibration term produced an all-zero gradient"


def test_sample_weights_are_respected():
    outputs = JevOutputs(route_logits=Tensor([[8.0, 0.0, 0.0, 0.0]] * 2, dtype=dtypes.float32))
    labels = Tensor([1, 1], dtype=dtypes.int32)
    unweighted = soft_calibration_loss(outputs, labels).item()
    zeroed = soft_calibration_loss(outputs, labels, sample_weights=Tensor([0.0, 0.0], dtype=dtypes.float32)).item()
    assert unweighted > 0.5
    # Every row down-weighted to nothing leaves no mass; the term must not blow up.
    assert zeroed == pytest.approx(0.0, abs=1e-3)
