"""Confidence is derived from the distribution, so its properties are testable."""

from __future__ import annotations

import math

import pytest

from minijev.confidence import (
    CONFIDENCE_MEASURES,
    DEFAULT_MEASURE,
    confidence_from,
    distribution_shape,
    effective_options,
    entropy,
    margin,
    normalized_entropy,
    top_probability,
    uniform_confidence,
)


def test_a_one_hot_distribution_is_maximally_confident():
    for measure in CONFIDENCE_MEASURES:
        assert confidence_from((1.0, 0.0, 0.0, 0.0), measure) == pytest.approx(1.0)


def test_a_uniform_distribution_is_minimally_confident():
    assert normalized_entropy((0.25, 0.25, 0.25, 0.25)) == pytest.approx(0.0)
    assert uniform_confidence("normalized_entropy") == pytest.approx(0.0)
    assert margin((0.25, 0.25, 0.25, 0.25)) == pytest.approx(0.0)


def test_entropy_and_effective_options_agree():
    assert entropy((1.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0)
    assert effective_options((1.0, 0.0, 0.0, 0.0)) == pytest.approx(1.0)
    assert effective_options((0.25, 0.25, 0.25, 0.25)) == pytest.approx(4.0)
    assert entropy((0.25, 0.25, 0.25, 0.25)) == pytest.approx(math.log(4))


def test_the_measures_disagree_on_purpose():
    """A flattish spread over four options: max-prob calls it confident, entropy does not."""
    spread = (0.40, 0.30, 0.20, 0.10)
    assert top_probability(spread) == pytest.approx(0.40)
    assert normalized_entropy(spread) < 0.15
    assert margin(spread) == pytest.approx(0.10)


def test_every_measure_is_a_monotone_response_to_concentration():
    """Grading a distribution towards one option must never lower any measure."""
    series = [
        (0.25, 0.25, 0.25, 0.25),
        (0.40, 0.25, 0.20, 0.15),
        (0.60, 0.20, 0.15, 0.05),
        (0.85, 0.10, 0.04, 0.01),
        (1.0, 0.0, 0.0, 0.0),
    ]
    for measure, function in CONFIDENCE_MEASURES.items():
        values = [function(row) for row in series]
        assert values == sorted(values), f"{measure} is not monotone in concentration"


def test_unnormalised_input_is_normalised_not_rejected():
    """Logits are commonly fed here by mistake; silently renormalising is friendlier than a wrong answer."""
    assert confidence_from((4.0, 0.0, 0.0, 0.0)) == confidence_from((1.0, 0.0, 0.0, 0.0))
    shape = distribution_shape((2.0, 2.0, 2.0, 2.0))
    assert shape["top_probability"] == pytest.approx(0.25)


def test_zero_vector_does_not_divide_by_zero():
    assert confidence_from((0.0, 0.0, 0.0, 0.0)) == pytest.approx(0.0)


def test_distribution_shape_reports_everything_a_caller_might_threshold_on():
    shape = distribution_shape((0.7, 0.2, 0.05, 0.05), "top_probability")
    assert shape["measure"] == "top_probability"
    assert shape["confidence"] == pytest.approx(0.7)
    assert shape["runner_up"] == pytest.approx(0.2)
    assert shape["margin"] == pytest.approx(0.5)
    assert shape["uniform_probability"] == pytest.approx(0.25)
    assert 1.0 < shape["effective_options"] < 3.0
    assert set(shape) >= {"confidence", "entropy_nats", "normalized_entropy", "top_probability", "effective_options"}


def test_unknown_measure_is_refused():
    with pytest.raises(KeyError):
        confidence_from((1.0, 0.0, 0.0, 0.0), "vibes")


def test_default_measure_is_the_scale_free_one():
    """max-probability is not comparable across different option counts, entropy is."""
    assert DEFAULT_MEASURE == "normalized_entropy"
    two_way = top_probability((0.7, 0.3))
    four_way = top_probability((0.7, 0.1, 0.1, 0.1))
    assert two_way == pytest.approx(four_way), "top probability ignores how many options there are"
    assert normalized_entropy((0.7, 0.3)) != pytest.approx(normalized_entropy((0.7, 0.1, 0.1, 0.1)))
