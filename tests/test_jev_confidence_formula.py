"""The confidence formula, recovered from the live Jev API and pinned against real responses.

The published documentation could not settle this. Its `choice` and `score` examples carry
placeholder task ids (`taxxxxxxxx-...`), so their `confidence` values are illustrative; no simple
statistic fits them. Probing the live API did settle it, across 34 responses and K = 2..8:

    confidence = (max(p) - 1/K) / (1 - 1/K)

The fixtures in `data/external/jev/` are real responses. The tests below assert the formula against
them rather than against a re-derivation, so a regression in either direction is caught.

Why this matters beyond tidiness: MiniJev derives confidence from the distribution and the risk
ladder thresholds that number to decide escalation. If the statistic differs from the one the Jev
contract reports, then MiniJev's gate values mean something different from the API's, and every
calibration number measured so far is about a quantity nobody else consumes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from tinygrad import Tensor, dtypes

from minijev.confidence import (
    CHANCE_CORRECTED_SOURCE,
    CONFIDENCE_MEASURES,
    DIFFERENTIABLE_MEASURES,
    chance_corrected_top,
    confidence_from,
    differentiable_confidence,
)

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "external" / "jev"


def _records() -> list[dict]:
    records: list[dict] = []
    for name in ("confidence_probe.json", "ambiguity_probe.json"):
        path = FIXTURES / name
        if path.exists():
            records.extend(json.loads(path.read_text(encoding="utf-8"))["records"])
    return records


RECORDS = _records()


# --- the formula itself ---------------------------------------------------------


def test_a_uniform_distribution_scores_zero():
    """The chance correction is the point: an uninformative read must not look confident."""
    assert chance_corrected_top([0.25, 0.25, 0.25, 0.25]) == pytest.approx(0.0)
    assert chance_corrected_top([0.5, 0.5]) == pytest.approx(0.0)


def test_certainty_scores_one_at_any_option_count():
    for count in (2, 3, 4, 8, 20):
        values = [0.0] * count
        values[0] = 1.0
        assert chance_corrected_top(values) == pytest.approx(1.0)


def test_a_fixed_winning_probability_scores_higher_with_more_options():
    """A skill score, so the same certainty is worth more when there is more to choose from.

    Direction taken from the computed values, not from intuition: an earlier version of this test
    asserted the opposite and failed. 0.7 among 2 options is 0.40; among 4 it is 0.60.
    """
    assert chance_corrected_top([0.7, 0.3]) == pytest.approx(0.4)
    assert chance_corrected_top([0.7, 0.1, 0.1, 0.1]) == pytest.approx(0.6)
    assert chance_corrected_top([0.7, 0.1, 0.1, 0.1]) > chance_corrected_top([0.7, 0.3])
    # `top_probability` cannot express the distinction at all.
    assert confidence_from([0.7, 0.3], "top_probability") == confidence_from([0.7, 0.1, 0.1, 0.1], "top_probability")


def test_it_differs_from_the_raw_maximum_and_from_entropy():
    values = [0.0, 0.2, 0.8]
    assert chance_corrected_top(values) == pytest.approx(0.7)
    assert confidence_from(values, "top_probability") == pytest.approx(0.8)
    assert confidence_from(values, "normalized_entropy") == pytest.approx(0.5445, abs=0.001)


def test_a_single_option_is_certainty_not_a_division_by_zero():
    assert chance_corrected_top([1.0]) == pytest.approx(1.0)


def test_it_is_registered_for_measurement_and_for_training():
    assert "chance_corrected_top" in CONFIDENCE_MEASURES
    assert "chance_corrected_top" in DIFFERENTIABLE_MEASURES


# --- against the real API -------------------------------------------------------


@pytest.mark.skipif(not RECORDS, reason="no captured Jev fixtures on disk")
def test_the_formula_matches_every_captured_response():
    """Real responses, not documentation examples."""
    assert len(RECORDS) >= 30, f"expected the full probe set, found {len(RECORDS)}"
    worst = 0.0
    for record in RECORDS:
        values = list(record["probabilities"].values())
        worst = max(worst, abs(chance_corrected_top(values) - record["reported_confidence"]))
    # The residual is the API printing probabilities to two decimals while computing confidence
    # from unrounded values; see the rounding test below.
    assert worst < 0.01, f"the formula no longer fits the captured responses (worst {worst:.4f})"


@pytest.mark.skipif(not RECORDS, reason="no captured Jev fixtures on disk")
def test_the_residual_is_explained_by_rounding_of_displayed_probabilities():
    """Back-solving the winner from the reported confidence reproduces the displayed maximum.

    This is what rules out a different formula that merely happens to be close: a wrong formula
    would not land inside the rounding interval every time.
    """
    compared = 0
    for record in RECORDS:
        count = record["k"]
        if count < 2:
            continue
        implied = record["reported_confidence"] * (1 - 1 / count) + 1 / count
        displayed = max(record["probabilities"].values())
        assert abs(round(implied, 2) - displayed) < 1e-9, (record["probabilities"], record["reported_confidence"])
        compared += 1
    assert compared >= 20


@pytest.mark.skipif(not RECORDS, reason="no captured Jev fixtures on disk")
def test_it_beats_the_measures_minijev_used_before():
    """The comparison that says the old default was wrong, not merely different."""
    errors = {name: 0.0 for name in (*CONFIDENCE_MEASURES, "chance_corrected_top")}
    for record in RECORDS:
        values = list(record["probabilities"].values())
        reported = record["reported_confidence"]
        for name in CONFIDENCE_MEASURES:
            errors[name] += abs(confidence_from(values, name) - reported)
        errors["chance_corrected_top"] += abs(chance_corrected_top(values) - reported)
    total = len(RECORDS)
    means = {name: value / total for name, value in errors.items()}
    assert means["chance_corrected_top"] < means["normalized_entropy"] / 10
    assert means["chance_corrected_top"] < means["top_probability"] / 10
    assert means["chance_corrected_top"] < means["margin"] / 5


@pytest.mark.skipif(not RECORDS, reason="no captured Jev fixtures on disk")
def test_the_old_default_was_materially_wrong_not_merely_different():
    """Quantifies what switching the default would change, so the decision is informed.

    On the ambiguous probe, `normalized_entropy` reports 0.071 where the API reports 0.16, and
    0.769 where the API reports 0.89. The risk ladder's gates are thresholds on that number.
    """
    ambiguous = [record for record in RECORDS if max(record["probabilities"].values()) < 0.95]
    assert ambiguous, "the fixture set should include ambiguous distributions"
    gaps = [
        abs(confidence_from(list(record["probabilities"].values()), "normalized_entropy") - record["reported_confidence"]) for record in ambiguous
    ]
    assert max(gaps) > 0.10, "if the measures agreed, switching the default would be cosmetic"


# --- the differentiable form ----------------------------------------------------


def test_the_differentiable_form_equals_the_scalar_one():
    rows = [[4.0, 0.0, -1.0], [0.5, 0.4, 0.3], [0.0, 0.0, 0.0], [10.0, 9.9, -5.0]]
    tensor = Tensor(rows, dtype=dtypes.float32)
    differentiable = differentiable_confidence(tensor, "chance_corrected_top").numpy().flatten().tolist()
    for row, value in zip(rows, differentiable):
        exponentials = [pow(2.718281828459045, item) for item in row]
        total = sum(exponentials)
        probabilities = [item / total for item in exponentials]
        assert value == pytest.approx(chance_corrected_top(probabilities), abs=1e-5)


def test_the_measure_records_where_it_came_from():
    assert "jev" in CHANCE_CORRECTED_SOURCE.lower()
    assert "API" in CHANCE_CORRECTED_SOURCE
