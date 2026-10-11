"""Metrics must make a dead confidence head and a minority class visible."""

from __future__ import annotations

import pytest

from minijev.evaluation import (
    classification_report,
    confidence_audit,
    confusion_matrix,
    expected_calibration_error,
    majority_baseline,
)
from minijev.schema import JevSample, Route


def test_confusion_matrix_counts_every_pair():
    matrix = confusion_matrix([Route.LOCAL, Route.RAG], [Route.LOCAL, Route.LOCAL])
    assert matrix["local"]["local"] == 1
    assert matrix["rag"]["local"] == 1
    assert matrix["cloud"]["review"] == 0


def test_classification_report_computes_macro_f1():
    actual = [Route.LOCAL, Route.LOCAL, Route.RAG, Route.RAG]
    predicted = [Route.LOCAL, Route.LOCAL, Route.RAG, Route.LOCAL]
    report = classification_report(actual, predicted)
    assert report["accuracy"] == pytest.approx(0.75)
    assert report["per_class"]["local"]["precision"] == pytest.approx(2 / 3)
    assert report["per_class"]["local"]["recall"] == pytest.approx(1.0)
    assert report["macro_f1"] == pytest.approx((report["per_class"]["local"]["f1"] + report["per_class"]["rag"]["f1"]) / 4)


def test_report_restricted_to_supported_labels_can_carry_no_zero_from_an_absent_route():
    """MASSIVE has no cloud samples; a full four-class macro average would be dragged down by a class nothing predicted."""
    actual = [Route.LOCAL, Route.RAG, Route.REVIEW]
    predicted = [Route.LOCAL, Route.RAG, Route.REVIEW]
    full = classification_report(actual, predicted)
    restricted = classification_report(actual, predicted, labels=(Route.LOCAL, Route.RAG))
    assert full["unrepresented_labels"] == ["cloud"]
    assert restricted["unrepresented_labels"] == []
    assert restricted["macro_f1"] > full["macro_f1"]


def test_scoping_drops_out_of_scope_rows_and_counts_stray_predictions_as_errors():
    actual = [Route.LOCAL, Route.RAG, Route.LOCAL, Route.CLOUD, Route.REVIEW]
    predicted = [Route.LOCAL, Route.RAG, Route.REVIEW, Route.CLOUD, Route.REVIEW]

    scoped = classification_report(actual, predicted, labels=(Route.LOCAL, Route.RAG), scope_to_labels=True)
    assert scoped["samples"] == 3, "cloud and review rows must leave the scope"
    assert scoped["out_of_scope_predictions"] == 1, "the local row predicted as review is a miss"
    assert scoped["accuracy"] == pytest.approx(2 / 3)
    assert set(scoped["confusion_matrix"]) == {"local", "rag"}

    unscoped = classification_report(actual, predicted, labels=(Route.LOCAL, Route.RAG))
    assert unscoped["samples"] == 5
    assert unscoped["scoped_to_labels"] is False


def test_majority_baseline_is_reported_for_a_real_comparison():
    samples = [JevSample(query=f"local {index}", confidence=1.0, route=Route.LOCAL) for index in range(7)] + [
        JevSample(query="rag 1", confidence=1.0, route=Route.RAG)
    ]
    baseline = majority_baseline(samples)
    assert baseline["majority_route"] == "local"
    assert baseline["accuracy"] == pytest.approx(7 / 8)


def test_majority_baseline_fits_on_training_not_on_the_evaluation_set():
    """Fitting the baseline on the test labels would let it read the answer key."""
    train = [JevSample(query=f"t{index}", confidence=1.0, route=Route.RAG) for index in range(9)] + [
        JevSample(query="one local", confidence=1.0, route=Route.LOCAL)
    ]
    test = [JevSample(query=f"e{index}", confidence=1.0, route=Route.LOCAL) for index in range(6)]

    honest = majority_baseline(test, training_samples=train)
    assert honest["majority_route"] == "rag"
    assert honest["accuracy"] == pytest.approx(0.0)
    assert honest["fitted_on"] == "training set"

    optimistic = majority_baseline(test)
    assert optimistic["majority_route"] == "local"
    assert optimistic["accuracy"] == pytest.approx(1.0)
    assert "optimistic" in optimistic["fitted_on"]


def test_a_collapsed_confidence_head_is_flagged_even_when_accuracy_looks_fine():
    """This is the exact failure the audit exists to catch."""
    confidences = [0.917] * 20
    correct = [True] * 18 + [False] * 2
    audit = confidence_audit(confidences, correct)
    assert audit["collapsed"] is True
    assert audit["accuracy"] == pytest.approx(0.9)
    assert audit["spread"] < 0.01


def test_a_varied_confidence_head_is_not_flagged():
    confidences = [0.1, 0.4, 0.6, 0.9] * 5
    correct = [False, False, True, True] * 5
    audit = confidence_audit(confidences, correct)
    assert audit["collapsed"] is False
    assert audit["spread"] == pytest.approx(0.8)


def test_ece_is_zero_for_a_perfectly_calibrated_head():
    confidences = [0.0, 0.0, 1.0, 1.0]
    correct = [False, False, True, True]
    assert expected_calibration_error(confidences, correct) == pytest.approx(0.0, abs=1e-9)


def test_ece_is_large_for_an_overconfident_head():
    confidences = [1.0] * 10
    correct = [False] * 10
    assert expected_calibration_error(confidences, correct) == pytest.approx(1.0)


def test_audit_handles_an_empty_split():
    assert confidence_audit([], [])["samples"] == 0
    assert expected_calibration_error([], []) == 0.0
