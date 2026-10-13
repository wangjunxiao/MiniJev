"""Calibration and gating are different questions, and the metrics must say so.

Reference: Fisch, Jaakkola & Barzilay, *Calibrated Selective Classification* (arXiv 2208.12084).
A selective model "may still allow for wrong predictions that have high confidence, or skip
correct predictions that have low confidence". MiniJev's risk ladder is exactly such a selector,
so the confidence has to be judged on whether it *ranks* errors, not only on whether it reports
honest probabilities.

The failure this file is built to catch is a report that shows a beautiful ECE while the gate does
nothing: constant confidence is perfectly calibrated at the base rate and completely useless for
deciding what to accept.
"""

from __future__ import annotations

import pytest

from minijev.evaluation import (
    area_under_risk_coverage,
    coverage_at_risk,
    risk_at_coverage,
    risk_coverage_curve,
    selective_audit,
)

# A perfectly ranked case: the three correct predictions are the most confident.
PERFECT_CONFIDENCE = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
PERFECT_CORRECT = [True, True, True, False, False, False]
BASE_RISK = 0.5


def test_the_curve_starts_at_the_most_confident_row():
    curve = risk_coverage_curve(PERFECT_CONFIDENCE, PERFECT_CORRECT)
    assert curve[0][0] == pytest.approx(1 / 6)
    assert curve[0][1] == pytest.approx(0.0), "the most confident row is correct"
    assert curve[-1][0] == pytest.approx(1.0)
    assert curve[-1][1] == pytest.approx(BASE_RISK), "accepting everything gives the base rate"


def test_a_perfect_ranking_has_zero_risk_until_it_runs_out_of_correct_answers():
    curve = risk_coverage_curve(PERFECT_CONFIDENCE, PERFECT_CORRECT)
    for coverage, risk in curve[:3]:
        assert risk == pytest.approx(0.0), coverage
    assert curve[3][1] == pytest.approx(0.25)


def test_a_perfect_ranking_beats_random_by_the_full_base_risk():
    aurc = area_under_risk_coverage(PERFECT_CONFIDENCE, PERFECT_CORRECT)
    assert aurc < BASE_RISK
    audit = selective_audit(PERFECT_CONFIDENCE, PERFECT_CORRECT)
    assert audit["skill_vs_random"] > 0.5


def test_a_constant_confidence_is_perfectly_calibrated_and_still_a_useless_gate():
    """The headline failure mode, stated without the tie-order confusion.

    A flat confidence at the base rate has zero ECE -- it is honest -- and its gating skill is not
    identifiable from one ordering (see the averaging test below). What is unambiguous is that
    being calibrated says nothing about being a useful gate, which is why `selective_audit`
    reports both families of numbers.
    """
    flat = [0.5] * 6
    audit = selective_audit(flat, PERFECT_CORRECT)
    assert audit["ece_all"] == pytest.approx(0.0)
    # The curve is a flat line at the base rate in expectation ...
    assert audit["aurc"] == pytest.approx(0.1917, abs=0.01)
    # ... which is not the same as the random-ranking null, because ties are broken by position.
    assert audit["no_skill_aurc"] == pytest.approx(BASE_RISK, abs=0.02)


def test_a_collapsed_head_has_no_skill_on_average_across_input_orders():
    """A tied confidence has no *identifiable* skill from a single ordering.

    With every confidence equal, the risk-coverage curve is decided entirely by how ties are
    broken, so one run can look skilful or anti-skiful purely by luck of input order -- on this
    file's fixture a flat confidence scores +0.62 skill because the tie-break happens to put the
    three correct rows first. That is a genuine limitation of reading skill for a tied head rather
    than a bug in the metric, and the only honest reading averages over orderings.
    """
    import random as _random
    import statistics

    size = 200
    labels = [index % 2 == 0 for index in range(size)]
    skills = []
    for seed in range(40):
        order = list(range(size))
        _random.Random(seed).shuffle(order)
        shuffled = [labels[index] for index in order]
        skills.append(selective_audit([0.5] * size, shuffled)["skill_vs_random"])
    assert statistics.mean(skills) == pytest.approx(0.0, abs=0.05), statistics.mean(skills)


def test_the_no_skill_reference_is_the_base_error_rate_in_expectation():
    """The null is the base rate on average, and a single tie-broken ordering need not equal it."""
    from minijev.evaluation import random_ranking_aurc

    assert random_ranking_aurc(PERFECT_CORRECT, trials=400) == pytest.approx(BASE_RISK, abs=0.02)


def test_an_anti_correlated_confidence_is_visible_rather_than_clamped():
    """A confidence that ranks *worse* than chance is a real failure, not a zero."""
    inverted = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    audit = selective_audit(inverted, PERFECT_CORRECT)
    assert audit["skill_vs_random"] < 0.0


def test_risk_at_coverage_is_monotone_for_a_useful_confidence():
    risks = [risk_at_coverage(PERFECT_CONFIDENCE, PERFECT_CORRECT, coverage) for coverage in (0.2, 0.5, 0.9, 1.0)]
    assert all(later >= earlier - 1e-9 for earlier, later in zip(risks, risks[1:])), risks


def test_coverage_at_risk_answers_the_deployment_question():
    """ "To hold errors under X, how much must be escalated?"""
    assert coverage_at_risk(PERFECT_CONFIDENCE, PERFECT_CORRECT, 0.0) == pytest.approx(0.5)
    assert coverage_at_risk(PERFECT_CONFIDENCE, PERFECT_CORRECT, 0.5) == pytest.approx(1.0)


def test_coverage_at_risk_is_zero_when_even_the_best_row_is_wrong():
    assert coverage_at_risk([0.9, 0.1], [False, False], 0.10) == pytest.approx(0.0)


def test_the_audit_reports_both_questions():
    audit = selective_audit(PERFECT_CONFIDENCE, PERFECT_CORRECT)
    for key in ("ece_all", "aurc", "skill_vs_random", "risk_at_80pct_coverage", "coverage_at_10pct_risk", "ece_accepted_80pct"):
        assert key in audit, key


def test_selective_calibration_is_measured_on_the_accepted_subset():
    """The paper's property: among what the gate lets through, is confidence still honest?

    Constructed so the accepted four are calibrated at 0.75 (three of four correct) while the
    rejected tail is a correct answer stated at 0.05 -- so the accepted-subset ECE is near zero
    and the all-rows figure is dominated by the row the gate removed.
    """
    confidence = [0.75, 0.75, 0.75, 0.75, 0.05]
    correct = [True, True, True, False, True]
    audit = selective_audit(confidence, correct)
    assert audit["ece_accepted_80pct"] < 0.02, "the accepted four are calibrated"
    assert audit["ece_all"] > 0.15, "the rejected row is badly stated and must show in the all-rows figure"


def test_mismatched_inputs_are_refused():
    with pytest.raises(ValueError):
        risk_coverage_curve([0.5], [True, False])
    with pytest.raises(ValueError):
        selective_audit([0.5], [True, False])
    with pytest.raises(ValueError):
        risk_at_coverage([0.5], [True], 0.0)
    with pytest.raises(ValueError):
        risk_at_coverage([0.5], [True], 1.5)


def test_an_empty_input_is_a_zero_report_rather_than_a_crash():
    assert risk_coverage_curve([], []) == []
    assert area_under_risk_coverage([], []) == 0.0
    assert selective_audit([], [])["rows"] == 0


def test_ties_are_broken_deterministically():
    """A collapsed head emits many identical values; the curve must not depend on dict order."""
    confidence = [0.5] * 100
    correct = [index % 2 == 0 for index in range(100)]
    first = risk_coverage_curve(confidence, correct)
    second = risk_coverage_curve(list(confidence), list(correct))
    assert first == second
