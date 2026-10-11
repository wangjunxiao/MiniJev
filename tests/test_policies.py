"""The policy layer is the only place a route is chosen, and the threshold depends on what is at stake."""

from __future__ import annotations

import pytest

from minijev.confidence import confidence_from
from minijev.policies import (
    DEFAULT_LADDER,
    ArgmaxPolicy,
    FlatThresholdPolicy,
    RiskLadderPolicy,
    RouteGate,
    policy_from_name,
    softmax,
    sweep_flat_threshold,
    sweep_ladder,
)
from minijev.schema import ROUTE_ORDER, Route, RouteScores

# One clear winner among four: normalized-entropy confidence of roughly 0.62.
DECISIVE_LOCAL = (3.0, 0.0, 0.0, 0.0)
# Nearly flat: the model cannot tell local from rag.
HESITANT_LOCAL = (0.6, 0.5, 0.0, 0.0)
# A confident cloud read.
DECISIVE_CLOUD = (0.0, 0.0, 3.0, 0.0)


def test_softmax_is_a_distribution_and_is_shift_invariant():
    probabilities = softmax((1.0, 2.0, 3.0, 4.0))
    assert sum(probabilities) == pytest.approx(1.0)
    assert softmax((101.0, 102.0, 103.0, 104.0)) == pytest.approx(probabilities)


def test_confidence_is_derived_from_the_distribution_not_supplied():
    """The same logits always yield the same confidence, and no head is involved."""
    decision = RiskLadderPolicy().decide(RouteScores(route_logits=DECISIVE_LOCAL))
    # softmax(3,0,0,0) is 0.870 / 0.043 x3, so 1 - H/log(4) is about 0.619.
    assert decision.confidence == pytest.approx(0.619, abs=0.005)
    assert decision.confidence_measure == "normalized_entropy"
    assert len(decision.route_probs) == len(ROUTE_ORDER)


def test_a_decisive_read_is_taken_as_is():
    decision = RiskLadderPolicy().decide(RouteScores(route_logits=DECISIVE_CLOUD))
    assert decision.route is Route.CLOUD
    assert decision.escalated is False
    assert decision.escalated_from is None
    assert decision.top_route is Route.CLOUD


def test_a_hesitant_read_is_escalated_one_step_towards_more_care():
    decision = RiskLadderPolicy().decide(RouteScores(route_logits=HESITANT_LOCAL))
    assert decision.escalated is True
    assert decision.escalated_from is Route.LOCAL
    assert decision.route in {Route.RAG, Route.CLOUD, Route.REVIEW}


def _logit_for_confidence(target: float) -> float:
    """Find the logit magnitude whose one-of-four read scores `target`."""
    low, high = 0.0, 12.0
    for _ in range(60):
        middle = (low + high) / 2
        if confidence_from(softmax((middle, 0.0, 0.0, 0.0))) < target:
            low = middle
        else:
            high = middle
    return high


def test_the_gates_are_ordered_by_what_is_at_stake():
    """A cheap recoverable action accepts readily; an expensive one demands confidence."""
    ladder = DEFAULT_LADDER
    assert ladder[Route.LOCAL].accept < ladder[Route.RAG].accept < ladder[Route.CLOUD].accept
    assert ladder[Route.REVIEW].accept == 0.0
    assert ladder[Route.REVIEW].escalate_to is Route.REVIEW


def test_the_gates_differ_per_route_which_is_the_whole_point():
    """Identical confidence, different route, different outcome."""
    ladder = DEFAULT_LADDER
    # A read that clears the rag gate but not the cloud one.
    target = (ladder[Route.RAG].accept + ladder[Route.CLOUD].accept) / 2
    logit = _logit_for_confidence(target)
    preferred = {Route.LOCAL: (logit, 0.0, 0.0, 0.0), Route.RAG: (0.0, logit, 0.0, 0.0), Route.CLOUD: (0.0, 0.0, logit, 0.0)}
    decisions = {route: RiskLadderPolicy().decide(RouteScores(route_logits=logits)) for route, logits in preferred.items()}

    # Normalized entropy is permutation-invariant, so all three reads are equally confident.
    assert len({round(decision.confidence, 9) for decision in decisions.values()}) == 1
    assert decisions[Route.LOCAL].confidence == pytest.approx(target, abs=1e-6)

    assert decisions[Route.LOCAL].escalated is False, "local is the cheapest action and accepts"
    assert decisions[Route.RAG].escalated is False, "this read clears the rag gate"
    assert decisions[Route.CLOUD].escalated is True, "cloud demands the most and this read does not clear it"
    assert decisions[Route.CLOUD].route is Route.REVIEW


def test_review_is_terminal_so_an_unknown_answer_does_not_bounce():
    flat = (0.0, 0.0, 0.0, 0.0)
    for _ in range(4):  # the ladder must not cycle
        decision = RiskLadderPolicy().decide(RouteScores(route_logits=flat))
    assert decision.route is Route.REVIEW
    assert DEFAULT_LADDER[Route.REVIEW].escalate_to is Route.REVIEW


def test_escalation_is_bounded_even_with_a_cyclic_ladder():
    """A ladder that demands more confidence than is achievable must still terminate."""
    cycle = {route: RouteGate(accept=1.0, escalate_to=Route.REVIEW) for route in ROUTE_ORDER}
    cycle[Route.REVIEW] = RouteGate(accept=1.0, escalate_to=Route.LOCAL)
    policy = RiskLadderPolicy(ladder=cycle, max_escalations=3)
    assert policy.decide(RouteScores(route_logits=(1.0, 0.0, 0.0, 0.0))).route in set(ROUTE_ORDER)


def test_a_ladder_missing_a_route_is_refused_at_construction():
    with pytest.raises(ValueError):
        RiskLadderPolicy(ladder={Route.LOCAL: RouteGate(0.5, Route.RAG)})


def test_gates_reject_out_of_range_thresholds():
    with pytest.raises(ValueError):
        RouteGate(accept=1.5, escalate_to=Route.REVIEW)


def test_caution_scales_every_gate_together():
    base = RiskLadderPolicy()
    cautious = base.scaled(2.0)
    for route in ROUTE_ORDER:
        expected = min(base.ladder[route].accept * 2, 1.0)  # clamped, so a gate cannot exceed a certainty
        assert cautious.ladder[route].accept == pytest.approx(expected)
    assert base.scaled(0.0).decide(RouteScores(route_logits=HESITANT_LOCAL)).escalated is False
    assert base.scaled(99.0).decide(RouteScores(route_logits=DECISIVE_CLOUD)).escalated is True
    assert base.scaled(99.0).ladder[Route.REVIEW].accept == pytest.approx(0.0), "the terminal route stays terminal"


def test_the_flat_policy_cannot_treat_routes_differently():
    """Kept only to measure against; this is the limitation it exists to demonstrate."""
    policy = FlatThresholdPolicy(review_threshold=0.9)
    logits = (1.2, 0.0, 0.0, 0.0)
    flat = policy.decide(RouteScores(route_logits=logits))
    assert flat.route is Route.REVIEW
    assert flat.escalated_from is Route.LOCAL
    # The ladder accepts the same confident read when it lands on review itself.
    assert RiskLadderPolicy().decide(RouteScores(route_logits=logits)).confidence == pytest.approx(flat.confidence)


def test_argmax_ignores_confidence_and_never_escalates():
    decision = ArgmaxPolicy().decide(RouteScores(route_logits=(0.0, 0.0, 0.0, 0.1)))
    assert decision.route is Route.REVIEW
    assert decision.escalated is False


def test_sweeping_caution_reports_the_trade_off_not_one_number():
    scores = [RouteScores(route_logits=DECISIVE_LOCAL), RouteScores(route_logits=HESITANT_LOCAL), RouteScores(route_logits=DECISIVE_CLOUD)]
    expected = [Route.LOCAL, Route.LOCAL, Route.CLOUD]
    best, points = sweep_ladder(RiskLadderPolicy(), scores, expected)
    assert len(points) > 1
    assert {"factor", "accuracy", "escalation_rate"} <= set(points[0])
    assert any(point["escalation_rate"] > 0 for point in points)
    assert best in [point["factor"] for point in points]


def test_sweep_can_be_bounded_by_an_escalation_budget():
    scores = [RouteScores(route_logits=HESITANT_LOCAL)] * 4
    expected = [Route.LOCAL] * 4
    best, points = sweep_ladder(RiskLadderPolicy(), scores, expected, max_escalation_rate=0.0)
    assert all(point["escalation_rate"] == 0.0 for point in points if point["factor"] == best)
    assert best == 0.0


def test_flat_sweep_is_available_for_comparison():
    scores = [RouteScores(route_logits=DECISIVE_LOCAL), RouteScores(route_logits=HESITANT_LOCAL)]
    expected = [Route.LOCAL, Route.LOCAL]
    best, points = sweep_flat_threshold(FlatThresholdPolicy(), scores, expected)
    assert {"threshold", "accuracy", "escalation_rate"} <= set(points[0])
    assert 0.0 <= best <= 1.0


def test_registry_rejects_unknown_policies():
    assert policy_from_name("risk_ladder").name == "risk_ladder"
    with pytest.raises(KeyError):
        policy_from_name("nonexistent")
