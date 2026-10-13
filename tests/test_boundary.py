"""The boundary benchmark must be able to tell separation from memorisation.

These tests drive `evaluate_boundary` with stub routers whose behaviour is known, so the
pairing analysis is checked against ground truth rather than against a model's opinion.
A router that always answers the same route must show a 100% collapse rate even if its
raw accuracy is respectable, because that is precisely the failure the benchmark exists
to expose.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from minijev.boundary import evaluate_boundary, format_report
from minijev.frame_product import generate_frame_product, product_rows
from minijev.frames import (
    BENCHMARK_FRAME_IDS,
    CONTRAST_PAIRS,
    RouteFrame,
    deciding_slots_differing,
    generate_benchmark,
    is_minimal_pair,
)
from minijev.frames import _frame as bank_frame
from minijev.routers import Router
from minijev.schema import ROUTE_ORDER, Decision, Route, RouteScores


class StubRouter(Router):
    """A router that answers from a fixed table, for testing the analyser itself."""

    name = "stub"

    def __init__(self, mapping: dict[str, Route], default: Route = Route.LOCAL):
        self.mapping = mapping
        self.default = default

    def score(self, queries: Sequence[str]) -> list[RouteScores]:
        records = []
        for query in queries:
            route = self.mapping.get(query, self.default)
            records.append(RouteScores(route_logits=tuple(1.0 if candidate is route else 0.0 for candidate in ROUTE_ORDER)))
        return records

    def decide(self, queries: Sequence[str]) -> list[Decision]:
        decisions = []
        for query in queries:
            route = self.mapping.get(query, self.default)
            logits = tuple(1.0 if candidate is route else 0.0 for candidate in ROUTE_ORDER)
            decisions.append(Decision(route=route, confidence=1.0, route_logits=logits, route_probs=logits, policy="stub"))
        return decisions


@pytest.fixture
def benchmark():
    return generate_benchmark(variants=3)


def test_a_perfect_router_separates_every_pair(benchmark):
    exact = {row.query: row.route for row in benchmark}
    result = evaluate_boundary(StubRouter(exact), benchmark)
    assert result["accuracy"] == 1.0
    assert result["pairs"]["both_correct"] == 1.0
    assert result["pairs"]["collapsed"] == 0.0


def test_a_constant_router_collapses_every_pair_even_when_accuracy_looks_fine(benchmark):
    """The headline failure: one answer for every question."""
    result = evaluate_boundary(StubRouter({}, default=Route.REVIEW), benchmark)
    assert result["pairs"]["collapsed"] == 1.0
    assert result["pairs"]["both_correct"] == 0.0
    assert result["minimal_pairs"]["collapsed"] == 1.0
    # Accuracy is set entirely by how many rows happen to be review.
    review_share = sum(1 for row in benchmark if row.route is Route.REVIEW) / len(benchmark)
    assert result["accuracy"] == pytest.approx(review_share)


def test_a_router_that_swaps_the_pair_is_not_collapsed(benchmark):
    """Getting both wrong in opposite directions still shows the slot was read."""
    swapped = {row.query: _other(row.route) for row in benchmark}
    result = evaluate_boundary(StubRouter(swapped), benchmark)
    assert result["pairs"]["neither_correct"] == 1.0
    assert result["pairs"]["collapsed"] == 0.0


def test_pairs_are_only_counted_when_both_halves_are_present(benchmark):
    """A half pair cannot measure separation, so it must not be counted."""
    half = [row for row in benchmark if row.metadata["frame_id"] not in {CONTRAST_PAIRS[0][1]}]
    result = evaluate_boundary(StubRouter({}), half)
    assert all(outcome["both_correct"] is not None for outcome in result["outcomes"])
    assert not any(outcome["right"] == CONTRAST_PAIRS[0][1] and outcome["left"] == CONTRAST_PAIRS[0][0] for outcome in result["outcomes"])


def test_slot_accuracy_is_reported_for_every_deciding_slot(benchmark):
    result = evaluate_boundary(StubRouter({}, default=Route.LOCAL), benchmark)
    assert set(result["slot_accuracy"]) == {"risk", "underspecified", "derivation", "steps", "source_class"}
    for buckets in result["slot_accuracy"].values():
        assert buckets, "every deciding slot must have at least one value in the benchmark"


def test_per_route_recall_is_reported_for_every_route(benchmark):
    result = evaluate_boundary(StubRouter({}), benchmark)
    assert set(result["per_route_recall"]) == {route.value for route in Route}


def test_the_report_mentions_collapse_and_separation(benchmark):
    text = format_report("stub", evaluate_boundary(StubRouter({}), benchmark))
    assert "collapsed=" in text
    assert "both_correct=" in text
    assert "minimal pairs" in text


def _other(route: Route) -> Route:
    """The next route in canonical order. A bijection, so swapping never collapses."""
    return ROUTE_ORDER[(ROUTE_ORDER.index(route) + 1) % len(ROUTE_ORDER)]


# --- the pair definitions themselves ----------------------------------------


def test_every_contrast_pair_is_wholly_held_out():
    """Both halves must be in the benchmark, or the pair cannot be scored."""
    for left, right in CONTRAST_PAIRS:
        assert left in BENCHMARK_FRAME_IDS, f"{left} is trained on but paired with a benchmark frame"
        assert right in BENCHMARK_FRAME_IDS, f"{right} is trained on but paired with a benchmark frame"


def test_every_contrast_pair_actually_spans_two_routes():
    from minijev.frames import route_from_frame

    for left, right in CONTRAST_PAIRS:
        left_frame, right_frame = bank_frame(left), bank_frame(right)
        assert route_from_frame(left_frame) is not route_from_frame(right_frame), f"{left}/{right} share a route, so they are not a contrast"


def test_the_benchmark_contains_minimal_pairs_for_several_slots():
    """A minimal pair isolates one slot, so a failure is attributable."""
    slots: set[str] = set()
    for left, right in CONTRAST_PAIRS:
        left_frame, right_frame = bank_frame(left), bank_frame(right)
        if is_minimal_pair(left_frame, right_frame):
            slots.update(deciding_slots_differing(left_frame, right_frame))
    assert {"risk", "source_class", "derivation", "steps", "underspecified"} <= slots, slots


def test_deciding_slots_ignore_surface_wording():
    """Two phrasings of the same decision differ in nothing that can change a route."""
    left = RouteFrame("分析", "甲方案", source=bank_frame("f01").source, derivation=bank_frame("f01").derivation)
    right = RouteFrame("审视", "乙方案", source=bank_frame("f01").source, derivation=bank_frame("f01").derivation)
    assert deciding_slots_differing(left, right) == ()


def test_an_empty_benchmark_file_yields_a_zero_report():
    result = evaluate_boundary(StubRouter({}), [])
    assert result["rows"] == 0
    assert result["pairs"]["pairs"] == 0


def test_product_benchmark_derives_disjoint_pairs_automatically():
    """A perfect router must separate every derived pair, whatever the split size is.

    The row count used to be asserted as a literal, which turned an incidental snapshot
    into a failure every time the split changed (reserving whole *objects* for the
    benchmark moved it from 960 to 1167). The invariants are what matter: pairs are found,
    they are disjoint, and a router that knows the answers collapses none of them.
    """
    product = generate_frame_product()
    product_rows_ = product_rows(product, split="benchmark")
    assert product_rows_, "the benchmark must not be empty"
    exact = {row.query: row.route for row in product_rows_}
    result = evaluate_boundary(StubRouter(exact), product_rows_)
    assert result["rows"] == len(product_rows_)
    assert result["groups"] == len(product.benchmark_cases)
    assert result["minimal_pairs"]["pairs"] >= 100
    assert result["minimal_pairs"]["collapsed"] == 0.0
