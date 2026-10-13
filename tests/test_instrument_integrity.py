"""The instrument's own integrity checks must be able to fail.

Two failures motivated this file:

1. **A check that cannot fail.** The entity-leakage check read `metadata["object"]`, found
   nothing, compared two empty sets and reported success. The key is `metadata["frame"]["object"]`;
   `object_of` now reads it from the one place it lives, and the tests below assert the accessor
   returns real names rather than that a comparison of empty sets is empty.
2. **A shortcut nobody measured.** A content-free classifier fitted on the training split reached
   0.396 on the implied benchmark against a 0.316 majority baseline. That gap is now computed into
   the manifest on every generation instead of living in an ad-hoc script.
"""

from __future__ import annotations

from collections import Counter

import pytest

from minijev.frame_product import (
    ProductOptions,
    entity_overlap,
    generate_frame_product,
    object_of,
    product_contrast_pairs,
    product_rows,
    shortcut_audit,
)
from minijev.schema import ROUTE_ORDER, JevSample, Route


@pytest.fixture(scope="module")
def splits():
    options = ProductOptions(naming="implied", benchmark_fraction=0.4, seed=17)
    product_set = generate_frame_product(options)
    return product_rows(product_set, split="train"), product_rows(product_set, split="benchmark")


def test_the_object_accessor_reads_a_real_name(splits):
    """Guards the vacuous check: an accessor returning None everywhere would pass an overlap test."""
    train, _ = splits
    names = [object_of(row) for row in train]
    assert all(isinstance(name, str) and name for name in names), "object_of returned nothing for some row"
    assert len(set(names)) > 5, "an accessor that returns one name makes every overlap check trivially pass"


def test_entities_are_disjoint_across_the_split(splits):
    train, benchmark = splits
    report = entity_overlap(train, benchmark)
    assert report["train_objects"] > 0 and report["benchmark_objects"] > 0
    assert report["shared_objects"] == [], report["shared_objects"]
    assert report["shared_text"] == 0


def test_the_overlap_check_would_notice_an_entity_on_both_sides(splits):
    """The check must be falsifiable: feed it a leak and it has to report one."""
    train, benchmark = splits
    leaked = JevSample(
        query=benchmark[0].query + "（泄漏）",
        confidence=1.0,
        route=benchmark[0].route,
        metadata={"frame": {"object": object_of(train[0])}},
    )
    report = entity_overlap(train, [*benchmark, leaked])
    assert report["shared_objects"] == [object_of(train[0])]


def test_the_shortcut_audit_reports_the_baselines_it_compares(splits):
    train, benchmark = splits
    audit = shortcut_audit(train, benchmark)
    for key in ("majority_accuracy", "marker_only_accuracy", "marker_plus_length_accuracy", "chance_accuracy"):
        assert key in audit, key
    assert audit["chance_accuracy"] == pytest.approx(1 / len(ROUTE_ORDER))
    assert audit["shortcut_gap_over_majority"] == pytest.approx(audit["marker_plus_length_accuracy"] - audit["majority_accuracy"])


def test_no_content_free_classifier_solves_the_benchmark(splits):
    """The explicit product failed exactly this: markers alone scored 1.0000.

    The implied split cannot be driven to certainty from surface cues. It is not free of them --
    a step clause is a genuine signal -- so the bar is a ceiling well below the model's own
    accuracy, not equality with chance.
    """
    train, benchmark = splits
    audit = shortcut_audit(train, benchmark)
    assert audit["marker_plus_length_accuracy"] < 0.60, "a content-free rule is approaching the model's own accuracy"
    assert audit["shortcut_gap_over_majority"] < 0.15, "the gap is large enough to flatter every result"


def test_a_shortcut_audit_on_a_leaky_corpus_reports_a_shortcut():
    """Falsifiability again: hand it a corpus where the answer is in the surface form."""

    def make(text: str, route: Route) -> JevSample:
        return JevSample(query=text, confidence=1.0, route=route, metadata={})

    train = [make("先再然后做一件事", Route.CLOUD) for _ in range(20)] + [make("查东西", Route.LOCAL) for _ in range(20)]
    benchmark = [make("先再然后做另一件事", Route.CLOUD) for _ in range(10)] + [make("查别的", Route.LOCAL) for _ in range(10)]
    audit = shortcut_audit(train, benchmark)
    assert audit["marker_only_accuracy"] == pytest.approx(1.0)
    assert audit["shortcut_gap_over_majority"] == pytest.approx(0.5)


def test_the_benchmark_has_enough_pairs_to_resolve_a_fifteen_point_difference(splits):
    """Pins the instrument's resolution so a later change cannot quietly shrink it.

    Two-proportion power at 95% confidence and 80% power from a 30% collapse baseline needs
    roughly 120 pairs for a 15-point drop and 294 for 10 points. This split clears the first.
    """
    _, benchmark = splits
    pairs = product_contrast_pairs(generate_frame_product(ProductOptions(naming="implied", benchmark_fraction=0.4, seed=17)))
    assert len(pairs) >= 120, f"only {len(pairs)} contrast pairs"


def test_the_routes_stay_balanced_after_expanding_the_object_table():
    """The table was expanded to lift the binding route; balance is what makes the quota fair."""
    product_set = generate_frame_product(ProductOptions(naming="implied", benchmark_fraction=0.4, seed=17))
    counts = Counter(row.route.value for row in product_set.cases and product_rows(product_set, split="all"))
    assert set(counts) == {route.value for route in ROUTE_ORDER}
    assert len(set(counts.values())) == 1, f"route counts drifted apart: {counts}"
