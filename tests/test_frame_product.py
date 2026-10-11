"""The slot product expands boundary evidence without leaking semantic groups."""

from __future__ import annotations

from collections import Counter

from minijev.data import group_leaks
from minijev.frame_product import ProductOptions, generate_frame_product, product_audit, product_contrast_pairs, product_rows
from minijev.frames import RouteFrame, route_from_frame
from minijev.schema import ROUTE_ORDER


def test_product_is_deterministic_and_route_balanced():
    options = ProductOptions(target_per_route=12, benchmark_fraction=0.25, variants=2, seed=23)
    first = generate_frame_product(options)
    second = generate_frame_product(options)
    assert first == second
    counts = Counter(route_from_frame(case.frame).value for case in first.cases)
    assert counts == {route.value: 12 for route in ROUTE_ORDER}


def test_product_rows_keep_route_derived_from_frame_and_split_groups_intact():
    product = generate_frame_product(ProductOptions(target_per_route=16, benchmark_fraction=0.3, variants=3, seed=9))
    train = product_rows(product, split="train")
    benchmark = product_rows(product, split="benchmark")
    assert not group_leaks((train, benchmark))
    assert {row.query for row in train}.isdisjoint({row.query for row in benchmark})
    for row in train + benchmark:
        assert row.route.value == route_from_frame(RouteFrame.from_dict(row.metadata["frame"])).value
        assert row.metadata["generator"] == "minijev.frame_product"


def test_product_covers_all_deciding_slots():
    product = generate_frame_product(ProductOptions(target_per_route=20, benchmark_fraction=0.25, variants=1, seed=4))
    frames = [case.frame for case in product.cases]
    assert {frame.source.value for frame in frames} == {"given_text", "device", "user_data", "document", "service", "none"}
    assert {frame.derivation.value for frame in frames} == {"extract", "construct"}
    assert {frame.steps for frame in frames} == {1, 2, 3}
    assert {frame.risk.value for frame in frames} == {"low", "medium", "high"}
    assert {frame.underspecified for frame in frames} == {False, True}


def test_default_product_is_thousand_scale_and_audit_is_consistent():
    product = generate_frame_product()
    audit = product_audit(product)
    assert audit["frames"] >= 1000
    assert audit["frames"] == sum(audit["route_counts"].values())
    assert audit["train_frames"] + audit["benchmark_frames"] == audit["frames"]
    assert audit["unique_queries"] == audit["rows"]
    assert audit["minimal_pairs"] >= 100


def test_product_pairs_are_disjoint_minimal_contrasts():
    product = generate_frame_product(ProductOptions(target_per_route=24, benchmark_fraction=0.25, variants=1, seed=17))
    pairs = product_contrast_pairs(product)
    used = {frame_id for pair in pairs for frame_id in pair}
    assert len(used) == 2 * len(pairs)
    assert pairs
