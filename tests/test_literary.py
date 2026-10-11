"""The literary supplement is a controlled experiment; its mechanics must hold."""

from __future__ import annotations

import pytest

from minijev.literary import generate
from minijev.schema import Route


def test_generates_balanced_grouped_rows():
    rows = generate(variants=3)
    assert len(rows) == 144
    assert {row.route for row in rows} == set(Route)
    assert all(sum(1 for row in rows if row.route is route) == 36 for route in Route)


def test_all_three_variants_are_distinct():
    rows = generate(variants=3)
    by_group: dict[str, list[str]] = {}
    for row in rows:
        by_group.setdefault(str(row.metadata["semantic_group_id"]), []).append(row.query)
    assert len(by_group) == 48
    assert all(len(set(queries)) == 3 for queries in by_group.values()), "every variant must differ in wording"


def test_labels_come_from_the_operation_not_the_work():
    rows = generate(variants=3)
    assert {str(row.metadata["label_note"]) for row in rows} == {"route is assigned from requested operation, not from the work itself"}
    # The same excerpt appears under all four routes.
    first_work = rows[0].metadata["literary_work"]
    same_work = [row.route for row in rows if row.metadata["literary_work"] == first_work]
    assert set(same_work) == set(Route)


def test_variant_count_must_be_positive():
    with pytest.raises(ValueError):
        generate(variants=0)
