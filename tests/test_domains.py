"""Every domain must cover every route, or merging it distorts the label prior.

This is the lesson MASSIVE taught, and it is measurable rather than assumed: MASSIVE's
derived rubric has zero cloud rows, and merging it drove cloud recall from 0.58 to 0.12
for the pooled model and to 0.00 for the transformer. An absent class is learned as a
class that does not exist. So the domains generator is built so that each passage yields
exactly one row per route, and these tests hold it to that.
"""

from __future__ import annotations

import pytest

from minijev.domains import (
    DOMAINS,
    TASKS,
    domain_audit,
    generate_all_domains,
    generate_domain,
    load_passages,
)
from minijev.frames import route_from_frame
from minijev.schema import ROUTE_ORDER, Route


@pytest.fixture(scope="module")
def passages():
    return load_passages()


def test_the_passage_file_parses_into_the_expected_domains(passages):
    assert set(passages) == {"news", "philosophy", "academic"}
    for domain, entries in passages.items():
        assert len(entries) == 16, domain
        assert all(entry.passage_id and entry.title and entry.text for entry in entries)
        assert len({entry.passage_id for entry in entries}) == len(entries)


def test_passages_are_long_enough_to_be_state():
    """A passage that fits in a glance cannot exercise reading."""
    for entries in load_passages().values():
        for entry in entries:
            assert 40 <= len(entry.text) <= 200, (entry.passage_id, len(entry.text))


def test_every_domain_covers_all_four_routes_equally():
    audit = domain_audit()
    for domain in DOMAINS:
        entry = audit["domains"][domain]
        assert entry["balanced"] is True, (domain, entry["route_share"])
        assert set(entry["route_share"].values()) == {0.25}, domain


def test_no_domain_is_small_enough_to_be_swamped():
    """A domain that contributes a rounding error cannot be said to have been covered."""
    audit = domain_audit()
    for domain in DOMAINS:
        assert audit["domains"][domain]["groups"] >= 32, domain


def test_the_four_domains_use_the_same_route_vocabulary():
    """Same labels, same meanings; only the surface text differs."""
    for domain, tasks in TASKS.items():
        assert {route for route, _, _, _ in tasks} == set(ROUTE_ORDER), domain


def test_每个_domain_task_frame_derives_the_declared_route():
    """A frame and its declared route must agree, or the annotation is a lie."""
    for domain, tasks in TASKS.items():
        for route, task_id, _, frame in tasks:
            assert route_from_frame(frame) is route, f"{domain}/{task_id}"


def test_literature_is_the_public_domain_excerpt_domain():
    rows = generate_domain("literature", variants=1)
    assert {row.route for row in rows} == set(ROUTE_ORDER)
    assert all(row.metadata.get("frame") for row in rows)


def test_the_route_never_depends_on_the_passage_text():
    """The same passage appears under all four routes, so the topic cannot be the cue."""
    rows = generate_domain("news", variants=1)
    by_passage: dict[str, set[Route]] = {}
    for row in rows:
        by_passage.setdefault(str(row.metadata["passage_id"]), set()).add(row.route)
    assert all(routes == set(ROUTE_ORDER) for routes in by_passage.values())


def test_each_group_is_one_task_over_one_passage():
    rows = generate_domain("academic", variants=3)
    groups: dict[str, set[Route]] = {}
    for row in rows:
        groups.setdefault(str(row.metadata["semantic_group_id"]), set()).add(row.route)
    assert all(len(routes) == 1 for routes in groups.values()), "a group must not span routes"


def test_every_row_carries_the_rule_that_decided_it():
    for row in generate_domain("philosophy", variants=1):
        assert row.metadata["route_rule"]
        assert row.metadata["frame_id"].startswith("dom:philosophy:")


def test_generating_an_unknown_domain_is_an_error():
    with pytest.raises(ValueError):
        generate_domain("poetry")


def test_all_domains_returns_one_list_per_domain():
    everything = generate_all_domains(variants=1)
    assert set(everything) == set(DOMAINS)
    assert all(rows for rows in everything.values())


def test_a_domain_row_encodes_its_passage_not_just_its_task():
    """If the task text were shared across passages, the model could route on the task alone."""
    rows = generate_domain("academic", variants=1)
    by_task: dict[str, set[str]] = {}
    for row in rows:
        by_task.setdefault(str(row.metadata["task_id"]), set()).add(row.query)
    assert all(len(queries) == 16 for queries in by_task.values()), "each task must appear over every passage"


def test_domain_audit_reports_groups_and_rows():
    """One group is a task over a passage, so rows are groups times the variant count."""
    audit = domain_audit(variants=2)
    for domain in DOMAINS:
        entry = audit["domains"][domain]
        assert entry["rows"] == entry["groups"] * 2
        assert entry["rows"] == sum(entry["route_counts"].values())
