"""Local-text generation: chunking, grouping and provenance must be honest.

The point of the text-file path is that a licensed novel can be used locally
without committing it. These tests use a throwaway paragraph, not a real work.
"""

from __future__ import annotations

import pytest

from minijev.literary import TEMPLATES, generate_from_text
from minijev.schema import Route

SAMPLE = "第一段文字，用来验证分块。\n\n第二段文字，比第一段稍长一些，用来确认按段落切块是否工作。\n第三行没有空行分隔。"


def test_every_route_is_generated_for_every_block():
    rows = generate_from_text(SAMPLE, variants=1, chunk_size=40)
    assert {row.route for row in rows} == set(Route)
    blocks = {row.metadata["text_block_index"] for row in rows}
    assert len(blocks) > 1, "chunk_size 40 must split this sample"
    assert all(row.metadata["text_block_count"] == len(blocks) for row in rows)
    assert len(rows) == len(blocks) * len(TEMPLATES)


def test_block_indices_are_contiguous_and_ordered():
    rows = generate_from_text(SAMPLE, variants=2, chunk_size=30)
    per_block: dict[int, list[str]] = {}
    for row in rows:
        per_block.setdefault(int(row.metadata["text_block_index"]), []).append(row.route.value)
    assert sorted(per_block) == list(range(len(per_block)))
    # Every block must carry the full task set, or a block would be half-labelled.
    for routes in per_block.values():
        assert sorted(routes) == sorted(route.value for route in Route for _ in range(2))


def test_variants_stay_inside_one_group():
    rows = generate_from_text(SAMPLE, variants=3, chunk_size=40)
    groups: dict[str, set[str]] = {}
    for row in rows:
        groups.setdefault(str(row.metadata["semantic_group_id"]), set()).add(row.query)
    assert all(len(queries) == 3 for queries in groups.values())
    assert len(groups) == len({row.metadata["text_block_index"] for row in rows}) * len(TEMPLATES)


def test_the_generated_rows_never_claim_to_be_public_domain():
    rows = generate_from_text(SAMPLE, title="局外人", author="加缪", variants=1, chunk_size=40)
    assert {row.source for row in rows} == {"literary-local-text-v1"}
    assert {row.metadata["literary_input"] for row in rows} == {"local_text"}
    provenance = {str(row.metadata["text_provenance"]) for row in rows}
    assert all("confirm license" in note for note in provenance)
    assert {row.metadata["literary_work"] for row in rows} == {"局外人"}
    assert {row.metadata["literary_author"] for row in rows} == {"加缪"}


def test_labels_are_still_driven_by_the_task_not_the_block():
    rows = generate_from_text(SAMPLE, variants=1, chunk_size=40)
    by_task = {row.metadata["literary_task"]: row.route for row in rows}
    assert by_task["local_extract"] is Route.LOCAL
    assert by_task["rag_provenance"] is Route.RAG
    assert by_task["cloud_interpret"] is Route.CLOUD
    assert by_task["review_rights"] is Route.REVIEW


def test_empty_text_and_bad_chunk_size_are_refused():
    with pytest.raises(ValueError):
        generate_from_text("\n\n\n")
    with pytest.raises(ValueError):
        generate_from_text(SAMPLE, chunk_size=0)
