"""Context must change the decision, and the probe must be able to prove it can.

The probe's whole value is that its two halves ask the *same question*, so an encoder
that ignores state has no way to answer them differently. If a pair's halves differed in
their query text as well, a collapse could be blamed on the wording.
"""

from __future__ import annotations

import pytest

from minijev.config import JevConfig, context_config
from minijev.contexts import (
    CONTEXT_PAIRS,
    CONTEXT_PROBE_PAIRS,
    CONTEXT_TRAIN_PAIRS,
    ContextCase,
    ContextPair,
    context_audit,
    context_pairs_in,
    generate_context_probe,
    generate_context_training,
)
from minijev.frames import route_from_frame
from minijev.policies import ArgmaxPolicy
from minijev.routers import RuleRouter
from minijev.schema import ROUTE_ORDER, Route
from minijev.tokenizer import TURN_SEPARATOR_ID, ByteTokenizer

# --- tokenizer: turns ---------------------------------------------------------


def test_turns_are_joined_by_the_separator_and_the_request_comes_last():
    tokenizer = ByteTokenizer(context_config())
    encoded = tokenizer.encode_turns(("我已经把手册贴给你了。", "查一下安装步骤"))
    ids = encoded.input_ids[: sum(encoded.attention_mask)]
    assert TURN_SEPARATOR_ID in ids
    assert ids.count(TURN_SEPARATOR_ID) == 1
    last_turn = "查一下安装步骤".encode("utf-8")
    assert ids[-len(last_turn) :] == [byte + 1 for byte in last_turn]


def test_more_turns_means_more_separators():
    tokenizer = ByteTokenizer(context_config())
    two = tokenizer.encode_turns(("甲", "乙"))
    three = tokenizer.encode_turns(("甲", "乙", "丙"))
    assert sum(three.attention_mask) > sum(two.attention_mask)
    assert three.input_ids[: sum(three.attention_mask)].count(TURN_SEPARATOR_ID) == 2


def test_an_empty_turn_does_not_emit_a_stray_separator():
    tokenizer = ByteTokenizer(context_config())
    encoded = tokenizer.encode_turns(("", "只有这句话"))
    ids = encoded.input_ids[: sum(encoded.attention_mask)]
    assert TURN_SEPARATOR_ID not in ids


def test_preserve_tail_keeps_the_request_when_the_state_is_long():
    """The request is the decisive part, so truncation must not eat it."""
    tokenizer = ByteTokenizer(context_config(max_seq_len=64))
    request = "请判断这样做是否合规"
    encoded = tokenizer.encode_turns(("这是一段很长的前置状态。" * 20, request))
    ids = encoded.input_ids[: sum(encoded.attention_mask)]
    expected = [byte + 1 for byte in request.encode("utf-8")]
    assert ids[-len(expected) :] == expected


def test_a_single_turn_encodes_exactly_as_before():
    """Context support must not change the bytes of a context-free corpus."""
    plain = ByteTokenizer(JevConfig())
    contextual = ByteTokenizer(context_config())
    text = "统计这段文字里的汉字数量"
    assert plain.encode(text).input_ids == contextual.encode_turns((text,)).input_ids


def test_a_config_without_a_separator_falls_back_rather_than_crashing():
    """A single-turn checkpoint stays loadable, so the probe can report its collapse."""
    tokenizer = ByteTokenizer(JevConfig())
    encoded = tokenizer.encode_turns(("状态", "请求"))
    ids = encoded.input_ids[: sum(encoded.attention_mask)]
    assert [byte + 1 for byte in b"\n"][0] in ids
    assert sum(encoded.attention_mask) > 0


def test_the_separator_must_fit_in_the_embedding_table():
    with pytest.raises(ValueError):
        JevConfig(vocab_size=257, turn_sep_id=TURN_SEPARATOR_ID)
    assert JevConfig().uses_context is False
    assert context_config().uses_context is True


def test_batch_turns_aligns_rows_with_samples():
    tokenizer = ByteTokenizer(context_config())
    ids, mask = tokenizer.batch_turns([("甲", "请求一"), ("乙丙丁", "请求二")])
    assert ids.shape == mask.shape
    assert ids.shape[0] == 2


# --- the probe ----------------------------------------------------------------


def test_every_pair_asks_the_same_question_in_both_halves():
    for pair in CONTEXT_PAIRS:
        first = {query for _, query in pair.first.realizations}
        second = {query for _, query in pair.second.realizations}
        assert first == second, pair.pair_id


def test_every_pair_routes_its_two_halves_differently():
    for pair in CONTEXT_PAIRS:
        assert pair.first.route is not pair.second.route, pair.pair_id


def test_a_pair_whose_halves_agree_on_the_route_is_rejected():
    """The constructor refuses a pair that cannot measure anything."""
    case = ContextCase("a", CONTEXT_PAIRS[0].first.frame, ((("状态",), "同一句话"),))
    with pytest.raises(ValueError):
        ContextPair("bad", case, case)


def test_a_pair_whose_halves_ask_different_questions_is_rejected():
    first = ContextCase("a", CONTEXT_PAIRS[0].first.frame, ((("状态",), "问题一"),))
    second = ContextCase("b", CONTEXT_PAIRS[0].second.frame, ((("状态",), "问题二"),))
    with pytest.raises(ValueError):
        ContextPair("bad", first, second)


def test_the_probe_is_route_balanced():
    audit = context_audit()
    assert audit["probe_balanced"] is True
    assert set(audit["probe_route_counts"].values()) == {len(generate_context_probe()) // len(ROUTE_ORDER)}


def test_the_probe_covers_every_route_combination():
    """Every pair of routes must be represented, or a slot could go untested."""
    audit = context_audit()
    combinations = {tuple(sorted(key.split("|"))) for key in audit["combinations"]}
    expected = {tuple(sorted((a.value, b.value))) for index, a in enumerate(ROUTE_ORDER) for b in ROUTE_ORDER[index + 1 :]}
    assert combinations == expected


def test_probe_and_training_pairs_are_disjoint():
    assert not (CONTEXT_PROBE_PAIRS & CONTEXT_TRAIN_PAIRS)
    assert CONTEXT_PROBE_PAIRS | CONTEXT_TRAIN_PAIRS == {pair.pair_id for pair in CONTEXT_PAIRS}


def test_probe_and_training_share_no_query_text():
    """A probe row whose words were trained on measures recall, not context use."""
    probe = {(row.query, row.context) for row in generate_context_probe()}
    training = {(row.query, row.context) for row in generate_context_training()}
    assert probe.isdisjoint(training)


def test_probe_rows_carry_their_context_and_pair_id():
    """At least one half of every pair must carry state, or the pair tests nothing."""
    probe = generate_context_probe()
    for row in probe:
        assert row.metadata["context_pair"] in CONTEXT_PROBE_PAIRS
        assert row.metadata["frame"]["underspecified"] in (True, False)
    grouped = context_pairs_in(probe)
    for pair_id, cases in grouped.items():
        assert any(case.context for case in cases.values()), f"{pair_id} has no state on either half"


def test_probe_rows_can_be_grouped_back_into_pairs():
    grouped = context_pairs_in(generate_context_probe())
    assert len(grouped) == len(CONTEXT_PROBE_PAIRS)
    assert all(len(cases) == 2 for cases in grouped.values())


def test_the_declared_route_matches_the_frame_in_every_probe_row():
    for row in generate_context_probe():
        frame = CONTEXT_PAIRS[0].first.frame.__class__.from_dict(row.metadata["frame"])
        assert route_from_frame(frame) is row.route


def test_both_halves_of_a_probe_pair_are_distinguishable_by_policy_alone():
    """Sanity: the label contrast is real, not an artefact of how the probe is written."""
    probe = generate_context_probe()
    grouped = context_pairs_in(probe)
    for cases in grouped.values():
        routes = {case.route for case in cases.values()}
        assert len(routes) == 2


# --- routers must actually accept context -------------------------------------


def test_a_router_accepts_context_without_changing_the_single_turn_path():
    router = RuleRouter()
    assert router.route("查一下手册").route is Route.RAG
    assert router.decide(["查一下手册"]) == router.decide(["查一下手册"], [()])


def test_decide_samples_uses_each_samples_context():
    router = RuleRouter()
    samples = generate_context_probe()[:2]
    decisions = router.decide_samples(samples)
    assert len(decisions) == len(samples)


def test_contexts_must_align_with_queries():
    router = RuleRouter()
    with pytest.raises(ValueError):
        router.decide(["一", "二"], [("只有一个",)])


def test_the_argmax_policy_is_reachable_through_a_context_call():
    router = RuleRouter(policy=ArgmaxPolicy())
    assert isinstance(router.route("查一下手册", ("前面的话",)).confidence, float)
