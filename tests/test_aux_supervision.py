"""Auxiliary slot supervision: the extra heads must help the route and change nothing else.

Three properties matter, and each is a way the idea could fail silently:

1. **The labels must be the real ones.** If a slot index disagreed with the frame, the
   heads would be trained on noise and the experiment would measure nothing.
2. **The decision path must not read the heads.** They are a training-time shaping term;
   if `route_logits` depended on them, an aux checkpoint and a plain one would answer
   differently and the comparison would be between two different systems.
3. **They must not be a capacity trick.** The whole point of the parameter-matched
   control in the encoder comparison was that a bigger model is not a better idea, so the
   aux heads' parameter cost is asserted to be negligible.
"""

from __future__ import annotations

import pytest
from tinygrad import Context, Tensor, dtypes
from tinygrad.nn import optim
from tinygrad.nn.state import get_parameters, get_state_dict

from minijev.config import JevConfig
from minijev.frames import (
    AUX_SLOTS,
    generate_benchmark,
    generate_hard_cases,
    load_frame,
    slot_index,
    slot_targets,
    slot_value,
    slot_vocabulary,
)
from minijev.model import TinyJev, load_checkpoint_config, save_checkpoint
from minijev.objectives import slot_loss, supervised_loss
from minijev.schema import ROUTE_ORDER, Route
from minijev.synthetic import SyntheticOptions, generate_dataset
from minijev.tokenizer import ByteTokenizer


@pytest.fixture(scope="module")
def framed_rows():
    """Every framed row available, training and benchmark alike.

    The named frames the slot assertions use (f01, f02, f24) are benchmark frames, so
    restricting this fixture to the training split would fail to find them.
    """
    return generate_hard_cases(variants=1) + generate_benchmark(variants=1)


def test_the_vocabulary_covers_exactly_the_auxiliary_slots():
    assert tuple(slot_vocabulary()) == AUX_SLOTS
    for slot, values in slot_vocabulary().items():
        assert len(values) >= 2, slot
        assert len(set(values)) == len(values), slot


def test_slot_values_match_the_routing_rule(framed_rows):
    by_id = {str(row.metadata["frame_id"]): row for row in framed_rows}
    # f01 is a document lookup: external source, extractive, one step, low risk.
    manual = load_frame(by_id["f01"])
    assert slot_value(manual, "source_class") == "external"
    assert slot_value(manual, "derivation") == "extract"
    assert slot_value(manual, "steps") == "1"
    assert slot_value(manual, "risk") == "low"
    assert slot_value(manual, "underspecified") == "false"

    # f02 is the same 谓宾 but must be built, so it is internal and constructive.
    system = load_frame(by_id["f02"])
    assert slot_value(system, "source_class") == "internal"
    assert slot_value(system, "derivation") == "construct"

    # f24 is the medication question: the risk slot is the only one that decides it.
    medication = load_frame(by_id["f24"])
    assert slot_value(medication, "risk") == "high"


def test_slot_index_agrees_with_slot_value(framed_rows):
    vocabulary = slot_vocabulary()
    for row in framed_rows:
        frame = load_frame(row)
        for slot in AUX_SLOTS:
            assert vocabulary[slot][slot_index(frame, slot)] == slot_value(frame, slot)


def test_multi_step_frames_land_in_the_three_plus_bucket(framed_rows):
    steps = {slot_value(load_frame(row), "steps") for row in framed_rows}
    assert steps <= {"1", "2", "3+"}
    assert "3+" in steps, "the bucket boundary needs a frame that actually reaches it"


def test_every_auxiliary_label_is_reachable(framed_rows):
    """A head whose every label is constant cannot learn anything and should not exist."""
    seen = {slot: {slot_value(load_frame(row), slot) for row in framed_rows} for slot in AUX_SLOTS}
    for slot, values in seen.items():
        assert len(values) >= 2, (slot, values)


def test_template_rows_are_masked_out_of_the_auxiliary_signal():
    templates = generate_dataset(SyntheticOptions(seed=1, samples_per_route=3, include_hard_cases=False))
    assert all(load_frame(row) is None for row in templates)
    columns, mask = slot_targets(templates)
    assert set(mask) == {0.0}
    assert all(len(values) == len(templates) for values in columns.values())


def test_framed_rows_are_masked_in(framed_rows):
    columns, mask = slot_targets(framed_rows)
    assert set(mask) == {1.0}
    assert all(len(values) == len(framed_rows) for values in columns.values())


def test_slot_targets_keep_rows_aligned_across_a_mixed_batch(framed_rows):
    mixed = list(framed_rows[:2]) + list(generate_dataset(SyntheticOptions(seed=1, samples_per_route=1, include_hard_cases=False))[:2])
    columns, mask = slot_targets(mixed)
    assert mask[:2] == [1.0, 1.0]
    assert mask[2:] == [0.0] * (len(mixed) - 2)
    assert len(columns["risk"]) == len(mixed)


def test_a_plain_model_has_no_slot_logits():
    model = TinyJev(JevConfig())
    tokenizer = ByteTokenizer(model.config)
    ids, mask = tokenizer.batch(["你好"])
    outputs = model(ids, mask)
    assert outputs.slot_logits is None
    assert model.slot_heads == {}


def test_an_aux_model_emits_one_logit_vector_per_slot(framed_rows):
    model = TinyJev(JevConfig(aux_slots=True))
    tokenizer = ByteTokenizer(model.config)
    ids, mask = tokenizer.batch([row.query for row in framed_rows[:8]])
    outputs = model(ids, mask)
    assert set(outputs.slot_logits) == set(AUX_SLOTS)
    for slot, logits in outputs.slot_logits.items():
        assert logits.shape == (8, len(slot_vocabulary()[slot]))
    assert outputs.route_logits.shape == (8, len(ROUTE_ORDER))


def test_the_auxiliary_heads_cost_almost_nothing():
    """Guards the confound: aux must not be a capacity win in disguise."""
    plain = sum(t.numel() for t in get_state_dict(TinyJev(JevConfig())).values())
    aux = sum(t.numel() for t in get_state_dict(TinyJev(JevConfig(aux_slots=True))).values())
    added = aux - plain
    # One Linear(hidden_dim, classes) per slot: classes * (hidden_dim + 1) parameters.
    hidden_dim = JevConfig().hidden_dim
    expected = sum(len(values) * (hidden_dim + 1) for values in slot_vocabulary().values())
    assert added == expected
    assert added / plain < 0.05, f"aux adds {added} of {plain} parameters, which is not negligible"


def test_slot_loss_is_none_without_heads(framed_rows):
    model = TinyJev(JevConfig())
    tokenizer = ByteTokenizer(model.config)
    ids, mask = tokenizer.batch([row.query for row in framed_rows[:4]])
    outputs = model(ids, mask)
    columns, row_mask = slot_targets(framed_rows[:4])
    targets = {slot: Tensor(values, dtype=dtypes.int32) for slot, values in columns.items()}
    assert slot_loss(outputs, targets, Tensor(row_mask, dtype=dtypes.float32)) is None


def test_slot_loss_is_none_when_no_row_in_the_batch_is_framed():
    """Returning a zero tensor would still build a graph and silently add nothing."""
    model = TinyJev(JevConfig(aux_slots=True))
    tokenizer = ByteTokenizer(model.config)
    templates = generate_dataset(SyntheticOptions(seed=1, samples_per_route=3, include_hard_cases=False))[:6]
    ids, mask = tokenizer.batch([row.query for row in templates])
    outputs = model(ids, mask)
    columns, row_mask = slot_targets(templates)
    targets = {slot: Tensor(values, dtype=dtypes.int32) for slot, values in columns.items()}
    assert slot_loss(outputs, targets, Tensor(row_mask, dtype=dtypes.float32)) is None


def test_slot_loss_is_finite_and_learnable(framed_rows):
    model = TinyJev(JevConfig(aux_slots=True))
    tokenizer = ByteTokenizer(model.config)
    ids, mask = tokenizer.batch([row.query for row in framed_rows])
    columns, row_mask = slot_targets(framed_rows)
    targets = {slot: Tensor(values, dtype=dtypes.int32) for slot, values in columns.items()}
    mask_tensor = Tensor(row_mask, dtype=dtypes.float32)
    routes = Tensor([ROUTE_ORDER.index(row.route) for row in framed_rows], dtype=dtypes.int32)

    with Context(TRAINING=1):
        optimizer = optim.Adam(get_parameters(model), lr=0.01)
        first = None
        for _ in range(8):
            outputs = model(ids, mask)
            route_term = supervised_loss(outputs, routes)
            auxiliary = slot_loss(outputs, targets, mask_tensor)
            assert auxiliary is not None
            loss = route_term + auxiliary
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if first is None:
                first = auxiliary.item()
        final = slot_loss(model(ids, mask), targets, mask_tensor).item()
    assert final < first, f"auxiliary loss did not fall: {first} -> {final}"


def test_a_slot_weight_of_zero_is_the_same_as_disabling_that_slot(framed_rows):
    model = TinyJev(JevConfig(aux_slots=True))
    tokenizer = ByteTokenizer(model.config)
    ids, mask = tokenizer.batch([row.query for row in framed_rows[:8]])
    outputs = model(ids, mask)
    columns, row_mask = slot_targets(framed_rows[:8])
    targets = {slot: Tensor(values, dtype=dtypes.int32) for slot, values in columns.items()}
    mask_tensor = Tensor(row_mask, dtype=dtypes.float32)
    everything = slot_loss(outputs, targets, mask_tensor)
    without_risk = slot_loss(outputs, targets, mask_tensor, weights={"risk": 0.0})
    assert everything is not None and without_risk is not None
    assert everything.item() != without_risk.item()


def test_the_checkpoint_records_whether_aux_heads_exist(tmp_path):
    path = tmp_path / "aux.safetensors"
    save_checkpoint(TinyJev(JevConfig(aux_slots=True)), path)
    config = load_checkpoint_config(path)
    assert config.aux_slots is True
    assert set(TinyJev(config).slot_heads) == set(AUX_SLOTS)

    plain_path = tmp_path / "plain.safetensors"
    save_checkpoint(TinyJev(JevConfig()), plain_path)
    assert load_checkpoint_config(plain_path).aux_slots is False


def test_one_route_index_per_route_stays_in_range():
    """The route head is untouched by aux, but the shared helper must stay consistent."""
    assert [ROUTE_ORDER.index(row) for row in Route] == [0, 1, 2, 3]
