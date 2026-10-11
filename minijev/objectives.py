"""Training objectives. The only module that defines a learning signal.

Supervised learning is what MiniJev actually uses today. `policy_gradient_loss` is
the reserved RL entry point: it is implemented and unit-tested, but nothing calls
it in the training loop until there is a real reward signal to feed it.
"""

from __future__ import annotations

from tinygrad import Tensor

from .model import JevOutputs


def supervised_loss(
    outputs: JevOutputs,
    route_targets: Tensor,
    *,
    sample_weights: Tensor | None = None,
    label_smoothing: float = 0.0,
) -> Tensor:
    """Cross-entropy on the route head, optionally weighted per sample.

    `sample_weights` is how the dataset's annotator-agreement label earns its keep:
    a row three annotators agreed on should count for more than a row they split
    on. It is a weight, not a prediction target -- nothing in the model tries to
    output it.

    There is no confidence loss here, because there is no confidence head. See
    `minijev.model` for why.
    """
    per_sample = outputs.route_logits.sparse_categorical_crossentropy(route_targets, label_smoothing=label_smoothing, reduction="none")
    if sample_weights is None:
        return per_sample.mean()
    total = sample_weights.sum().maximum(1e-6)
    return (per_sample * sample_weights).sum() / total


def slot_loss(
    outputs: JevOutputs,
    targets: dict[str, Tensor],
    mask: Tensor,
    *,
    weights: dict[str, float] | None = None,
) -> Tensor | None:
    """Mean cross-entropy over the auxiliary deciding-slot heads.

    Only rows that carry a frame contribute: `mask` is 1 for a framed row and 0 for a
    template row, so a mixed corpus does not force the heads to predict a label that was
    never annotated. Returns `None` when the model has no auxiliary heads, or when no row
    in the batch is framed -- in which case the caller adds nothing rather than a zero
    that would still be differentiated.

    `weights` lets a slot be up- or down-weighted. Entropy over slots is not comparable:
    `underspecified` is nearly always false and a head can score well on it by being
    constant, so a caller may want to discount it.
    """
    if not outputs.slot_logits:
        return None
    # Check the raw count, not a clamped one: `maximum(1e-6)` would turn "no framed rows"
    # into "one framed row" and the contract below would silently never fire.
    framed = mask.sum()
    if float(framed.numpy()) <= 0.0:
        return None
    total_rows = framed.maximum(1e-6)

    total: Tensor | None = None
    weight_sum = 0.0
    for slot, logits in outputs.slot_logits.items():
        if slot not in targets:
            continue
        per_sample = logits.sparse_categorical_crossentropy(targets[slot], reduction="none")
        slot_weight = 1.0 if weights is None else float(weights.get(slot, 1.0))
        if slot_weight == 0.0:
            continue
        term = (per_sample * mask).sum() / total_rows * slot_weight
        total = term if total is None else total + term
        weight_sum += slot_weight
    if total is None or weight_sum == 0.0:
        return None
    return total / weight_sum


def policy_gradient_loss(log_probabilities: Tensor, rewards: Tensor, *, baseline: Tensor | None = None) -> Tensor:
    """REINFORCE-style objective for a single-step (contextual bandit) decision.

    Routing is a one-step decision: a query goes to one route and one reward comes
    back, with no state transition. The policy-gradient objective for that case is
    just the negative log-probability weighted by the advantage.

    `log_probabilities` must come from `Decision.log_prob` on sampled routes, not
    from an argmax -- sampling is what makes the gradient non-zero. Subtracting a
    `baseline` (a value estimate or a batch mean) reduces variance; passing `None`
    uses the raw reward, which is unbiased but noisier.
    """
    advantage = rewards if baseline is None else rewards - baseline
    return -(log_probabilities * advantage).mean()
