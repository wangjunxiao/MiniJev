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
