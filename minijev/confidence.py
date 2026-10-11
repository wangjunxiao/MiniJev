"""Derive a scalar confidence from the route distribution.

This follows the contract TypeSafe uses: every answer carries a full probability
distribution over the options, and `confidence` is a *statistic of that
distribution's shape*. A distribution concentrated on one route is a confident
read; a flat one means the model cannot tell the routes apart.

Two consequences of deriving confidence this way rather than learning it:

* it cannot collapse to a constant while the route head still discriminates, and
* it is measured against whether the model was actually right, which is the
  question a router's safety branch needs answered.

The measure is a choice, not a law -- a score distribution and a choice
distribution want different statistics. Several are provided, `Decision` records
which one produced its number, and `distribution_shape` hands back the raw shape so
a caller can apply a threshold on something we did not anticipate.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

from .schema import ROUTE_ORDER

SMALLEST_PROBABILITY = 1e-12


def _clean(probabilities: Sequence[float]) -> tuple[float, ...]:
    """Renormalise and clamp so log() is safe and the values sum to one."""
    values = [max(float(value), 0.0) for value in probabilities]
    total = sum(values)
    if total <= 0:
        return tuple(1.0 / len(values) for _ in values)
    return tuple(value / total for value in values)


def entropy(probabilities: Sequence[float]) -> float:
    """Shannon entropy in nats. Zero for a one-hot distribution."""
    values = _clean(probabilities)
    return -sum(value * math.log(value) for value in values if value > 0)


def top_probability(probabilities: Sequence[float]) -> float:
    """The largest probability. The simplest measure, and the most overconfident.

    It is not comparable across questions with different numbers of options: 0.7 is
    a confident read among 2 choices and a weak one among 10.
    """
    return max(_clean(probabilities))


def normalized_entropy(probabilities: Sequence[float]) -> float:
    """`1 - H / log(K)`: 1 when one route dominates, 0 when all are equal.

    Scale-free in the number of routes, which is what makes it the sane default.
    """
    values = _clean(probabilities)
    options = len(values)
    if options < 2:
        return 1.0
    return 1.0 - entropy(values) / math.log(options)


def margin(probabilities: Sequence[float]) -> float:
    """Top minus runner-up. Distinguishes a clear winner from a near-tie.

    Useful when the decision is "is there an obvious winner", less useful when many
    options split the remainder.
    """
    values = sorted(_clean(probabilities), reverse=True)
    return values[0] - values[1] if len(values) > 1 else 1.0


def effective_options(probabilities: Sequence[float]) -> float:
    """Perplexity `exp(H)`: how many routes the model is genuinely choosing between.

    1.0 means one clear route, `len(routes)` means it cannot choose at all. This is
    a count rather than a confidence, and is often the easiest number to explain.
    """
    return math.exp(entropy(probabilities))


CONFIDENCE_MEASURES: dict[str, Callable[[Sequence[float]], float]] = {
    "normalized_entropy": normalized_entropy,
    "top_probability": top_probability,
    "margin": margin,
}

DEFAULT_MEASURE = "normalized_entropy"


def confidence_from(probabilities: Sequence[float], measure: str = DEFAULT_MEASURE) -> float:
    """Score a distribution in `[0, 1]`. Higher means a clearer read."""
    try:
        function = CONFIDENCE_MEASURES[measure]
    except KeyError:
        raise KeyError(f"unknown confidence measure {measure!r}; expected one of {sorted(CONFIDENCE_MEASURES)}") from None
    return min(max(function(probabilities), 0.0), 1.0)


def distribution_shape(probabilities: Sequence[float], measure: str = DEFAULT_MEASURE) -> dict[str, float]:
    """Everything about the shape, so a caller is never bound by our default."""
    values = _clean(probabilities)
    ranked = sorted(values, reverse=True)
    return {
        "confidence": confidence_from(values, measure),
        "measure": measure,  # type: ignore[dict-item]
        "top_probability": ranked[0],
        "runner_up": ranked[1] if len(ranked) > 1 else 0.0,
        "margin": margin(values),
        "entropy_nats": entropy(values),
        "normalized_entropy": normalized_entropy(values),
        "effective_options": effective_options(values),
        "uniform_probability": 1.0 / len(values),
    }


def uniform_confidence(measure: str = DEFAULT_MEASURE) -> float:
    """What a maximally uninformative distribution scores under `measure`.

    Every real reading should sit above this; a batch that does not means the route
    head has stopped discriminating.
    """
    return confidence_from([1.0 / len(ROUTE_ORDER)] * len(ROUTE_ORDER), measure)


__all__ = [
    "CONFIDENCE_MEASURES",
    "DEFAULT_MEASURE",
    "SMALLEST_PROBABILITY",
    "confidence_from",
    "distribution_shape",
    "effective_options",
    "entropy",
    "margin",
    "normalized_entropy",
    "top_probability",
    "uniform_confidence",
]
