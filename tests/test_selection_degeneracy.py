"""Model selection must not be degenerate, and ECE's is.

The sealed-split run made this concrete. Two arms selected by `select_on="calibration"` were
chosen at epoch 4 with accuracy 0.46, while the same runs reached 0.5567 at epoch 10 -- identical
to the plain baseline. The loss had not cost any capacity; the ruler discarded it.

The mechanism is the mirror image of the round-4 failure. There, selecting on macro-F1 selected
*against* calibration. Here, selecting on ECE selects against accuracy, because an under-trained
model is under-confident and under-confidence scores well on ECE. ECE's optimum sits at the
untrained end of the curve.

These tests pin the arithmetic of that claim, since the phenomenon itself needs a training run.
"""

from __future__ import annotations

import pytest

from minijev.evaluation import expected_calibration_error
from minijev.training import SELECTION_CRITERIA, TrainingConfig, selection_score

# Three models at the same accuracy, differing only in how confident they claim to be.
BASE_RATE = [True, True, False, False]


def _report(accuracy: float, ece: float, aurc: float, macro_f1: float = 0.5) -> dict:
    return {"macro_f1": macro_f1, "confidence": {"ece": ece}, "selective": {"aurc": aurc}}


def test_an_under_confident_untrained_model_scores_better_on_ece():
    """The degenerate optimum, in arithmetic.

    A model that guesses nothing gets ECE near zero: predicting the base rate with the base rate's
    confidence is perfectly calibrated and completely useless. Any selection rule that maximises
    this will walk toward the untrained end.
    """
    hedged = [0.5, 0.5, 0.5, 0.5]  # says "even odds" and is right half the time
    trained_looking = [0.9, 0.9, 0.9, 0.9]  # confident, and wrong half the time
    assert expected_calibration_error(hedged, BASE_RATE) == pytest.approx(0.0, abs=1e-9)
    assert expected_calibration_error(trained_looking, BASE_RATE) > 0.3
    assert expected_calibration_error(hedged, BASE_RATE) < expected_calibration_error(trained_looking, BASE_RATE)


def test_selecting_on_ece_prefers_the_useless_model():
    """So `select_on="calibration"` picks the hedged one. That is the bug, not a preference."""
    config = TrainingConfig(select_on="calibration")
    hedged = _report(accuracy=0.5, ece=0.02, aurc=0.48)
    useful = _report(accuracy=0.56, ece=0.22, aurc=0.33)
    assert selection_score(hedged, config) > selection_score(useful, config), "ECE selection should prefer the hedged model"


def test_selecting_on_aurc_prefers_the_useful_model():
    """The gate metric does not have that degeneracy: an uninformative confidence ranks at chance."""
    config = TrainingConfig(select_on="selective")
    hedged = _report(accuracy=0.5, ece=0.02, aurc=0.48)
    useful = _report(accuracy=0.56, ece=0.22, aurc=0.33)
    assert selection_score(useful, config) > selection_score(hedged, config)


def test_every_criterion_trades_something_away():
    """Documents what each ruler buys, so the choice is deliberate.

    macro_f1 ignores calibration entirely; calibration ignores accuracy; selective ignores how
    honest the reported probability is; combined is the only one that prices two things at once.
    """
    honest_weak = _report(accuracy=0.50, ece=0.02, aurc=0.48, macro_f1=0.45)
    strong_overconfident = _report(accuracy=0.56, ece=0.35, aurc=0.33, macro_f1=0.56)
    assert selection_score(honest_weak, TrainingConfig(select_on="macro_f1")) < selection_score(
        strong_overconfident, TrainingConfig(select_on="macro_f1")
    )
    assert selection_score(honest_weak, TrainingConfig(select_on="calibration")) > selection_score(
        strong_overconfident, TrainingConfig(select_on="calibration")
    )
    assert selection_score(honest_weak, TrainingConfig(select_on="selective")) < selection_score(
        strong_overconfident, TrainingConfig(select_on="selective")
    )


def test_the_criteria_list_still_names_every_ruler():
    assert set(SELECTION_CRITERIA) == {"macro_f1", "calibration", "combined", "selective"}
