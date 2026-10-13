"""Model selection must use the goal's own ruler.

The bug this guards against is not a crash, it is a wrong conclusion. Round 3 compared
calibration weights while selecting checkpoints on validation macro-F1 and concluded that "a
weak calibration penalty is worse than none" -- weights 0.1 and 0.3 appeared to make
calibration *worse*. Re-run with calibration-aware selection, the same weight 0.1 is the best
arm on the frontier (ECE -57.7%). The penalty had not changed; the ruler had.

Selecting on macro-F1 is not neutral when calibration is part of the goal. It picks the epoch
where the calibration term has done least, which for a heavy weight is an early epoch where the
model is simply under-trained and therefore accidentally less overconfident.
"""

from __future__ import annotations

import pytest

from minijev.training import SELECTION_CRITERIA, TrainingConfig, selection_score


def _report(macro_f1: float, ece: float) -> dict:
    return {"macro_f1": macro_f1, "confidence": {"ece": ece}}


def test_macro_f1_selection_ignores_calibration_entirely():
    config = TrainingConfig(select_on="macro_f1")
    assert selection_score(_report(0.60, 0.10), config) == pytest.approx(0.60)
    assert selection_score(_report(0.60, 0.90), config) == pytest.approx(0.60)


def test_calibration_selection_ignores_accuracy_entirely():
    """It is a pure calibration ruler, which is why it is not the default."""
    config = TrainingConfig(select_on="calibration")
    assert selection_score(_report(0.90, 0.10), config) == pytest.approx(-0.10)
    assert selection_score(_report(0.10, 0.10), config) == pytest.approx(-0.10)
    assert selection_score(_report(0.10, 0.05), config) > selection_score(_report(0.90, 0.10), config)


def test_combined_selection_trades_one_against_the_other():
    config = TrainingConfig(select_on="combined", calibration_tradeoff=1.0)
    # Equal macro-F1: the better calibrated model wins.
    assert selection_score(_report(0.60, 0.10), config) > selection_score(_report(0.60, 0.30), config)
    # A large enough accuracy gain can still outweigh worse calibration.
    assert selection_score(_report(0.90, 0.30), config) > selection_score(_report(0.60, 0.10), config)
    # A larger tradeoff makes calibration matter more.
    strict = TrainingConfig(select_on="combined", calibration_tradeoff=5.0)
    assert selection_score(_report(0.60, 0.10), strict) > selection_score(_report(0.90, 0.30), strict)


def test_the_tradeoff_is_exactly_a_penalty():
    config = TrainingConfig(select_on="combined", calibration_tradeoff=2.0)
    assert selection_score(_report(0.80, 0.25), config) == pytest.approx(0.80 - 2.0 * 0.25)


def test_an_unknown_criterion_is_refused():
    with pytest.raises(ValueError):
        selection_score(_report(0.5, 0.1), TrainingConfig(select_on="vibes"))


def test_the_criteria_are_the_ones_the_cli_offers():
    assert set(SELECTION_CRITERIA) == {"macro_f1", "calibration", "combined"}


def test_the_default_is_still_macro_f1_for_backward_compatibility():
    """Every earlier experiment was selected this way; changing the default would silently
    invalidate them."""
    assert TrainingConfig().select_on == "macro_f1"


def test_selection_sees_the_same_ece_the_report_shows():
    """Selection reads `validation['confidence']['ece']`, the same field a report prints."""
    import json
    from pathlib import Path

    metrics = sorted(Path(__file__).resolve().parent.parent.glob("models/cal_*_metrics.json"))
    if not metrics:
        pytest.skip("no calibration runs on disk")
    checked = 0
    for path in metrics:
        data = json.loads(path.read_text(encoding="utf-8"))
        validation = data["validation"]
        assert "ece" in validation["confidence"], path.name
        if "select_on" not in data["config"]:
            # Written before the selection rule became configurable; asserting on it would fail
            # for a reason that has nothing to do with this test.
            continue
        checked += 1
        assert data["config"]["select_on"] in SELECTION_CRITERIA, path.name
        assert "best_score" in data["config"], path.name
        assert "best_macro_f1" in data["config"], f"{path.name} hides which ruler chose it"
    assert checked, "no run records its selection rule; the field is not reaching the report"
