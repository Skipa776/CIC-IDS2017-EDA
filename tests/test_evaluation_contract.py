import numpy as np
import pytest

from src.evaluation.contract import evaluate, frozen_thresholds, wilson_interval
from src.evaluation.folds import FOLDS_BY_NAME, VAL_HOLDOUT, day_key, partition


def test_wilson_interval_brackets_the_rate():
    low, high = wilson_interval(9, 10)
    assert low < 0.9 < high <= 1.0
    assert wilson_interval(0, 0) != wilson_interval(0, 0)  # nan pair when n == 0


def test_frozen_threshold_reports_actual_test_fpr():
    # Validation benign tops out at 1.0; test benign drifts upward, so the
    # "1%" threshold lets far more than 1% of test benign through.
    val_benign = np.linspace(0, 1, 1000)
    thresholds = frozen_thresholds(val_benign)
    y = np.array([0] * 100 + [1] * 10)
    families = np.array(["Benign"] * 100 + ["DoS"] * 10)
    scores = np.r_[np.full(100, 2.0), np.full(10, 5.0)]
    out = evaluate(y, families, scores, thresholds)
    primary = out["budgets"]["0.01"]
    assert primary["fpr"] == 1.0 and primary["recall"] == 1.0
    assert out["prevalence"] == pytest.approx(10 / 110)
    assert primary["per_family"]["DoS"]["n"] == 10
    assert out["roc_auc"] == 1.0  # every attack outscores every benign flow


def test_weighted_prevalence_undoes_benign_sampling():
    y = np.array([0, 0, 1, 1])
    out = evaluate(y, np.array(["Benign", "Benign", "Bot", "Bot"]), np.array([0.1, 0.2, 0.8, 0.9]),
                   frozen_thresholds(np.array([0.1, 0.2])), weight=np.array([10, 10, 1, 1]))
    assert out["prevalence"] == pytest.approx(2 / 22)


def test_day_keys():
    assert day_key("Thursday-WorkingHours-Morning-WebAttacks", "2017") == "Thursday"
    assert day_key("Thuesday-20-02-2018", "2018") == "2018-02-20"


def test_day_fold_never_trains_on_later_days():
    days = np.array(["Monday"] * 4 + ["Tuesday"] * 4 + ["Wednesday"] * 4 + ["Thursday"] * 4)
    y = np.tile([0, 0, 0, 1], 4)
    tr, va, te = partition(FOLDS_BY_NAME["2017_B"], days, y, seed=0)
    assert set(days[tr]) == {"Monday", "Tuesday"}
    assert set(days[va]) == {"Wednesday"} and set(days[te]) == {"Thursday"}


def test_holdout_fold_trains_on_the_repeated_attack_day():
    fold = FOLDS_BY_NAME["2018_same_attack_web"]
    assert fold.val_days == (VAL_HOLDOUT,) and "2018-02-22" in fold.train_days
    days = np.repeat(np.array(["2018-02-21", "2018-02-22", "2018-02-23"]), 10)
    y = np.tile([0] * 8 + [1] * 2, 3)
    tr, va, te = partition(fold, days, y, seed=0)
    assert not np.intersect1d(tr, va).size
    assert set(days[te]) == {"2018-02-23"} and "2018-02-23" not in set(days[np.r_[tr, va]])
