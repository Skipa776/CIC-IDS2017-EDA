import numpy as np

from src.models.evaluate import threshold_at_fpr, threshold_on_benign


def test_threshold_at_fpr_respects_target():
    # 100 benign scores spread over [0, 1]; 10 attacks at 0.995.
    # Only one benign score (1.0) is above the attacks, so 1% FPR catches all attacks.
    y = np.array([0] * 100 + [1] * 10)
    scores = np.r_[np.linspace(0, 1, 100), np.full(10, 0.995)]

    threshold, fpr, recall = threshold_at_fpr(y, scores, 0.01)
    assert (threshold, fpr, recall) == (0.995, 0.01, 1.0)

    # A stricter target has to give up the attacks.
    _, fpr, recall = threshold_at_fpr(y, scores, 0.005)
    assert fpr == 0.0 and recall == 0.0


def test_benign_threshold_handles_ties_and_zero_budget():
    scores = np.r_[np.zeros(97), [0.8, 0.8, 0.9]]
    threshold = threshold_on_benign(scores, 0.02)
    assert np.mean(scores >= threshold) == 0.01
    assert threshold_on_benign(scores, 0) > scores.max()


def test_benign_threshold_does_not_promise_test_fpr():
    validation = np.linspace(0, 0.1, 100)
    threshold = threshold_on_benign(validation, 0.01)
    assert np.mean(validation >= threshold) <= 0.01
    assert np.mean(np.array([0.2, 0.3]) >= threshold) == 1.0


def test_benign_threshold_rejects_invalid_inputs():
    import pytest

    for scores, budget in [([], 0.01), ([np.nan], 0.01), ([0.1], -0.1), ([0.1], 1)]:
        with pytest.raises(ValueError):
            threshold_on_benign(scores, budget)
