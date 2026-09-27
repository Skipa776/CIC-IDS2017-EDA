import numpy as np

from src.models.evaluate import threshold_at_fpr


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
