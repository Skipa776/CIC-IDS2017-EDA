import numpy as np

from scripts.evaluate_generalization import evaluate_candidate, make_partitions, summarize_scores


def test_day_partitions_are_disjoint_and_chronological_where_claimed():
    days = np.repeat(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"], 20)
    y = np.tile([0, 1], 50)
    for name in ["random_split", "test_friday", "test_wed_thu", "forward_thursday"]:
        tr, va, te = make_partitions(days, y, name, 42)
        assert not set(tr) & set(va)
        assert not set(tr) & set(te)
        assert not set(va) & set(te)
        if name != "random_split":
            assert not set(days[tr]) & set(days[va])
            assert not set(days[tr]) & set(days[te])
        if name == "forward_thursday":
            assert set(days[tr]) == {"Monday", "Tuesday"}
            assert set(days[va]) == {"Wednesday"}
            assert set(days[te]) == {"Thursday"}


def test_frozen_threshold_reports_drift_and_family_support():
    scores = np.array([0.6, 0.7, 0.4, 0.9])
    y = np.array([0, 0, 1, 1])
    labels = np.array(["BENIGN", "BENIGN", "Bot", "DDoS"])
    report = summarize_scores(y, labels, scores, 0.5)
    assert report["fpr"] == 1.0
    assert report["recall"] == 0.5
    assert report["alerts_per_10000_flows"] == 7500
    assert report["per_attack"]["Bot"] == {"n": 1, "recall": 0.0}


def test_benign_only_day_has_no_attack_recall():
    report = summarize_scores(np.zeros(2), np.array(["BENIGN", "BENIGN"]),
                              np.array([0.4, 0.9]), 0.5)
    assert report["recall"] is None
    assert report["fpr"] == 0.5


def test_candidate_thresholds_do_not_depend_on_outer_test_labels():
    X = np.random.RandomState(42).normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)
    labels = np.where(y, "Bot", "BENIGN")
    train, val, test = np.arange(80), np.arange(80, 140), np.arange(140, 200)
    days = np.repeat(["Tuesday", "Wednesday", "Thursday"], [80, 60, 60])
    first = evaluate_candidate(X, y, labels, train, val, test, [0, 1], "lgbm20", 42, 1, days)
    changed_y = y.copy()
    changed_y[test] = 1 - changed_y[test]
    changed_labels = np.where(changed_y, "Bot", "BENIGN")
    second = evaluate_candidate(X, changed_y, changed_labels, train, val, test,
                                [0, 1], "lgbm20", 42, 1, days)
    assert first["validation_macro_attack_recall"] == second["validation_macro_attack_recall"]
    for budget in ["0.001", "0.01", "0.05"]:
        assert first["test_at_frozen_thresholds"][budget]["threshold"] == second["test_at_frozen_thresholds"][budget]["threshold"]
    assert first["test_average_precision"] != second["test_average_precision"]
