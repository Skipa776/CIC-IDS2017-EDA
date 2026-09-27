#!/usr/bin/env python3
"""
Combine a supervised detector with Isolation Forest and test on unseen attacks.

Detectors:
    logreg   StandardScaler + LogisticRegression, 71 features (as in cicids2017_eda.ipynb)
    lgbm     Layer 1 LightGBM, 20 FAST_FEATURES (as in train_models.py)
    iforest  Isolation Forest on 71 features, trained on benign flows only
    logreg+iforest, lgbm+iforest
             alert if EITHER model alerts. A fixed alert budget (1% of benign flows)
             is split between the two models.

How the budget split is chosen: leave-one-day-out inside the TRAINING days only.
Each training weekday with attacks is held out in turn, models are fit on the rest,
and we keep the split that catches the most held-out attacks. This imitates meeting
unseen attack types without looking at the test set.

Two kinds of thresholds are reported on the test set:
    deployed      thresholds set on held-out benign flows from the training days.
                  Realistic, but the false positive rate drifts on new traffic.
    equal_budget  thresholds set on the test set's own benign flows, so every
                  detector alerts on exactly the same share of benign traffic.
                  Optimistic, but a fair comparison between detectors.

Experiments:
    2017_crossday   train CIC-IDS2017 Mon+Tue+Fri, test Wed+Thu
    2017_to_2018    train all of CIC-IDS2017, test all of CSE-CIC-IDS2018

Usage:
    python scripts/build_dataset.py --year 2018   # once
    python scripts/combined_model.py

Writes reports/combined_model.json
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.train_models import DATA_PATH_V2, crossday_splits, downsample_benign
from src.data.cicids2018 import BENIGN_SAMPLE_RATE
from src.features.engineering import FAST_FEATURES, create_scaler
from src.models.train import train_layer1_binary

DATA_PATH_2018 = ROOT / "data" / "processed" / "cicids2018_clean.parquet"
OUT_JSON = ROOT / "reports" / "combined_model.json"
BUDGET = 0.01
SHARES = [1.0, 0.75, 0.5, 0.25, 0.0]  # share of the budget given to the supervised model
SUPERVISED = ["logreg", "lgbm"]
SEED = 42


def fit_detectors(df, features, fast_idx, y, pool):
    """Fit the three detectors on a pool of training rows; return a scoring function."""
    tr = downsample_benign(pool, y)
    X_tr = df[features].values[tr]
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced"),
    ).fit(X_tr, y[tr])
    lgbm_scaler = create_scaler(X_tr[:, fast_idx])
    lgbm = train_layer1_binary(lgbm_scaler.transform(X_tr[:, fast_idx]), y[tr])
    benign = X_tr[y[tr] == 0]
    iso_scaler = StandardScaler().fit(benign)
    iforest = IsolationForest(n_estimators=1000, random_state=SEED, n_jobs=-1).fit(iso_scaler.transform(benign))

    def score(X):
        """Higher = more likely attack, for each detector."""
        return {
            "logreg": logreg.predict_proba(X)[:, 1],
            "lgbm": lgbm.predict_proba(lgbm_scaler.transform(X[:, fast_idx]))[:, 1],
            "iforest": -iforest.decision_function(iso_scaler.transform(X)),
        }

    return score, tr


def thresholds(benign_scores, sup, share):
    """Per-model thresholds so the pair uses at most BUDGET of benign flows."""
    def q(s, frac):
        return np.quantile(s, 1 - frac) if frac > 0 else np.inf
    return q(benign_scores[sup], BUDGET * share), q(benign_scores["iforest"], BUDGET * (1 - share))


def alerts(scores, sup, th):
    return (scores[sup] > th[0]) | (scores["iforest"] > th[1])


def single_alerts(scores, name, benign_scores):
    return scores[name] > np.quantile(benign_scores[name], 1 - BUDGET)


def choose_shares(df, features, fast_idx, y, pool, day):
    """Leave-one-day-out over the training weekdays that contain attacks."""
    held_out_days = sorted(np.unique(day[pool][y[pool] == 1]))
    recall = {sup: {s: [] for s in SHARES} for sup in SUPERVISED}
    for d in held_out_days:
        inner_pool, inner_test = pool[day[pool] != d], pool[day[pool] == d]
        score, _ = fit_detectors(df, features, fast_idx, y, inner_pool)
        sc = score(df[features].values[inner_test])
        y_te = y[inner_test]
        benign_sc = {k: v[y_te == 0] for k, v in sc.items()}
        for sup in SUPERVISED:
            for s in SHARES:
                recall[sup][s].append(alerts(sc, sup, thresholds(benign_sc, sup, s))[y_te == 1].mean())
        print(f"    held out {d}: " + ", ".join(
            f"{sup} best share {max(SHARES, key=lambda s: recall[sup][s][-1])}" for sup in SUPERVISED))
    mean = {sup: {str(s): float(np.mean(r)) for s, r in recall[sup].items()} for sup in SUPERVISED}
    best = {sup: max(SHARES, key=lambda s: mean[sup][str(s)]) for sup in SUPERVISED}
    return best, mean, held_out_days


def summarize(alert, y, labels):
    return {
        "fpr": float(alert[y == 0].mean()),
        "recall": float(alert[y == 1].mean()),
        "per_attack_recall": {l: float(alert[labels == l].mean()) for l in np.unique(labels[y == 1])},
    }


def evaluate(name, train_df, test_df, features, train_pool, day, weight_test):
    print(f"\n=== {name}")
    fast_idx = [features.index(f) for f in FAST_FEATURES]
    y_tr = (train_df["Label"] != "BENIGN").astype(int).values
    best, share_search, folds = choose_shares(train_df, features, fast_idx, y_tr, train_pool, day)
    print(f"  chosen budget share for the supervised model: {best}")

    score, used = fit_detectors(train_df, features, fast_idx, y_tr, train_pool)
    # benign training-day flows the detectors never saw, for setting deployed thresholds
    calib = np.setdiff1d(train_pool[y_tr[train_pool] == 0], used)
    calib = np.random.RandomState(SEED).choice(calib, size=min(len(calib), 200_000), replace=False)
    calib_sc = score(train_df[features].values[calib])

    y = (test_df["Label"] != "BENIGN").astype(int).values
    labels = test_df["Label"].values
    sc = score(test_df[features].values)
    test_benign_sc = {k: v[y == 0] for k, v in sc.items()}

    detectors = {}
    for det in ["logreg", "lgbm", "iforest"] + [f"{s}+iforest" for s in SUPERVISED]:
        if "+" in det:
            sup = det.split("+")[0]
            deployed = alerts(sc, sup, thresholds(calib_sc, sup, best[sup]))
            equal = alerts(sc, sup, thresholds(test_benign_sc, sup, best[sup]))
        else:
            deployed = single_alerts(sc, det, calib_sc)
            equal = single_alerts(sc, det, test_benign_sc)
        detectors[det] = {"deployed": summarize(deployed, y, labels), "equal_budget": summarize(equal, y, labels)}
        print(f"  {det:<16} deployed: FPR {detectors[det]['deployed']['fpr']:.3f} "
              f"recall {detectors[det]['deployed']['recall']:.3f} | "
              f"equal budget: recall {detectors[det]['equal_budget']['recall']:.3f}")

    return {
        "n_test_rows": int(len(y)),
        "attack_share": float(np.average(y, weights=weight_test)),
        "leave_one_day_out_folds": folds,
        "share_search_mean_recall": share_search,
        "chosen_share": {k: float(v) for k, v in best.items()},
        "pr_auc": {k: float(average_precision_score(y, v, sample_weight=weight_test)) for k, v in sc.items()},
        "detectors": detectors,
    }


def main():
    df17 = pd.read_parquet(DATA_PATH_V2)
    features = df17.select_dtypes(include=[np.number]).columns.tolist()
    y17 = (df17["Label"] != "BENIGN").astype(int).values
    day17 = df17["Meta_source"].str.split("-").str[0].values  # weekday name
    results = {"budget": BUDGET, "shares_tried": SHARES, "experiments": {}}

    _, test_days, _, te = next((s for s in crossday_splits(df17, y17) if s[0] == "test_wed_thu"))
    pool = np.setdiff1d(np.arange(len(df17)), te)
    results["experiments"]["2017_crossday"] = {
        "test_days": test_days,
        **evaluate("CIC-IDS2017: train Mon+Tue+Fri, test Wed+Thu", df17, df17.iloc[te].reset_index(drop=True),
                   features, pool, day17, np.ones(len(te))),
    }

    if not DATA_PATH_2018.exists():
        sys.exit(f"Missing {DATA_PATH_2018}. Build it with: python scripts/build_dataset.py --year 2018")
    df18 = pd.read_parquet(DATA_PATH_2018)
    assert df18.select_dtypes(include=[np.number]).columns.tolist() == features
    weight18 = np.where(df18["Label"] == "BENIGN", 1 / BENIGN_SAMPLE_RATE, 1.0)
    results["experiments"]["2017_to_2018"] = {
        "benign_sample_rate_2018": BENIGN_SAMPLE_RATE,
        **evaluate("Train all CIC-IDS2017, test CSE-CIC-IDS2018", df17, df18,
                   features, np.arange(len(df17)), day17, weight18),
    }

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
