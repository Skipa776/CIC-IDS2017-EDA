"""Train on one year of CIC-IDS data, test on the other.

Used by notebooks/train_2017_test_2018.ipynb and notebooks/train_2018_test_2017.ipynb
so both directions run exactly the same experiment.

Variants (the question each one asks):
    baseline      all 71 features, one scaler fit on the training year
    drop_header   drop the header-length features the flow tool measured differently
                  in the two years
    drop_network  drop destination port and initial TCP window sizes, which describe the
                  services and operating systems of one particular network
    per_domain    all features, but each year is quantile-normalized on its OWN
                  unlabeled traffic, so "unusual for this network" means the same thing
    header_and_domain  drop_header + per_domain
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from src.data.cicids2018 import BENIGN_SAMPLE_RATE
from src.models.evaluate import threshold_at_fpr
from src.models.train import train_layer1_binary

ROOT = Path(__file__).parent.parent.parent
PATHS = {
    "2017": ROOT / "data" / "processed" / "cicids2017_clean_v2.parquet",
    "2018": ROOT / "data" / "processed" / "cicids2018_clean.parquet",
}
# 2018 benign was sampled while loading (src/data/cicids2018.py); weight it back up
BENIGN_WEIGHT = {"2017": 1.0, "2018": 1 / BENIGN_SAMPLE_RATE}

FAMILY = {
    "BENIGN": "Benign",
    # CIC-IDS2017
    "DoS Hulk": "DoS", "DoS GoldenEye": "DoS", "DoS slowloris": "DoS", "DoS Slowhttptest": "DoS",
    "DDoS": "DDoS", "FTP-Patator": "Brute force (FTP/SSH)", "SSH-Patator": "Brute force (FTP/SSH)",
    "Web Attack - Brute Force": "Web attack", "Web Attack - XSS": "Web attack",
    "Web Attack - Sql Injection": "Web attack", "Bot": "Bot", "Infiltration": "Infiltration",
    "PortScan": "PortScan", "Heartbleed": "Heartbleed",
    # CSE-CIC-IDS2018
    "DoS attacks-Hulk": "DoS", "DoS attacks-GoldenEye": "DoS", "DoS attacks-Slowloris": "DoS",
    "DoS attacks-SlowHTTPTest": "DoS", "DDoS attacks-LOIC-HTTP": "DDoS", "DDOS attack-HOIC": "DDoS",
    "DDOS attack-LOIC-UDP": "DDoS", "FTP-BruteForce": "Brute force (FTP/SSH)",
    "SSH-Bruteforce": "Brute force (FTP/SSH)", "Brute Force -Web": "Web attack",
    "Brute Force -XSS": "Web attack", "SQL Injection": "Web attack", "Infilteration": "Infiltration",
}

# Measured differently by the flow tool in the two years
# (notebooks/cross_year_diagnostics.ipynb, single-packet DNS check)
HEADER_FEATURES = ["min_seg_size_forward", "Fwd Header Length", "Bwd Header Length"]
# Set by the services and operating systems of one particular network
NETWORK_FEATURES = ["Destination Port", "Init_Win_bytes_forward", "Init_Win_bytes_backward"]

# variant -> (features to drop, normalize each year on its own traffic?)
VARIANTS = {
    "baseline": ([], False),
    "drop_header": (HEADER_FEATURES, False),
    "drop_network": (NETWORK_FEATURES, False),
    "per_domain": ([], True),
    "header_and_domain": (HEADER_FEATURES, True),
}
MAX_BENIGN_TRAIN = 200_000
BUDGET = 0.01
SEED = 42


def load_year(year):
    df = pd.read_parquet(PATHS[year])
    unmapped = set(df["Label"]) - set(FAMILY)
    if unmapped:
        raise ValueError(f"Labels without a family: {sorted(unmapped)}")
    y = (df["Label"] != "BENIGN").astype(int).values
    weight = np.where(y == 0, BENIGN_WEIGHT[year], 1.0)
    return df, y, weight


def natural_mix(y, year, seed=SEED):
    """Row indices that look like the year's real traffic mix (undoes 2018 benign sampling).

    Used only to fit the per-domain normalizer, which never sees labels itself.
    """
    idx = np.arange(len(y))
    if BENIGN_WEIGHT[year] == 1.0:
        return idx
    keep_attack = np.random.RandomState(seed).random_sample(len(y)) < BENIGN_SAMPLE_RATE
    return idx[(y == 0) | keep_attack]


def downsample_benign(idx, y, seed=SEED):
    benign, attack = idx[y[idx] == 0], idx[y[idx] == 1]
    if len(benign) > MAX_BENIGN_TRAIN:
        benign = np.random.RandomState(seed).choice(benign, MAX_BENIGN_TRAIN, replace=False)
    return np.concatenate([benign, attack])


def fit_normalizer(X):
    return QuantileTransformer(n_quantiles=1000, subsample=200_000, random_state=SEED).fit(X)


def fit_models(X, y):
    logreg = make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, n_jobs=-1, class_weight="balanced"),
    ).fit(X, y)
    lgbm_scaler = StandardScaler().fit(X)
    lgbm = train_layer1_binary(lgbm_scaler.transform(X), y)
    return {
        "Logistic regression": lambda Z: logreg.predict_proba(Z)[:, 1],
        "LightGBM": lambda Z: lgbm.predict_proba(lgbm_scaler.transform(Z))[:, 1],
    }


def score_report(scores, y, weight, labels):
    """PR-AUC (weighted to natural prevalence), recall at 1% FPR, per-family recall."""
    threshold, _, recall = threshold_at_fpr(y, scores, BUDGET)
    flagged = scores >= threshold
    families = pd.Series(labels).map(FAMILY).values
    return {
        "PR-AUC": average_precision_score(y, scores, sample_weight=weight),
        "no-skill": float(np.average(y, weights=weight)),
        f"recall at {BUDGET:.0%} FPR": recall,
        "per_family": {f: float(flagged[families == f].mean()) for f in np.unique(families) if f != "Benign"},
    }


def run_direction(source, target):
    """Train on `source` year, test on held-out source rows and on all of `target`."""
    src, y_s, w_s = load_year(source)
    tgt, y_t, w_t = load_year(target)
    features = src.select_dtypes(include=[np.number]).columns.tolist()
    assert features == tgt.select_dtypes(include=[np.number]).columns.tolist()

    idx_train, idx_holdout = train_test_split(
        np.arange(len(src)), test_size=0.2, stratify=y_s, random_state=SEED)
    idx_train = downsample_benign(idx_train, y_s)
    src_mix = np.intersect1d(natural_mix(y_s, source), np.setdiff1d(np.arange(len(src)), idx_holdout))

    rows, per_family = [], {}
    for variant, (dropped, per_domain) in VARIANTS.items():
        cols = [c for c in features if c not in dropped]
        X_src, X_tgt = src[cols].values, tgt[cols].values
        if per_domain:
            X_src = fit_normalizer(X_src[src_mix]).transform(X_src)
            X_tgt = fit_normalizer(X_tgt[natural_mix(y_t, target)]).transform(X_tgt)
        models = fit_models(X_src[idx_train], y_s[idx_train])
        for model, score in models.items():
            for test_name, X, y, w, labels in [
                (f"{source} held-out", X_src[idx_holdout], y_s[idx_holdout], w_s[idx_holdout],
                 src["Label"].values[idx_holdout]),
                (f"{target} (other year)", X_tgt, y_t, w_t, tgt["Label"].values),
            ]:
                report = score_report(score(X), y, w, labels)
                per_family[(variant, model, test_name)] = report.pop("per_family")
                rows.append({"variant": variant, "model": model, "test set": test_name,
                             "n features": len(cols), **report})
    summary = pd.DataFrame(rows).set_index(["variant", "model", "test set"])
    return summary, pd.DataFrame(per_family)
