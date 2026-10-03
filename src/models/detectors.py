"""The detectors compared in the forward-in-time evaluation.

Each `fit_*` returns a function X -> score (higher = more likely attack).
Supervised models train on train-days rows with benign capped at 200k;
anomaly models train on train-days benign flows only, capped the same way.
"""

import warnings

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.models.train import train_layer1_binary

SUPERVISED = ["lgbm", "logreg"]
ANOMALY = ["iforest", "autoencoder"]
MAX_BENIGN = 200_000


def cap_benign(idx, y, seed):
    benign, attack = idx[y[idx] == 0], idx[y[idx] == 1]
    if len(benign) > MAX_BENIGN:
        benign = np.random.RandomState(seed).choice(benign, MAX_BENIGN, replace=False)
    return np.sort(np.concatenate([benign, attack]))


def _batched(fn, X, size=200_000):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        return np.concatenate([fn(X[i:i + size]) for i in range(0, len(X), size)])


def fit_lgbm(X, y, seed, threads):
    scaler = StandardScaler().fit(X)
    model = train_layer1_binary(scaler.transform(X), y, random_state=seed, n_jobs=threads)
    return lambda Z: _batched(lambda B: model.predict_proba(scaler.transform(B))[:, 1], Z)


def fit_logreg(X, y, seed, threads):
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
    model.fit(scaler.transform(X), y)
    if model.n_iter_.max() >= model.max_iter:
        raise RuntimeError("logistic regression did not converge")
    return lambda Z: _batched(lambda B: model.predict_proba(scaler.transform(B))[:, 1], Z)


def fit_iforest(X_benign, seed, threads):
    # 1,000 trees: at 200, recall at a 1% budget swung 0.227-0.545 across seeds
    scaler = StandardScaler().fit(X_benign)
    model = IsolationForest(n_estimators=1000, random_state=seed, n_jobs=threads).fit(scaler.transform(X_benign))
    return lambda Z: _batched(lambda B: -model.decision_function(scaler.transform(B)), Z)


def fit_autoencoder(X_benign, seed, threads):
    from src.models.autoencoder import Autoencoder  # torch only loads when used
    model = Autoencoder(seed=seed, threads=threads).fit(X_benign)
    return model.score


def fit_all(X, y, train_idx, seed, threads, anomaly_only=False):
    """Fit every detector on the training rows; returns {name: score_fn}."""
    capped = cap_benign(train_idx, y, seed)
    X_benign = X[capped[y[capped] == 0]]
    detectors = {
        "iforest": fit_iforest(X_benign, seed, threads),
        "autoencoder": fit_autoencoder(X_benign, seed, threads),
    }
    if not anomaly_only:
        detectors["lgbm"] = fit_lgbm(X[capped], y[capped], seed, threads)
        detectors["logreg"] = fit_logreg(X[capped], y[capped], seed, threads)
    return detectors
