#!/usr/bin/env python3
"""Explain the one strong unseen-attack result before it is used anywhere.

Fold 2017_B (train Mon+Tue, validate Wed, test Thu): LightGBM catches most of
Thursday's web attacks at a 1% validation budget. Is that real transfer from
Tuesday's FTP/SSH brute force, or a shortcut? This script reports:
  1. recall per attack label (web brute force vs XSS vs SQL injection),
  2. the features that push web-attack flows toward "attack" (LightGBM
     contributions, mean over flagged web-attack flows),
  3. recall after retraining without the top features, one at a time,
  4. how many flows carry the single backward-window value that drives it.

Writes reports/results/explain_2017_B.json.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.evaluation.contract import PRIMARY_BUDGET, frozen_thresholds, provenance
from src.evaluation.folds import FOLDS_BY_NAME, day_key, partition
from src.models.detectors import cap_benign
from src.models.train import train_layer1_binary

DATA = ROOT / "data" / "processed" / "cicids2017_eval.parquet"
OUT = ROOT / "reports" / "results" / "explain_2017_B.json"
SEED, THREADS, TOP_K = 42, 6, 5


def fit(X, y, cols):
    scaler = StandardScaler().fit(X[:, cols])
    model = train_layer1_binary(scaler.transform(X[:, cols]), y, random_state=SEED, n_jobs=THREADS)
    return scaler, model


def recall_at_budget(scaler, model, cols, X, y, va, te, labels):
    def score(rows):
        return model.predict_proba(scaler.transform(X[rows][:, cols]))[:, 1]
    t = frozen_thresholds(score(va)[y[va] == 0])[str(PRIMARY_BUDGET)]
    test_scores = score(te)
    flagged = test_scores >= t
    per_label = {str(l): float(flagged[labels[te] == l].mean()) for l in np.unique(labels[te]) if l != "BENIGN"}
    return t, flagged, per_label, float(flagged[y[te] == 0].mean())


def main():
    df = pd.read_parquet(DATA)
    features = df.select_dtypes(include=[np.number]).columns.tolist()
    X = np.nan_to_num(df[features].to_numpy(dtype=np.float32), posinf=0.0, neginf=0.0)
    y = (df["Label"] != "BENIGN").to_numpy().astype(int)
    labels = df["Label"].to_numpy()
    days = df["Meta_source"].map(lambda s: day_key(s, "2017")).to_numpy()
    tr, va, te = partition(FOLDS_BY_NAME["2017_B"], days, y, SEED)
    tr = cap_benign(tr, y, SEED)
    all_cols = list(range(len(features)))

    with threadpool_limits(limits=THREADS):
        scaler, model = fit(X[tr], y[tr], all_cols)
        _, flagged, per_label, fpr = recall_at_budget(scaler, model, all_cols, X, y, va, te, labels)

        web_mask = np.char.startswith(labels[te].astype(str), "Web Attack")
        contrib = model.booster_.predict(scaler.transform(X[te[web_mask & flagged]]), pred_contrib=True)[:, :-1]
        mean_contrib = pd.Series(contrib.mean(axis=0), index=features).sort_values(ascending=False)
        top = mean_contrib.index[:TOP_K].tolist()

        ablations = {}
        for feature in top:
            cols = [i for i, f in enumerate(features) if f != feature]
            s, m = fit(X[tr], y[tr], cols)
            _, _, pl, fp = recall_at_budget(s, m, cols, X, y, va, te, labels)
            ablations[f"without {feature}"] = {"per_label_recall": pl, "test_fpr": fp}
        cols = [i for i, f in enumerate(features) if f not in top]
        s, m = fit(X[tr], y[tr], cols)
        _, _, pl, fp = recall_at_budget(s, m, cols, X, y, va, te, labels)
        ablations[f"without all top {TOP_K}"] = {"per_label_recall": pl, "test_fpr": fp}

    medians = {}
    for name, mask in [("Tuesday FTP-Patator", (days == "Tuesday") & (labels == "FTP-Patator")),
                       ("Tuesday SSH-Patator", (days == "Tuesday") & (labels == "SSH-Patator")),
                       ("Thursday web brute force", labels == "Web Attack - Brute Force"),
                       ("Thursday XSS", labels == "Web Attack - XSS"),
                       ("Benign, Mon+Tue", np.isin(days, ["Monday", "Tuesday"]) & (y == 0))]:
        medians[name] = df.loc[mask, top].median().round(2).to_dict()

    # The top feature is the server's initial TCP window. Share of flows with the
    # value most web attacks carry (the victim web server's window, if an artifact)
    win = df["Init_Win_bytes_backward"].to_numpy()
    web_all = np.char.startswith(labels.astype(str), "Web Attack")
    common = int(pd.Series(win[web_all]).mode().iloc[0])
    window_artifact = {
        "most_common_web_attack_value": common,
        "web_attack_share_with_value": float((win[web_all] == common).mean()),
        "benign_share_with_value": float((win[y == 0] == common).mean()),
        "benign_ports_with_value": {int(k): int(v) for k, v in
                                    pd.Series(df["Destination Port"].to_numpy()[(y == 0) & (win == common)]).value_counts().head(5).items()},
    }

    result = {
        "fold": "2017_B", "seed": SEED, "budget": PRIMARY_BUDGET,
        "test_fpr": fpr, "per_label_recall": per_label,
        "top_contributing_features": mean_contrib.head(10).round(4).to_dict(),
        "ablations": ablations, "medians_of_top_features": medians,
        "window_artifact": window_artifact,
        "provenance": provenance([DATA]),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1, default=float))
    print(json.dumps({k: result[k] for k in ["per_label_recall", "window_artifact"]}, indent=1))


if __name__ == "__main__":
    main()
