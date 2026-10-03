#!/usr/bin/env python3
"""Run the forward-in-time evaluation: every detector, every fold, every seed.

Usage:
    python scripts/run_pipeline.py                       # all folds, 5 seeds
    python scripts/run_pipeline.py --folds 2017_B --seeds 42

Writes one results file per fold to reports/results/<fold>.json. Notebooks and
the README read these files; they never compute metrics themselves.

Order inside a fold (nothing below the line sees test labels):
    fit on train -> score validation -> freeze thresholds, choose the hybrid
    ---------------------------------------------------------------------
    score test once -> evaluate at the frozen thresholds
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from src.evaluation.contract import BUDGETS, PRIMARY_BUDGET, evaluate, frozen_thresholds, provenance
from src.evaluation.folds import FOLDS, FOLDS_BY_NAME, day_key, partition
from src.models.cross_dataset import BENIGN_WEIGHT, FAMILY
from src.models.detectors import ANOMALY, SUPERVISED, cap_benign, fit_all, fit_lgbm
from src.models.evaluate import threshold_on_benign

DATA = {year: ROOT / "data" / "processed" / f"cicids{year}_eval.parquet" for year in ("2017", "2018")}
OUT_DIR = ROOT / "reports" / "results"
SHARES = [1.0, 0.75, 0.5, 0.25, 0.0]  # share of the alert budget given to the supervised model
SHORTCUT_SAMPLE = 200_000


def load(year, cache={}):
    if year not in cache:
        df = pd.read_parquet(DATA[year])
        features = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(features) != 71:
            raise ValueError(f"{year}: expected 71 numeric features, found {len(features)}")
        unmapped = set(df["Label"]) - set(FAMILY)
        if unmapped:
            raise ValueError(f"{year}: labels without a family: {sorted(unmapped)}")
        y = (df["Label"] != "BENIGN").to_numpy().astype(np.int8)
        cache[year] = {
            "features": features,
            "X": np.nan_to_num(df[features].to_numpy(dtype=np.float32), posinf=0.0, neginf=0.0),
            "y": y,
            "labels": df["Label"].to_numpy(),
            "families": df["Label"].map(FAMILY).to_numpy(),
            "days": df["Meta_source"].map(lambda s: day_key(s, year)).to_numpy(),
            "weight": np.where(y == 0, BENIGN_WEIGHT[year], 1.0),
        }
    return cache[year]


def hybrid_alerts(val_scores, test_scores, val_benign, sup, anom, share):
    """OR rule: supervised gets `share` of each budget, the anomaly model the rest."""
    out = {}
    for b in BUDGETS:
        # A zero share means that model never alerts. (A threshold "above every validation
        # benign score" would still fire on test flows more extreme than validation.)
        none = np.zeros(len(test_scores[sup]), dtype=bool)
        sup_alerts = test_scores[sup] >= threshold_on_benign(val_scores[sup][val_benign], b * share) if share > 0 else none
        anom_alerts = (test_scores[anom] >= threshold_on_benign(val_scores[anom][val_benign], b * (1 - share))
                       if share < 1 else none)
        out[str(b)] = sup_alerts | anom_alerts
    return out


def val_recall_at_primary(val_scores, val_y, sup, anom, share):
    val_benign = val_y == 0
    alerts = hybrid_alerts(val_scores, val_scores, val_benign, sup, anom, share)[str(PRIMARY_BUDGET)]
    return float(alerts[val_y == 1].mean())


def gates(tr, te, X_tr, X_te, y_tr, y_te, weight_te, seed, threads):
    """Leakage gates for one fold: exact twins, shuffled labels, single-feature shortcut."""
    h_tr = pd.util.hash_pandas_object(pd.DataFrame(X_tr), index=False).to_numpy()
    h_te = pd.util.hash_pandas_object(pd.DataFrame(X_te), index=False).to_numpy()
    twin = np.isin(h_te, h_tr)

    rng = np.random.RandomState(seed)
    capped = cap_benign(np.arange(len(y_tr)), y_tr, seed)
    shuffled = fit_lgbm(X_tr[capped], rng.permutation(y_tr[capped]), seed, threads)
    shuffled_ap = float(average_precision_score(y_te, shuffled(X_te), sample_weight=weight_te))

    sample = rng.choice(len(y_te), min(SHORTCUT_SAMPLE, len(y_te)), replace=False)
    ys, ws = y_te[sample], weight_te[sample]
    single = [max(average_precision_score(ys, X_te[sample, j], sample_weight=ws),
                  average_precision_score(ys, -X_te[sample, j], sample_weight=ws))
              for j in range(X_te.shape[1])]
    best = int(np.argmax(single))
    return {
        "exact_twin_test_rows": int(twin.sum()),
        "exact_twin_test_fraction": float(twin.mean()),
        "exact_twin_attack_rows": int((twin & (y_te == 1)).sum()),
        "shuffled_label_ap": shuffled_ap,
        "best_single_feature_index": best,
        "best_single_feature_ap": float(single[best]),
    }


def run_fold(fold, seed, threads):
    tr_data, te_data = load(fold.train_year), load(fold.test_year)
    cross_year = fold.train_year != fold.test_year
    tr, va, te = partition(fold, tr_data["days"], tr_data["y"], seed,
                           te_data["days"] if cross_year else None)
    X, y = tr_data["X"], tr_data["y"]
    X_te, y_te = te_data["X"][te], te_data["y"][te]
    w_te, fam_te = te_data["weight"][te], te_data["families"][te]
    y_va = y[va]
    val_benign = y_va == 0

    t0 = time.time()
    detectors = fit_all(X, y, tr, seed, threads, anomaly_only=fold.anomaly_only)
    val_scores = {n: f(X[va]) for n, f in detectors.items()}
    thresholds = {n: frozen_thresholds(s[val_benign]) for n, s in val_scores.items()}

    # Hybrid choices use validation only: best anomaly model, then best budget split
    choices = {}
    anom = max(ANOMALY, key=lambda a: float((val_scores[a][y_va == 1] >= thresholds[a][str(PRIMARY_BUDGET)]).mean())
               if y_va.any() else 0.0)
    for sup in [s for s in SUPERVISED if s in detectors]:
        search = {str(s): val_recall_at_primary(val_scores, y_va, sup, anom, s) for s in SHARES}
        choices[f"{sup}+{anom}"] = {"anomaly": anom, "share_supervised": float(max(SHARES, key=lambda s: search[str(s)])),
                                     "validation_recall_by_share": search}
    # ---- thresholds and choices are frozen; the test set is scored once below ----
    test_scores = {n: f(X_te) for n, f in detectors.items()}
    results = {n: evaluate(y_te, fam_te, test_scores[n], thresholds[n], w_te) for n in detectors}
    for name, c in choices.items():
        sup = name.split("+")[0]
        alerts = hybrid_alerts(val_scores, test_scores, val_benign, sup, c["anomaly"], c["share_supervised"])
        results[name] = evaluate(y_te, fam_te, alerts, {b: None for b in alerts}, w_te, ap=False)

    train_families = set(tr_data["families"][tr][y[tr] == 1])
    test_families = set(fam_te[y_te == 1])
    return {
        "seed": seed,
        "sizes": {"train": int(len(tr)), "validation": int(len(va)), "test": int(len(te)),
                  "validation_attacks": int(y_va.sum())},
        "unseen_test_families": sorted(test_families - train_families),
        "seen_test_families": sorted(test_families & train_families),
        "hybrid_choices": choices,
        "detectors": results,
        "gates": gates(tr, te, X[tr], X_te, y[tr], y_te, w_te, seed, threads),
        "seconds": round(time.time() - t0, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", nargs="+", choices=[f.name for f in FOLDS], default=[f.name for f in FOLDS])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with threadpool_limits(limits=args.threads):
        for name in args.folds:
            fold = FOLDS_BY_NAME[name]
            years = sorted({fold.train_year, fold.test_year})
            out = {"fold": {k: v for k, v in fold.__dict__.items()},
                   "features": load(fold.train_year)["features"],
                   "budgets": list(BUDGETS), "primary_budget": PRIMARY_BUDGET,
                   "provenance": provenance([DATA[y] for y in years]),
                   "complete": False, "runs": []}
            path = OUT_DIR / f"{name}.json"
            for seed in args.seeds:
                print(f"{name} seed {seed} ...", flush=True)
                run = run_fold(fold, seed, args.threads)
                out["runs"].append(run)
                path.write_text(json.dumps(out, indent=1, allow_nan=False, default=float))
                summary = ", ".join(f"{n} {r['budgets'][str(PRIMARY_BUDGET)]['recall']:.3f}"
                                    for n, r in run["detectors"].items())
                print(f"  recall at 1%: {summary} ({run['seconds']}s)", flush=True)
            out["complete"] = True
            path.write_text(json.dumps(out, indent=1, allow_nan=False, default=float))


if __name__ == "__main__":
    main()
