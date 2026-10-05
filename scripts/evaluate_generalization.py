#!/usr/bin/env python3
"""Controlled day-transfer evaluation; never overwrites production models.

Thresholds and candidate selection use validation data only. Existing day
holdouts have already been inspected: this is an exploratory comparison,
not a fresh confirmatory test or a production readiness certificate.
"""

import argparse
import hashlib
import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from scripts.train_models import DATA_PATH_V2, downsample_benign
from src.data.loader import get_feature_columns, prepare_binary_labels
from src.features.engineering import FAST_FEATURES
from src.models.evaluate import threshold_at_fpr, threshold_on_benign
from src.models.train import train_layer1_binary

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MODELS = ["lgbm20", "lgbm71", "logreg20", "logreg71", "lgbm71_no_network"]
SPLITS = ["random_split", "test_friday", "test_wed_thu", "forward_thursday"]
NETWORK_FEATURES = {"Destination Port", "Init_Win_bytes_forward", "Init_Win_bytes_backward"}
BUDGETS = [0.001, 0.01, 0.05]
PRIMARY_BUDGET = 0.01


def make_partitions(days, y, name, seed):
    """Return disjoint raw train/validation/test indices before downsampling.

    Random benchmark: 64/16/20. Day tests: latest available training day is
    validation. Wed+Thu is deliberately a nonchronological stress test.
    Forward Thursday uses Mon+Tue training, Wed validation, Thu testing.
    """
    idx = np.arange(len(y))
    if name == "random_split":
        pool, test = train_test_split(idx, test_size=0.2, stratify=y, random_state=seed)
        train, val = train_test_split(pool, test_size=0.2, stratify=y[pool], random_state=seed)
    else:
        test_days = {
            "test_friday": ["Friday"],
            "test_wed_thu": ["Wednesday", "Thursday"],
            "forward_thursday": ["Thursday"],
        }[name]
        val_day = {"test_friday": "Thursday", "test_wed_thu": "Friday",
                   "forward_thursday": "Wednesday"}[name]
        pool_days = [d for d in DAY_ORDER if d not in test_days and d != val_day]
        if name == "forward_thursday":
            pool_days = ["Monday", "Tuesday"]
        train = idx[np.isin(days, pool_days)]
        val = idx[days == val_day]
        test = idx[np.isin(days, test_days)]
    for label, subset in [("train", train), ("validation", val), ("test", test)]:
        if len(np.unique(y[subset])) != 2:
            raise ValueError(f"{name}: {label} must contain benign and attack flows")
    return train, val, test


def summarize_scores(y, labels, scores, threshold):
    alert = scores >= threshold
    tp = int(np.sum(alert & (y == 1)))
    fp = int(np.sum(alert & (y == 0)))
    return {
        "threshold": float(threshold),
        "n_benign": int(np.sum(y == 0)), "n_attack": int(np.sum(y == 1)),
        "fpr": float(alert[y == 0].mean()) if np.any(y == 0) else None,
        "recall": float(alert[y == 1].mean()) if np.any(y == 1) else None,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "tp": tp, "fp": fp,
        "fn": int(np.sum(~alert & (y == 1))),
        "tn": int(np.sum(~alert & (y == 0))),
        "alerts_per_10000_flows": float(alert.mean() * 10000),
        "per_attack": {
            str(label): {"n": int(np.sum(labels == label)),
                         "recall": float(alert[labels == label].mean())}
            for label in np.unique(labels[y == 1])
        },
    }


def score_quantiles(scores):
    return dict(zip(["p01", "p50", "p95", "p99", "p999"],
                    map(float, np.quantile(scores, [0.01, 0.5, 0.95, 0.99, 0.999]))))


def score_batches(model, scaler, X, rows, cols):
    scores = []
    for start in range(0, len(rows), 50000):
        batch = X[np.ix_(rows[start:start + 50000], cols)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
            scores.append(model.predict_proba(scaler.transform(batch))[:, 1])
    return np.concatenate(scores)


def evaluate_candidate(X, y, labels, train, val, test, cols, name, seed, threads, days):
    X_train = X[np.ix_(train, cols)]
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    if name.startswith("lgbm"):
        model = train_layer1_binary(X_train, y[train], random_state=seed, n_jobs=threads)
    else:
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        model.fit(X_train, y[train])
        if model.n_iter_.max() >= model.max_iter:
            raise RuntimeError(f"{name} did not converge; do not compare an unfinished fit")
    del X_train
    val_scores = score_batches(model, scaler, X, val, cols)
    # All thresholds are frozen here, before computing any outer-test scores.
    thresholds = {str(b): threshold_on_benign(val_scores[y[val] == 0], b) for b in BUDGETS}
    test_scores = score_batches(model, scaler, X, test, cols)
    validation = summarize_scores(y[val], labels[val], val_scores, thresholds[str(PRIMARY_BUDGET)])
    primary = validation["per_attack"].values()
    selection_score = float(np.mean([r["recall"] for r in primary]))
    test_report = {str(b): summarize_scores(y[test], labels[test], test_scores, t)
                   for b, t in ((b, thresholds[str(b)]) for b in BUDGETS)}
    retrospective = {}
    for b in BUDGETS:
        t, fpr, recall = threshold_at_fpr(y[test], test_scores, b)
        retrospective[str(b)] = {"threshold": float(t) if np.isfinite(t) else None,
                                 "fpr": fpr, "recall": recall}
    train_benign = train[y[train] == 0]
    train_scores = score_batches(model, scaler, X, train_benign, cols)
    return {
        "n_features": len(cols), "validation_macro_attack_recall": selection_score,
        "validation_at_1pct": validation,
        "test_average_precision": float(average_precision_score(y[test], test_scores)),
        "test_prevalence": float(y[test].mean()),
        "test_at_frozen_thresholds": test_report,
        "test_at_default_0_5": summarize_scores(y[test], labels[test], test_scores, 0.5),
        "test_by_day_at_1pct": {
            str(day): summarize_scores(y[test][mask], labels[test][mask], test_scores[mask],
                                       thresholds[str(PRIMARY_BUDGET)])
            for day in np.unique(days[test]) if (mask := days[test] == day).any()
        },
        "retrospective_test_selected_thresholds": retrospective,
        "model_parameters": model.get_params(),
        "benign_score_quantiles": {"train": score_quantiles(train_scores),
                                   "validation": score_quantiles(val_scores[y[val] == 0]),
                                   "test": score_quantiles(test_scores[y[test] == 0])},
    }


def write_summary(result, path):
    """Write ranges across seeds, keeping operational FPR beside recall."""
    lines = ["# Controlled day-transfer evaluation", "",
             "The models do not establish reliable generalization across all tested days. "
             "This comparison separates ranking from detection at validation-frozen thresholds.", "",
             "Source: [generalization_evaluation.json](generalization_evaluation.json). "
             "All ranges below are minimum–maximum over seeds, not confidence intervals.", "",
             "## Protocol", "",
             "Training-only benign cap: 200,000. Scalers fit on training only. "
             "Validation/test prevalence is unchanged. Thresholds use validation benign scores only; "
             "ties are handled so empirical validation FPR stays within budget.", "",
             "Friday: train Mon–Wed, validate Thu, test Fri. "
             "Wed+Thu: train Mon+Tue, validate Fri, test Wed+Thu (nonchronological). "
             "Forward Thursday: train Mon+Tue, validate Wed, test Thu. "
             "Random benchmark: 64/16/20 train/validation/test, before training downsampling.", "",
             "## All candidates at the 1% validation FPR budget", "",
             "| Test | Candidate | Test AP | Actual test FPR | Test recall |",
             "|---|---|---:|---:|---:|"]
    def span(values):
        return f"{min(values):.4f}–{max(values):.4f}"
    for split in result["config"]["splits"]:
        runs = [r for r in result["runs"] if r["split"] == split]
        for name in result["config"]["models"]:
            reports = [r["candidates"][name] for r in runs]
            primary = [r["test_at_frozen_thresholds"]["0.01"] for r in reports]
            lines.append(f"| {split} | {name} | {span([r['test_average_precision'] for r in reports])} "
                         f"| {span([r['fpr'] for r in primary])} | {span([r['recall'] for r in primary])} |")
    lines += ["", "## Selection without outer-test tuning", "",
              "Candidates are selected by validation macro attack recall at the 1% validation budget. "
              "The selection itself can fail to transfer: validation and test contain different attack labels.", "",
              "| Test | Seed | Validation-selected candidate | Actual test FPR | Test recall |",
              "|---|---:|---|---:|---:|"]
    for run in result["runs"]:
        name = run["selected_on_validation"]
        primary = run["candidates"][name]["test_at_frozen_thresholds"]["0.01"]
        lines.append(f"| {run['split']} | {run['seed']} | {name} | {primary['fpr']:.4f} | {primary['recall']:.4f} |")
    lines += ["", "## Scope of the conclusion", ""]
    lines += [f"- {limitation}" for limitation in result["limitations"]]
    lines += ["", "A candidate with better recall but a test FPR above 1% has exceeded the comparison budget. "
              "A higher AP alone does not remedy this. No production artifacts were replaced.", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_PATH_V2)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/generalization_evaluation.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS[:4])
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=SPLITS)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    df = pd.read_parquet(args.data)
    features = get_feature_columns(df)
    missing = set(FAST_FEATURES) - set(features)
    if missing or len(features) != 71:
        raise ValueError(f"Expected the v2 71-feature schema; missing fast features: {sorted(missing)}")
    X = np.nan_to_num(df[features].to_numpy(dtype=float), copy=False,
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = prepare_binary_labels(df)
    labels = df["Label"].to_numpy()
    days = df["Meta_source"].str.split("-").str[0].to_numpy()
    support = pd.crosstab(pd.Series(days, name="day"), df["Label"]).to_dict(orient="index")
    data_hash = hashlib.sha256()
    with args.data.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            data_hash.update(block)
    digest = data_hash.hexdigest()
    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "complete": False,
        "protocol": "validation-frozen thresholds; exploratory outer evaluation",
        "data": {"path": str(args.data.relative_to(ROOT)) if args.data.is_relative_to(ROOT) else str(args.data),
                 "sha256": digest, "rows": len(df), "features": features},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__, "lightgbm": lightgbm.__version__},
        "code_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [Path(__file__), ROOT / "src/models/train.py", ROOT / "src/models/evaluate.py",
                                  ROOT / "scripts/train_models.py", ROOT / "src/features/engineering.py"]},
        "config": {"seeds": args.seeds, "models": args.models, "splits": args.splits,
                   "threads": args.threads, "budgets": BUDGETS,
                   "selection": "highest validation macro attack recall at <=1% validation benign FPR; ties follow model order"},
        "day_attack_support": support,
        "limitations": ["Day and attack family change together; no causal isolation.",
                        "Only file provenance is available, not session IDs or timestamps.",
                        "Seed ranges measure fit/downsampling variation, not population confidence intervals.",
                        "Outer holdouts were previously inspected; final confirmation needs fresh captures.",
                        "No operational minimum recall has been agreed; no model is certified for deployment."],
        "runs": [],
    }
    del df
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads):
        for seed in args.seeds:
            for split in args.splits:
                raw_train, val, test = make_partitions(days, y, split, seed)
                train = downsample_benign(raw_train, y, seed=seed)
                run = {"seed": seed, "split": split,
                       "chronological": split in {"test_friday", "forward_thursday"},
                       "partitions": {p: {"n": len(idx), "attack_prevalence": float(y[idx].mean()),
                                          "days": sorted(np.unique(days[idx]).tolist()),
                                          "index_sha256": hashlib.sha256(idx.tobytes()).hexdigest()}
                                      for p, idx in [("train", train), ("validation", val), ("test", test)]},
                       "candidates": {}}
                result["runs"].append(run)
                for name in args.models:
                    selected = FAST_FEATURES if name.endswith("20") else features
                    if name == "lgbm71_no_network":
                        selected = [f for f in features if f not in NETWORK_FEATURES]
                    cols = [features.index(f) for f in selected]
                    print(f"{split} seed={seed} {name}: train={len(train):,} val={len(val):,} test={len(test):,}", flush=True)
                    report = evaluate_candidate(X, y, labels, train, val, test, cols, name, seed, args.threads, days)
                    report["features"] = [features[i] for i in cols]
                    run["candidates"][name] = report
                    primary = report["test_at_frozen_thresholds"][str(PRIMARY_BUDGET)]
                    print(f"  AP={report['test_average_precision']:.4f} test FPR={primary['fpr']:.4f} recall={primary['recall']:.4f}", flush=True)
                    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                run["selected_on_validation"] = max(run["candidates"], key=lambda n: run["candidates"][n]["validation_macro_attack_recall"])
                args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    result["complete"] = True
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_summary(result, args.output.with_suffix(".md"))
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
