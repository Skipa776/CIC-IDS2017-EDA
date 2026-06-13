#!/usr/bin/env python3
"""
Validation Suite for the cicids2017_eda.ipynb Logistic Regression Baseline

Tests whether the notebook's reported PR-AUC of 0.996 is plausible or an
artifact of the evaluation design, via seven experiments:

1. Baseline reproduction  - replicate the notebook pipeline exactly (control)
2. Duplicate-leakage audit - measure train/test row overlap, rerun deduplicated
3. Cross-day holdout       - split by Meta_source instead of randomly
4. Destination Port ablation - drop the port feature / use port alone
5. Realistic prevalence    - evaluate on a test set with the natural benign share
6. Label-shuffle sanity    - randomized labels must collapse to the no-skill line
7. Per-class recall        - which attack types carry the binary score

Usage:
    python scripts/validate_eda_baseline.py [--data PATH] [--tag SUFFIX]

    --data  parquet to validate (default: data/processed/cicids2017_clean.parquet)
    --tag   suffix for output filenames, e.g. "_v2" (default: "")

Output:
    - Console report
    - reports/eda_baseline_validation<tag>.json
    - reports/figures/validation_pr_curves<tag>.png
    - reports/figures/validation_per_class_recall<tag>.png
"""

import argparse
import json
import os
import warnings
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed/cicids2017_clean.parquet")
FIGURES_PATH = os.path.join(PROJECT_ROOT, "reports/figures")

# Mirrors notebook cell 54
MAX_BENIGN = 200_000
RANDOM_STATE = 42
TEST_SIZE = 0.2

FRIDAY_SOURCES = [
    "Friday-WorkingHours-Morning",
    "Friday-WorkingHours-Afternoon-PortScan",
    "Friday-WorkingHours-Afternoon-DDos",
]
WED_THU_SOURCES = [
    "Wednesday-workingHours",
    "Thursday-WorkingHours-Morning-WebAttacks",
    "Thursday-WorkingHours-Afternoon-Infilteration",
]


def make_logreg():
    """Same hyperparameters as notebook cell 56."""
    return LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",
    )


def fit_eval(X_train, y_train, X_test, y_test):
    """Scale on train only, fit LogReg, return metrics + test probabilities."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = make_logreg()
    model.fit(X_train_s, y_train)

    train_proba = model.predict_proba(X_train_s)[:, 1]
    test_proba = model.predict_proba(X_test_s)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average="binary", zero_division=0
    )
    metrics = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "attack_rate_test": round(float(y_test.mean()), 4),
        "no_skill_pr_auc": round(float(y_test.mean()), 4),
        "pr_auc_train": round(float(average_precision_score(y_train, train_proba)), 4),
        "pr_auc_test": round(float(average_precision_score(y_test, test_proba)), 4),
        "precision_attack": round(float(precision), 4),
        "recall_attack": round(float(recall), 4),
        "f1_attack": round(float(f1), 4),
    }
    return metrics, model, test_proba


def downsample_benign(y_binary, rng_seed=RANDOM_STATE, max_benign=MAX_BENIGN):
    """Replicate notebook cell 54: cap benign rows, keep all attacks."""
    benign_idx = np.where(y_binary == 0)[0]
    attack_idx = np.where(y_binary == 1)[0]
    if len(benign_idx) > max_benign:
        rng = np.random.RandomState(rng_seed)
        benign_idx = rng.choice(benign_idx, size=max_benign, replace=False)
    return np.concatenate([benign_idx, attack_idx])


def load_data(data_path):
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns]
    print(f"  {df.shape[0]:,} rows, {len(feature_cols)} numeric features")
    return df, feature_cols


def exp1_baseline(df, feature_cols):
    """Replicate the notebook pipeline; also return split pieces for exp7."""
    print("\n[1/7] Baseline reproduction (notebook protocol)...")
    y_all = (df["Label"] != "BENIGN").astype(int).values
    keep = downsample_benign(y_all)
    X = df[feature_cols].values[keep]
    y = y_all[keep]
    labels = df["Label"].values[keep]

    X_tr, X_te, y_tr, y_te, _, lab_te = train_test_split(
        X, y, labels, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    metrics, model, test_proba = fit_eval(X_tr, y_tr, X_te, y_te)
    print(f"  test PR-AUC = {metrics['pr_auc_test']}  (train {metrics['pr_auc_train']}, "
          f"no-skill {metrics['no_skill_pr_auc']})")
    return metrics, (y_te, lab_te, test_proba)


def exp2_duplicates(df, feature_cols):
    """Quantify duplicate leakage; rerun the baseline on deduplicated data."""
    print("\n[2/7] Duplicate-leakage audit...")
    n_dup = int(df.duplicated(subset=feature_cols).sum())
    n_dup_lab = int(df.duplicated(subset=feature_cols + ["Label"]).sum())

    # Overlap under the baseline split: share of test rows whose exact feature
    # vector also appears in train.
    y_all = (df["Label"] != "BENIGN").astype(int).values
    keep = downsample_benign(y_all)
    feats = df[feature_cols].iloc[keep].reset_index(drop=True)
    y = y_all[keep]
    row_hash = pd.util.hash_pandas_object(feats, index=False).values
    h_tr, h_te = train_test_split(
        row_hash, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    overlap = float(np.isin(h_te, h_tr).mean())
    print(f"  duplicate feature rows: {n_dup:,} ({n_dup / len(df):.1%}); "
          f"test rows with exact twin in train: {overlap:.1%}")

    dedup = df.drop_duplicates(subset=feature_cols + ["Label"])
    y_d = (dedup["Label"] != "BENIGN").astype(int).values
    keep_d = downsample_benign(y_d)
    X_d = dedup[feature_cols].values[keep_d]
    y_d = y_d[keep_d]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_d, y_d, test_size=TEST_SIZE, stratify=y_d, random_state=RANDOM_STATE
    )
    metrics, _, test_proba = fit_eval(X_tr, y_tr, X_te, y_te)
    metrics.update(
        {
            "duplicate_feature_rows": n_dup,
            "duplicate_feature_label_rows": n_dup_lab,
            "test_rows_with_exact_twin_in_train": round(overlap, 4),
        }
    )
    print(f"  deduplicated test PR-AUC = {metrics['pr_auc_test']}")
    return metrics, (y_te, test_proba)


def crossday_split(df, feature_cols, test_sources, name):
    """Train on all days except test_sources; benign downsampled in train only."""
    test_mask = df["Meta_source"].isin(test_sources).values
    y_all = (df["Label"] != "BENIGN").astype(int).values

    train_df = df[~test_mask]
    y_tr_full = y_all[~test_mask]
    keep = downsample_benign(y_tr_full)
    X_tr = train_df[feature_cols].values[keep]
    y_tr = y_tr_full[keep]

    X_te = df[test_mask][feature_cols].values
    y_te = y_all[test_mask]

    metrics, _, test_proba = fit_eval(X_tr, y_tr, X_te, y_te)
    metrics["test_days"] = test_sources
    metrics["unseen_attack_types_in_test"] = sorted(
        set(df[test_mask]["Label"]) - set(train_df["Label"]) - {"BENIGN"}
    )
    print(f"  {name}: test PR-AUC = {metrics['pr_auc_test']} "
          f"(no-skill {metrics['no_skill_pr_auc']})")
    return metrics, (y_te, test_proba)


def exp3_crossday(df, feature_cols):
    print("\n[3/7] Cross-day (temporal) holdout...")
    m_a, curve_a = crossday_split(df, feature_cols, FRIDAY_SOURCES, "train Mon-Thu / test Fri")
    m_b, curve_b = crossday_split(df, feature_cols, WED_THU_SOURCES, "train Mon+Tue+Fri / test Wed+Thu")
    return {"train_monthu_test_friday": m_a, "train_montuefri_test_wedthu": m_b}, curve_a, curve_b


def exp4_port_ablation(df, feature_cols):
    print("\n[4/7] Destination Port ablation...")
    y_all = (df["Label"] != "BENIGN").astype(int).values
    keep = downsample_benign(y_all)
    y = y_all[keep]

    results = {}
    for name, cols in [
        ("without_destination_port", [c for c in feature_cols if c != "Destination Port"]),
        ("destination_port_only", ["Destination Port"]),
    ]:
        X = df[cols].values[keep]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )
        metrics, _, _ = fit_eval(X_tr, y_tr, X_te, y_te)
        metrics["n_features"] = len(cols)
        results[name] = metrics
        print(f"  {name}: test PR-AUC = {metrics['pr_auc_test']}")
    return results


def exp5_realistic_prevalence(df, feature_cols):
    """Train as the notebook does, but test at the natural benign share."""
    print("\n[5/7] Realistic-prevalence evaluation...")
    y_all = (df["Label"] != "BENIGN").astype(int).values
    X_all = df[feature_cols].values

    idx_tr, idx_te = train_test_split(
        np.arange(len(y_all)), test_size=TEST_SIZE, stratify=y_all,
        random_state=RANDOM_STATE,
    )
    keep_tr = idx_tr[downsample_benign(y_all[idx_tr])]
    metrics, _, test_proba = fit_eval(
        X_all[keep_tr], y_all[keep_tr], X_all[idx_te], y_all[idx_te]
    )
    print(f"  natural-prevalence test PR-AUC = {metrics['pr_auc_test']} "
          f"(no-skill {metrics['no_skill_pr_auc']})")
    return metrics, (y_all[idx_te], test_proba)


def exp6_shuffle(df, feature_cols):
    print("\n[6/7] Label-shuffle sanity test...")
    y_all = (df["Label"] != "BENIGN").astype(int).values
    keep = downsample_benign(y_all)
    X = df[feature_cols].values[keep]
    y = y_all[keep]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    rng = np.random.RandomState(RANDOM_STATE)
    y_tr_shuffled = rng.permutation(y_tr)
    metrics, _, _ = fit_eval(X_tr, y_tr_shuffled, X_te, y_te)
    print(f"  shuffled-label test PR-AUC = {metrics['pr_auc_test']} "
          f"(should be near no-skill {metrics['no_skill_pr_auc']})")
    return metrics


def exp7_per_class(baseline_split):
    """Recall of the binary baseline per original attack label."""
    print("\n[7/7] Per-class recall on the baseline test set...")
    y_te, lab_te, test_proba = baseline_split
    pred = (test_proba >= 0.5).astype(int)
    rows = {}
    for label in sorted(set(lab_te)):
        mask = lab_te == label
        detected = float(pred[mask].mean())  # for BENIGN this is false-positive rate
        rows[label] = {"n_test": int(mask.sum()), "flagged_as_attack": round(detected, 4)}
        print(f"  {label:<35s} n={mask.sum():>6,}  flagged={detected:.3f}")
    return rows


def save_figures(curves, per_class, tag=""):
    os.makedirs(FIGURES_PATH, exist_ok=True)

    plt.figure(figsize=(9, 7))
    for name, (y_true, proba) in curves.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("LogReg baseline under different evaluation protocols")
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    pr_path = os.path.join(FIGURES_PATH, f"validation_pr_curves{tag}.png")
    plt.savefig(pr_path, dpi=120)
    plt.close()

    labels = [l for l in per_class if l != "BENIGN"]
    vals = [per_class[l]["flagged_as_attack"] for l in labels]
    plt.figure(figsize=(11, 5))
    plt.bar(labels, vals, color="steelblue")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Recall (share flagged as attack)")
    plt.title("Per-attack-type recall of the binary LogReg baseline (random split)")
    plt.tight_layout()
    pc_path = os.path.join(FIGURES_PATH, f"validation_per_class_recall{tag}.png")
    plt.savefig(pc_path, dpi=120)
    plt.close()
    print(f"\nFigures saved: {pr_path}, {pc_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA_PATH, help="parquet file to validate")
    parser.add_argument("--tag", default="", help="suffix for output filenames")
    args = parser.parse_args()

    report_path = os.path.join(
        PROJECT_ROOT, f"reports/eda_baseline_validation{args.tag}.json"
    )
    df, feature_cols = load_data(args.data)

    baseline, baseline_split = exp1_baseline(df, feature_cols)
    dedup, dedup_curve = exp2_duplicates(df, feature_cols)
    crossday, curve_a, curve_b = exp3_crossday(df, feature_cols)
    ablation = exp4_port_ablation(df, feature_cols)
    realistic, realistic_curve = exp5_realistic_prevalence(df, feature_cols)
    shuffle = exp6_shuffle(df, feature_cols)
    per_class = exp7_per_class(baseline_split)

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "data_path": args.data,
        "n_rows": int(len(df)),
        "n_features": len(feature_cols),
        "experiments": {
            "1_baseline_reproduction": baseline,
            "2_deduplicated": dedup,
            "3_crossday_holdout": crossday,
            "4_port_ablation": ablation,
            "5_realistic_prevalence": realistic,
            "6_label_shuffle": shuffle,
            "7_per_class_recall": per_class,
        },
    }
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")

    save_figures(
        {
            "random split (notebook)": (baseline_split[0], baseline_split[2]),
            "deduplicated": dedup_curve,
            "cross-day: test Friday": curve_a,
            "cross-day: test Wed+Thu": curve_b,
            "natural prevalence": realistic_curve,
        },
        per_class,
        tag=args.tag,
    )

    print("\n=== SUMMARY (test PR-AUC vs no-skill baseline) ===")
    rows = [
        ("Notebook protocol (random split)", baseline),
        ("Deduplicated", dedup),
        ("Cross-day: test Friday", crossday["train_monthu_test_friday"]),
        ("Cross-day: test Wed+Thu", crossday["train_montuefri_test_wedthu"]),
        ("No Destination Port", ablation["without_destination_port"]),
        ("Destination Port only", ablation["destination_port_only"]),
        ("Natural prevalence test", realistic),
        ("Shuffled labels (sanity)", shuffle),
    ]
    for name, m in rows:
        print(f"  {name:<38s} {m['pr_auc_test']:.3f}  (no-skill {m['no_skill_pr_auc']:.3f})")


if __name__ == "__main__":
    main()
