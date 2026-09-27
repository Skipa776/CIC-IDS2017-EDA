#!/usr/bin/env python3
"""
Cross-day threshold analysis for Layer 1.

The default 0.5 threshold catches only a few percent of attack flows on
held-out days. This script asks what recall is available at fixed false
positive rates, using the same splits and Layer 1 training as
scripts/train_models.py.

Usage:
    python scripts/crossday_threshold_analysis.py

Writes:
    reports/crossday_threshold_analysis.json
    reports/crossday_pr_curves.png
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

from scripts.train_models import DATA_PATH_V2, crossday_splits, random_split
from src.data.loader import get_feature_columns, load_processed_data, prepare_binary_labels
from src.features.engineering import FAST_FEATURES, create_scaler
from src.models.train import train_layer1_binary

TARGET_FPRS = [0.001, 0.01, 0.05]
PER_ATTACK_FPR = 0.01
OUT_JSON = ROOT / "reports" / "crossday_threshold_analysis.json"
OUT_PNG = ROOT / "reports" / "crossday_pr_curves.png"

# Okabe-Ito colorblind-safe palette
COLORS = {"random_split": "#0072B2", "test_friday": "#E69F00", "test_wed_thu": "#009E73"}
TITLES = {
    "random_split": "Random 80/20 split",
    "test_friday": "Friday held out",
    "test_wed_thu": "Wed+Thu held out",
}


def threshold_at_fpr(y_true, scores, target_fpr):
    """Lowest score threshold whose FPR on benign stays at or below target_fpr."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    i = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(thresholds[i]), float(fpr[i]), float(tpr[i])


def evaluate_split(X, y, labels, tr, te):
    """Train Layer 1 as train_models.py does and score the held-out rows."""
    scaler = create_scaler(X[tr])
    model = train_layer1_binary(scaler.transform(X[tr]), y[tr])
    scores = model.predict_proba(scaler.transform(X[te]))[:, 1]
    y_te, labels_te = y[te], labels[te]

    at_fpr = {}
    for target in TARGET_FPRS:
        thr, fpr, rec = threshold_at_fpr(y_te, scores, target)
        at_fpr[str(target)] = {"threshold": thr, "actual_fpr": fpr, "recall": rec}

    thr = at_fpr[str(PER_ATTACK_FPR)]["threshold"]
    flagged = scores >= thr
    per_attack = {
        label: {"n_test": int((labels_te == label).sum()),
                "recall": float(flagged[labels_te == label].mean())}
        for label in np.unique(labels_te) if label != "BENIGN"
    }

    result = {
        "n_test": int(len(te)),
        "pr_auc": float(average_precision_score(y_te, scores)),
        "no_skill_pr_auc": float(y_te.mean()),
        "recall_at_fpr": at_fpr,
        f"per_attack_recall_at_fpr_{PER_ATTACK_FPR}": per_attack,
    }
    return result, precision_recall_curve(y_te, scores)


def plot_pr_curves(curves, results):
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, (precision, recall, _) in curves.items():
        color = COLORS[name]
        ax.plot(recall, precision, color=color,
                label=f"{TITLES[name]} (PR-AUC {results[name]['pr_auc']:.3f})")
        ax.axhline(results[name]["no_skill_pr_auc"], color=color, linestyle="--", linewidth=1,
                   label=f"No skill, {TITLES[name]} ({results[name]['no_skill_pr_auc']:.3f})")
    ax.set_xlabel("Recall (attack flows detected)")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title("Layer 1 precision-recall: random split vs held-out days")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


def main():
    if not DATA_PATH_V2.exists():
        sys.exit(f"Missing {DATA_PATH_V2}. Build it with: python scripts/build_dataset.py")

    df = load_processed_data(parquet_path=DATA_PATH_V2)
    features = [f for f in FAST_FEATURES if f in get_feature_columns(df)]
    X = np.nan_to_num(df[features].values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    y = prepare_binary_labels(df)
    labels = df["Label"].values

    splits = [("random_split", *random_split(y))]
    splits += [(name, tr, te) for name, _, tr, te in crossday_splits(df, y)]

    results, curves = {}, {}
    for name, tr, te in splits:
        print(f"{name}: training on {len(tr):,}, testing on {len(te):,}")
        results[name], curves[name] = evaluate_split(X, y, labels, tr, te)
        for target, r in results[name]["recall_at_fpr"].items():
            print(f"  FPR {float(target):.3f} (actual {r['actual_fpr']:.4f}): recall {r['recall']:.3f}")

    OUT_JSON.write_text(json.dumps({"features": features, "splits": results}, indent=2))
    plot_pr_curves(curves, results)
    print(f"Wrote {OUT_JSON} and {OUT_PNG}")


if __name__ == "__main__":
    main()
