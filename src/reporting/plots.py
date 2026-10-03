"""Shared figures for the numbered notebooks. They read results files; no metrics here.

Colors follow the detector, never its rank (validated categorical palette,
light surface; three slots are under 3:1 contrast, so every chart carries
direct labels and a table of the same numbers beside it).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS = ROOT / "reports" / "results"

COLORS = {
    "lgbm": "#2a78d6",
    "logreg": "#eb6834",
    "iforest": "#1baf7a",
    "autoencoder": "#eda100",
    "lgbm+anomaly": "#e87ba4",
    "logreg+anomaly": "#008300",
}
NAMES = {
    "lgbm": "LightGBM", "logreg": "Logistic regression", "iforest": "Isolation Forest",
    "autoencoder": "Autoencoder",
    "lgbm+anomaly": "Hybrid: LightGBM + anomaly model",
    "logreg+anomaly": "Hybrid: log. regression + anomaly model",
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"


def load_summary():
    return json.loads((RESULTS / "summary.json").read_text())


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def recall_table(summary, folds, budget="0.01"):
    """Median recall and actual FPR (min-max over seeds) per fold and detector."""
    rows = []
    for fold in folds:
        f = summary["folds"][fold]
        for det, d in f["detectors"].items():
            r, fpr = d["budgets"][budget]["recall"], d["budgets"][budget]["fpr"]
            rows.append({"fold": fold, "question": f["question"], "detector": NAMES[det],
                         "recall (median)": round(r["median"], 3),
                         "recall range": f"{r['min']:.3f}-{r['max']:.3f}",
                         "actual FPR range": f"{fpr['min']:.4f}-{fpr['max']:.4f}",
                         "AP (median)": (round(d["average_precision"]["median"], 3)
                                         if d["average_precision"] else None),
                         "prevalence": round(f["prevalence"], 4)})
    return pd.DataFrame(rows)


def recall_dots(summary, folds, title, fold_labels=None, budget="0.01"):
    """One row per fold x detector: dot at the median recall, whisker = min-max over seeds.

    Filled dot = actual test FPR stayed within budget; hollow = over budget (not quotable).
    Rows are labelled with the detector name, so identity never rests on color alone.
    """
    fold_labels = fold_labels or {f: f for f in folds}
    rows, y, ticks, headers = [], 0, [], []
    for fold in folds:
        headers.append((y, fold_labels[fold]))
        y += 1
        for det, d in summary["folds"][fold]["detectors"].items():
            rows.append((y, det, d))
            ticks.append((y, NAMES[det]))
            y += 1
        y += 0.6
    fig, ax = plt.subplots(figsize=(10, 0.32 * y + 1.2))
    for yy, det, d in rows:
        r = d["budgets"][budget]["recall"]
        over = d["budget_violated"][budget]
        ax.plot([r["min"], r["max"]], [yy, yy], color=COLORS[det], linewidth=2, solid_capstyle="round")
        ax.plot(r["median"], yy, "o", markersize=7, markeredgewidth=1.8, markeredgecolor=COLORS[det],
                markerfacecolor="white" if over else COLORS[det])
        ax.text(r["max"] + 0.012, yy, f"{r['median']:.2f}" + ("  over budget" if over else ""),
                va="center", fontsize=8, color=MUTED if over else INK)
    for yy, label in headers:
        ax.text(-0.01, yy, label, ha="right", va="center", fontsize=9.5, fontweight="bold", color=INK,
                transform=ax.get_yaxis_transform())
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([n for _, n in ticks], fontsize=8.5)
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel(f"Recall at a {float(budget):.0%} validation false-alarm budget  ·  "
                  "dot = median, line = seed range, hollow = over budget", fontsize=9)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    style(ax)
    fig.tight_layout()
    return fig
