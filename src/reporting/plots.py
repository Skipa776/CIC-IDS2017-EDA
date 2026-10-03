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
    "lgbm+autoencoder": "#e87ba4",
    "lgbm+iforest": "#e87ba4",
    "logreg+autoencoder": "#008300",
    "logreg+iforest": "#008300",
}
NAMES = {
    "lgbm": "LightGBM", "logreg": "Logistic regression", "iforest": "Isolation Forest",
    "autoencoder": "Autoencoder", "lgbm+autoencoder": "Hybrid: LightGBM + autoencoder",
    "lgbm+iforest": "Hybrid: LightGBM + Isolation Forest",
    "logreg+autoencoder": "Hybrid: log. regression + autoencoder",
    "logreg+iforest": "Hybrid: log. regression + Isolation Forest",
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
    """One row per fold; per detector a dot at the median recall, whisker = min-max over seeds."""
    fold_labels = fold_labels or {f: f for f in folds}
    dets = []
    for f in folds:
        dets += [d for d in summary["folds"][f]["detectors"] if d not in dets]
    fig, ax = plt.subplots(figsize=(10, 1.1 + 0.9 * len(folds)))
    step = 0.7 / max(1, len(dets) - 1)
    for i, fold in enumerate(folds):
        for j, det in enumerate(dets):
            d = summary["folds"][fold]["detectors"].get(det)
            if d is None:
                continue
            r = d["budgets"][budget]["recall"]
            y = i - 0.35 + j * step
            ax.plot([r["min"], r["max"]], [y, y], color=COLORS[det], linewidth=2, solid_capstyle="round")
            ax.plot(r["median"], y, "o", color=COLORS[det], markersize=7, markeredgecolor="white",
                    markeredgewidth=1.5, label=NAMES[det] if i == 0 or det not in summary["folds"][folds[0]]["detectors"] else None)
            ax.text(r["max"] + 0.012, y, f"{r['median']:.2f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(folds)))
    ax.set_yticklabels([fold_labels[f] for f in folds])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel(f"Recall: share of test attack flows flagged, at a {float(budget):.0%} validation false-alarm budget")
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    style(ax)
    handles, labels = ax.get_legend_handles_labels()
    seen = dict(zip(labels, handles))
    ax.legend(seen.values(), seen.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.18 - 0.1 / len(folds)),
              ncol=3, frameon=False, fontsize=8.5)
    fig.tight_layout()
    return fig
