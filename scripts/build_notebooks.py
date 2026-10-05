#!/usr/bin/env python3
"""Generate the numbered notebooks (00-06) from the results files.

Each notebook opens with its question, then a code cell that loads results
and prints the finding, key figure and the numbers behind it. Computation
lives in src/ and scripts/run_pipeline.py; notebooks only read and plot.

Usage:
    python scripts/build_notebooks.py            # write notebooks/0*.ipynb
    then execute: jupyter nbconvert --to notebook --execute --inplace notebooks/0*.ipynb
"""

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.notebook_text import TEXT  # narrative written after the results were read

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = """import sys, json, warnings
sys.path.insert(0, "..")  # make src/ importable from notebooks/
warnings.filterwarnings("ignore", message="X does not have valid feature names")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.reporting.plots import load_summary, recall_dots, recall_table, style, COLORS, NAMES, INK, MUTED
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 60)
REPORTS = "../reports"
"""


def notebook(name, cells):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nbf.write(nb, ROOT / "notebooks" / f"{name}.ipynb")


def nb00():
    notebook("00_data_and_cleaning", [
        md(TEXT["00_intro"]),
        code(SETUP + """
m17 = json.load(open(f"{REPORTS}/dataset_v2_manifest.json"))
m17e = json.load(open(f"{REPORTS}/dataset_2017_eval_manifest.json"))
m18 = json.load(open(f"{REPORTS}/dataset_2018_manifest.json"))
m18e = json.load(open(f"{REPORTS}/dataset_2018_eval_manifest.json"))
import pyarrow.parquet as pq
v1_rows = pq.ParquetFile("../data/processed/cicids2017_clean.parquet").metadata.num_rows

print(f"Finding: the first cleaning pass kept {v1_rows:,} of {m17['rows_raw']:,} CIC-IDS2017 rows "
      f"({v1_rows / m17['rows_raw']:.1%}). The fix keeps {m17['sentinel_flows_kept']:,} flows that report "
      f"'no TCP window' instead of deleting them; the evaluation copy keeps {m17e['rows_clean']:,} rows "
      f"({m17e['rows_clean'] / m17e['rows_raw']:.1%}).")

steps = [("Raw CSVs", m17["rows_raw"]),
         ("Drop infinite rates", m17["rows_raw"] - m17["rows_dropped_inf_rates"]),
         ("Drop impossible negatives", m17["rows_raw"] - m17["rows_dropped_inf_rates"] - m17["rows_dropped_negative"]),
         ("Drop exact duplicates (model-training copy only)", m17["rows_clean"] + m17["rows_dropped_contradictory_labels"]),
         ("Drop contradictory labels (model-training copy only)", m17["rows_clean"]),
         ("First cleaning pass (sentinel bug)", v1_rows)]
fig, ax = plt.subplots(figsize=(10, 3.4))
labels, values = zip(*steps)
colors = ["#2a78d6"] * 5 + ["#e34948"]
ax.barh(labels, values, color=colors, height=0.6)
for i, v in enumerate(values):
    ax.text(v + 2e4, i, f"{v:,}", va="center", fontsize=9, color=INK)
ax.invert_yaxis(); ax.set_xlim(0, 3.3e6); ax.set_xlabel("CIC-IDS2017 rows (millions)")
ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1e6:.1f}")
ax.set_title("The sentinel bug, not cleaning itself, removed half the data", loc="left", fontweight="bold", color=INK)
style(ax); fig.tight_layout(); plt.show()"""),
        code("""keys = ["rows_raw", "flow_bytes_nan_filled", "rows_dropped_inf_rates", "sentinel_flows_kept",
        "rows_dropped_negative", "rows_dropped_duplicates", "rows_dropped_contradictory_labels", "rows_clean"]
pd.DataFrame({"2017 training copy": [m17[k] for k in keys], "2017 evaluation copy": [m17e[k] for k in keys],
              "2018 training copy": [m18[k] for k in keys], "2018 evaluation copy": [m18e[k] for k in keys]},
             index=keys)"""),
        md(TEXT["00_body"]),
    ])


def nb02():
    notebook("02_why_random_splits_mislead", [
        md(TEXT["02_intro"]),
        code(SETUP + """
gen = json.load(open(f"{REPORTS}/generalization_evaluation.json"))
aud = json.load(open(f"{REPORTS}/leakage_audit.json"))
rows = []
def add(label, kind, reps):
    ap = [r["test_average_precision"] for r in reps]
    rows.append({"protocol": label, "kind": kind, "AP min": min(ap), "AP max": max(ap),
                 "prevalence": reps[0]["test_prevalence"],
                 "recall at 1% (median)": float(np.median([r["test_at_frozen_thresholds"]["0.01"]["recall"] for r in reps])),
                 "actual FPR at 1% (median)": float(np.median([r["test_at_frozen_thresholds"]["0.01"]["fpr"] for r in reps]))})
add("Random rows", "same capture", [r["candidates"]["lgbm71"] for r in gen["runs"] if r["split"] == "random_split"])
for proto, label, kind in [("grouped_quarter_octave", "Grouped similar flows", "same capture"),
                           ("purged_file_blocks", "End of each file, with gaps", "same capture"),
                           ("test_friday", "Hold out Friday", "new day"),
                           ("forward_thursday", "Train Mon-Tue, test Thu", "new day"),
                           ("test_wed_thu", "Hold out Wed+Thu", "new day")]:
    add(label, kind, [r["candidates"]["lgbm71"] for r in aud["runs"] if r["protocol"] == proto])
protocols = pd.DataFrame(rows)
same, new = protocols[protocols.kind == "same capture"], protocols[protocols.kind == "new day"]
print(f"Finding: when test flows come from the same capture as training, LightGBM scores AP "
      f"{same['AP min'].min():.4f}-{same['AP max'].max():.4f} even after grouping similar flows together. "
      f"When the test is a new day, AP falls to {new['AP min'].min():.2f}-{new['AP max'].max():.2f}. "
      f"The near-perfect number answers 'does it recognize this capture's attacks?', not 'can it detect new ones?'")

fig, ax = plt.subplots(figsize=(10, 3.8))
for i, r in protocols.iterrows():
    color = "#2a78d6" if r.kind == "same capture" else "#eb6834"
    ax.plot([r["AP min"], r["AP max"]], [i, i], color=color, linewidth=3, solid_capstyle="round")
    ax.plot(r["AP max"], i, "o", color=color, markersize=7, markeredgecolor="white")
    ax.plot(r["prevalence"], i, "|", color=MUTED, markersize=14, markeredgewidth=2)
    ax.text(r["AP max"] + 0.025, i, f"{r['AP min']:.4f}-{r['AP max']:.4f}", va="center", fontsize=8.5, color=INK)
ax.set_yticks(range(len(protocols))); ax.set_yticklabels(protocols.protocol); ax.invert_yaxis()
ax.set_xlim(0, 1.25); ax.set_xlabel("Average precision (range over 3 seeds); grey tick = no-skill baseline (attack share)")
ax.set_title("Same-capture tests are near perfect; new-day tests are not", loc="left", fontweight="bold", color=INK)
ax.plot([], [], color="#2a78d6", linewidth=3, label="test flows from the training capture")
ax.plot([], [], color="#eb6834", linewidth=3, label="test is a later or held-out day")
ax.legend(frameon=False, fontsize=8.5, loc="lower right"); style(ax); fig.tight_layout(); plt.show()"""),
        code("""protocols.round(4)"""),
        md(TEXT["02_gates"]),
        code("""legacy = aud["legacy_audit"]["random_split_seed_42"]
shuffle = aud["shuffled_training_labels"]
summary = load_summary()
fold_gates = pd.DataFrame({f: {"exact twins (share of test rows)": g["gates"]["exact_twin_test_fraction"]["max"],
                               "exact twins that are attacks": g["gates"]["exact_twin_attack_rows"]["max"],
                               "shuffled-label AP (max)": g["gates"]["shuffled_label_ap"]["max"],
                               "prevalence": g["prevalence"],
                               "best single feature AP (max)": g["gates"]["best_single_feature_ap"]["max"],
                               "LightGBM AP (median)": (g["detectors"]["lgbm"]["average_precision"]["median"]
                                                         if "lgbm" in g["detectors"] else None)}
                           for f, g in summary["folds"].items()}).T
print(f"Legacy random split: {legacy['exact_input_overlap_n']:,} test rows had an exact input twin in training "
      f"({legacy['exact_input_overlap_fraction']:.2%}); {legacy['attack_exact_overlap_n']} were attacks.")
print(f"Shuffled-label control (grouped split): AP {shuffle['test_average_precision']:.4f} vs prevalence {shuffle['test_prevalence']:.4f}.")
fold_gates.round(4)"""),
        md(TEXT["02_body"]),
    ])


def results_notebook(name, folds, labels, title):
    return [
        md(TEXT[f"{name}_intro"]),
        code(f"""FOLDS = {folds!r}
LABELS = {labels!r}
fig = recall_dots(summary, FOLDS, {title!r}, LABELS)
plt.show()
recall_table(summary, FOLDS)"""),
        code("""# Recall per attack family at the 1% budget (median over seeds, n = flows in the test set)
rows = {}
for f in FOLDS:
    fold = summary["folds"][f]
    for det, d in fold["detectors"].items():
        for fam, r in d["per_family_recall_at_1pct"].items():
            rows.setdefault((LABELS[f], fam), {})[NAMES[det]] = round(r["median"], 3)
pd.DataFrame(rows).T"""),
    ]


def nb03():
    cells = results_notebook("03", ["2017_A", "2017_B", "2017_C"],
                             {"2017_A": "A: train Mon (benign), test Wed", "2017_B": "B: train Mon-Tue, test Thu",
                              "2017_C": "C: train Mon-Wed, test Fri"},
                             "Forward in time on CIC-IDS2017: recall on the next day's new attacks")
    cells.insert(1, code(SETUP + "summary = load_summary()\n" + TEXT["03_finding_code"]))
    cells += [md(TEXT["03_explain"]),
              code("""ex = json.load(open(f"{REPORTS}/results/explain_2017_B.json"))
print("Recall per label (LightGBM, seed 42):", {k: round(v, 3) for k, v in ex["per_label_recall"].items()})
print("Window artifact:", ex["window_artifact"])
pd.DataFrame({k: v["per_label_recall"] for k, v in ex["ablations"].items()}).T.round(3)"""),
              md(TEXT["03_body"])]
    notebook("03_forward_2017", cells)


def nb04():
    folds = ["2018_new_tool_dos", "2018_new_tool_ddos", "2018_same_attack_web",
             "2018_same_attack_infiltration", "2018_new_family_bot"]
    labels = {"2018_new_tool_dos": "Same family, new tool: DoS (16 Feb)",
              "2018_new_tool_ddos": "Same family, new tool: DDoS (21 Feb)",
              "2018_same_attack_web": "Same attack, later date: web (23 Feb)",
              "2018_same_attack_infiltration": "Same attack, later date: infiltration (1 Mar)",
              "2018_new_family_bot": "New family: Bot (2 Mar)"}
    cells = results_notebook("04", folds, labels, "Forward in time on CSE-CIC-IDS2018: recall by question type")
    cells.insert(1, code(SETUP + "summary = load_summary()\n" + TEXT["04_finding_code"]))
    cells += [md(TEXT["04_body"])]
    notebook("04_forward_2018", cells)


def nb05():
    notebook("05_high_recall_hybrid", [
        md(TEXT["05_intro"]),
        code(SETUP + "summary = load_summary()\n" + TEXT["05_finding_code"]),
        code("""crit = summary["success_criteria"]
pd.DataFrame({h: {"1. beats best single model": r["criterion_1_beats_best_single"]["pass"],
                  "2. actual FPR <= 1.5%": r["criterion_2_fpr_at_most_1.5pct"]["pass"],
                  "worst actual FPR": round(r["criterion_2_fpr_at_most_1.5pct"]["worst_fpr"], 4),
                  "3. no family lost": r["criterion_3_no_family_lost"]["pass"],
                  "overall": r["overall_pass"]} for h, r in crit.items()}).T"""),
        code("""# Criterion 1, run by run: hybrid recall vs the best single model on the same fold and seed
rows = []
for h, r in crit.items():
    for run in r["criterion_1_beats_best_single"]["runs"]:
        rows.append({"detector": NAMES[h], "fold": run["fold"], "seed": run["seed"],
                     "hybrid recall": run["hybrid"], "best single-model recall": run["best_single"]})
pd.DataFrame(rows).pivot_table(index=["detector", "fold"], values=["hybrid recall", "best single-model recall"],
                               aggfunc="median").round(3)"""),
        code("""# Budget split chosen on validation (share of the 1% budget given to the supervised model), per seed
pd.DataFrame({f: {h: s for h, s in d.get("hybrid_shares_chosen", {}).items()}
              for f, d in summary["folds"].items() if d.get("hybrid_shares_chosen")}).T"""),
        code("""# Hybrid vs the best single model on the criterion folds (median over seeds, 1% budget)
labels = {"2017_B": "2017 B: test Thu", "2017_C": "2017 C: test Fri", "2018_new_family_bot": "2018: new family (Bot)"}
fig, ax = plt.subplots(figsize=(10, 3.6))
y = 0
ticks = []
for h in crit:
    runs = pd.DataFrame(crit[h]["criterion_1_beats_best_single"]["runs"])
    for fold, label in labels.items():
        r = runs[runs.fold == fold]
        hyb, single = r["hybrid"].median(), r["best_single"].median()
        ax.plot([hyb, single], [y, y], color="#e4e3dc", linewidth=3, zorder=1)
        ax.plot(single, y, "o", color=MUTED, markersize=8, zorder=2)
        ax.plot(hyb, y, "o", color=COLORS[h], markersize=8, zorder=3)
        ax.text(max(hyb, single) + 0.02, y, f"hybrid {hyb:.2f} vs best single {single:.2f}", va="center", fontsize=8, color=INK)
        ticks.append((y, f"{NAMES[h].replace('Hybrid: ', '')} | {label}"))
        y += 1
    y += 0.5
ax.set_yticks([t for t, _ in ticks]); ax.set_yticklabels([l for _, l in ticks], fontsize=8.5); ax.invert_yaxis()
ax.set_xlim(0, 1.25); ax.set_xlabel("Recall at a 1% validation false-alarm budget (median of 5 seeds); grey = best single model")
wins = sum(x["pass"] for h in crit for x in crit[h]["criterion_1_beats_best_single"]["runs"])
total = sum(len(crit[h]["criterion_1_beats_best_single"]["runs"]) for h in crit)
ax.set_title("The hybrid never beats the best single model" if wins == 0 else
             f"The hybrid beats the best single model in {wins} of {total} runs", loc="left", fontweight="bold", color=INK)
style(ax); fig.tight_layout(); plt.show()"""),
        md(TEXT["05_body"]),
    ])


def nb06():
    cells = results_notebook("06", ["2017_to_2018"], {"2017_to_2018": "Train all 2017, test all 2018"},
                             "A 2017 model on 2018 traffic")
    cells.insert(1, code(SETUP + "summary = load_summary()\n" + TEXT["06_finding_code"]))
    cells += [
        md(TEXT["06_why"]),
        code("""cols = ["Destination Port", "Total Fwd Packets", "Fwd Header Length", "min_seg_size_forward", "Has_Init_Win_fwd", "Label"]
rows = {}
for year in ["2017", "2018"]:
    d = pd.read_parquet(f"../data/processed/cicids{year}_eval.parquet", columns=cols)
    dns = d[(d["Label"] == "BENIGN") & (d["Destination Port"] == 53) & (d["Total Fwd Packets"] == 1)]
    rows[year] = {"single-packet DNS flows": len(dns),
                  "share with a TCP window": round(dns["Has_Init_Win_fwd"].mean(), 3),
                  "Fwd Header Length (top values)": dns["Fwd Header Length"].value_counts(normalize=True).head(3).round(3).to_dict(),
                  "min_seg_size_forward (top values)": dns["min_seg_size_forward"].value_counts(normalize=True).head(3).round(3).to_dict()}
pd.DataFrame(rows)"""),
        md(TEXT["06_body"]),
    ]
    notebook("06_cross_year", cells)


if __name__ == "__main__":
    from scripts.eda_notebooks import nb01a, nb01b, nb01c
    builders = {"00": nb00, "01a": lambda: nb01a(notebook), "01b": lambda: nb01b(notebook),
                "01c": lambda: nb01c(notebook), "02": nb02, "03": nb03, "04": nb04, "05": nb05, "06": nb06}
    for key in sys.argv[1:] or builders:
        builders[key]()
        print(f"Wrote notebook {key}")
