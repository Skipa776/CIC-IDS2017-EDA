"""Narrative cells for scripts/build_notebooks.py.

Written after reading the results files. Every number in a markdown cell here
must also be printed by a code cell in the same notebook, so a stale number is
visible next to the live one.
"""

TEXT = {}

TEXT["00_intro"] = """# 00 · Data and cleaning

**Question.** What is in the raw data, and what did cleaning keep and why?

The finding, its figure and the numbers come first; how they were produced follows."""

TEXT["00_body"] = """## What the cleaning does

The same code (`src/data/cleaning.py`) cleans both datasets. It strips column names, fixes the
garbled web-attack labels, drops 8 constant columns and one duplicate column, fills genuine
missing rates with 0, and drops rows whose rate columns are infinite or whose counts are negative
where negatives are impossible. Inter-arrival times can be negative in this data and are kept.

**The sentinel bug.** `Init_Win_bytes_forward/backward = -1` means "no TCP window was observed"
(UDP flows, flows without a handshake). The first cleaning pass treated it as corrupt and deleted
those rows: half the dataset, mostly ordinary benign traffic. The fix keeps them, adds
`Has_Init_Win_fwd/bwd` indicator columns, and sets the -1 to 0.

**Two copies of each dataset.**

- *Training copy* (`*_clean*.parquet`): exact duplicates and contradictory labels removed. Used by
  the older reports and the served model.
- *Evaluation copy* (`*_eval.parquet`, `python scripts/build_dataset.py --year YYYY --no-dedup`):
  keeps repeated flows and conflicting labels, because real traffic has both, and because removing
  contradictory labels across the whole dataset uses test labels before any split. All numbered
  notebooks 03-06 use the evaluation copy.

**CSE-CIC-IDS2018.** Its columns are renamed onto the 2017 names (`src/data/cicids2018.py`; all
78 columns map one to one). To fit in memory, 10% of benign flows are kept and every attack flow;
prevalence-dependent metrics weight benign rows by 10.

## Limits

- The first-pass row count comes from the old parquet file, which is kept only for this comparison.
- Cleaning cannot fix label errors in the source data (Engelen, Rimmer & Joosen, 2021)."""

TEXT["01_intro"] = """# 01 · Exploratory analysis

**Question.** Which attacks happen on which capture day, and how do the days differ?

This decides what any split can measure: a split that holds out a day also holds out whatever
attacks happened that day."""

TEXT["01_body"] = """## What this means for evaluation

- **CIC-IDS2017:** each attack family sits on a single day, and Monday has no attacks at all.
  Training on earlier days and testing on a later day therefore always tests attack types the
  model has never seen. 2017 can measure unseen-attack detection, but not "the same attack later".
- **CSE-CIC-IDS2018:** some families repeat (DoS, DDoS, web attacks, infiltration). That lets
  `04_forward_2018` separate three questions: the same attack on a later date, the same family with a
  new tool, and a brand-new family.
- **Imbalance differs by day.** The attack share of a test day sets the no-skill baseline for
  average precision, so average precision is never compared across days without its prevalence.

## Limits

The 2018 benign counts are a 10% sample; the attack counts are complete. The heatmap uses a log
color scale, so small classes stay visible; read the printed counts, not the shade."""

TEXT["02_intro"] = """# 02 · Why random splits score near 1.0

**Question.** Why does a random train/test split give near-perfect scores, and do the
leakage checks pass?

A score of 1.0 is not impossible: it means every test attack outranked every benign flow. It is a
red flag that has to be explained before it is shown."""

TEXT["02_gates"] = """## The leakage gates

No number reaches the README until these hold. The table below the printout shows them for every
fold of the forward-in-time evaluation (notebooks 03-06).

| Gate | Pass rule |
| --- | --- |
| No exact twins | Report test rows whose model inputs exactly match a training row. Across days some twins are legitimate (identical benign DNS lookups, repeated flood packets); inside one capture they are leakage. |
| Feature choice inside training data | All 71 features are used; no selection step. The old 20-feature subset was picked using importance on all data, so it is not used here. |
| Shuffled-label control | A model trained on shuffled labels scores near the prevalence. |
| Shortcut check | Flag any fold where one feature alone comes within 0.05 average precision of the full model. |
| Thresholds from validation only | No threshold or model choice uses test labels; the actual test false positive rate is reported. |"""

TEXT["02_body"] = """## Interpretation

The near-perfect score is mostly not a code bug. Exact input twins between the old random split's
train and test sets were few (see the printout), the shuffled-label control sits at chance, and
grouping similar flows so they cannot straddle the split still leaves the score near 1.0.

The explanation is that the question is easy. Inside one lab capture, scripted attacks from a few
attacker machines are trivially separable from benign traffic. A random split asks "does the model
recognize attacks it has already seen, from the same capture?" That is a sanity check, not a
result. The forward-in-time notebooks (03-06) ask the question that matters for an intrusion
detector: can it flag the attacks of a day it has not seen?

## Limits

- Grouping similar flows uses coarse bins of 20 features; it is a proxy, not real session IDs.
- The random-row result uses the deduplicated training copy (`generalization_evaluation.json`);
  the other rows use the raw-derived population (`leakage_audit.json`).
- All of these captures have been looked at before, so none is a fresh, untouched test."""
