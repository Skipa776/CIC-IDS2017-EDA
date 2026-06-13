# Review: `notebooks/cicids2017_eda.ipynb`

**Scope:** data representation, analysis workflow, and whether the reported Logistic Regression
PR-AUC of 0.996 is plausible or an artifact of overfitting/leakage.

**Evidence:** static review of all 65 cells, plus seven empirical experiments run by
`scripts/validate_eda_baseline.py`. Every number below traces to
`reports/eda_baseline_validation.json` (generated 2026-06-12).

---

## 1. Verdict

**The 0.996 PR-AUC is real and reproducible — it is not classic overfitting — but it is an
optimistic, protocol-specific number that should not be read as deployable detection skill.**

- It reproduces exactly (0.9956), and the train-set PR-AUC equals the test-set PR-AUC
  (0.9956 vs 0.9956): there is **no train/test gap**, so the linear model is not memorizing noise.
- Deduplicating the data barely moves it (0.995), so duplicate-row leakage is not the driver either.
- What *does* drive it: (a) the random 80/20 split lets the model see flows from every attack
  session it is later tested on; under an honest **cross-day split the score falls to 0.76–0.82**;
  (b) benign downsampling makes attacks **68.5% of the test set**, so the no-skill PR-AUC baseline
  is already 0.685; (c) the binary score is carried by three voluminous, trivially separable
  classes (DoS Hulk, PortScan, DDoS = 93% of attack rows, each detected at ~99%), while the model
  is **blind to web attacks (0% recall), SSH-Patator (1%), Infiltration (17%), and Bot (29%)**.

In short: plausible for this protocol (published random-split CICIDS2017 results routinely exceed
0.99 even with simple models), but the protocol answers an easy question. The deployment-relevant
question — "does it flag attack behavior it hasn't seen?" — gets 0.76–0.82, with a long tail of
missed attack families.

---

## 2. The 0.996 question — experiment results

| # | Experiment | Test PR-AUC | No-skill baseline | Interpretation |
|---|------------|------------:|------------------:|----------------|
| 1 | Notebook protocol (random split, benign↓200k) | **0.996** | 0.685 | Reproduces the notebook. Train PR-AUC = 0.996 too → no overfitting gap. |
| 2 | Same, after exact-row deduplication | 0.995 | 0.644 | 15.7% of test rows have an identical twin in train, yet removing them changes nothing — the classes are separable, not memorized. |
| 3a | Cross-day: train Mon–Thu, test Friday | **0.818** | 0.578 | Tested on unseen days/attacks (Bot, PortScan, DDoS). −0.18 vs baseline. |
| 3b | Cross-day: train Mon+Tue+Fri, test Wed+Thu | **0.761** | 0.329 | Unseen DoS family, web attacks, Infiltration, Heartbleed. −0.24 vs baseline. |
| 4a | Without `Destination Port` | 0.996 | 0.685 | Port is *not* the shortcut — fully redundant with other features. |
| 4b | `Destination Port` alone | 0.817 | 0.685 | A single feature gets most of the way to the headline number — a measure of how easy the task is. |
| 5 | Test at natural prevalence (31.3% attack) | 0.981 | 0.313 | Prevalence inversion inflates the headline modestly (0.996 → 0.981 at realistic prevalence; the *baseline* drops from 0.685 to 0.313). |
| 6 | Shuffled training labels (sanity) | 0.557 | 0.685 | Collapses to no-skill territory → the pipeline itself does not leak labels. |

**Per-class recall under the notebook's own protocol (experiment 7):**

| Attack | n (test) | Recall | | Attack | n (test) | Recall |
|---|---:|---:|---|---|---:|---:|
| DDoS | 16,316 | 0.994 | | FTP-Patator | 1,303 | 0.620 |
| DoS Hulk | 32,907 | 0.994 | | Bot | 401 | 0.287 |
| PortScan | 31,646 | 0.989 | | Infiltration | 6 | 0.167 |
| DoS GoldenEye | 1,547 | 0.912 | | SSH-Patator | 1,142 | **0.010** |
| DoS Slowhttptest | 466 | 0.867 | | Web Attack – Brute Force | 298 | **0.000** |
| DoS slowloris | 833 | 0.779 | | Web Attack – XSS | 114 | **0.000** |
| Heartbleed | 3 | 1.000 | | Web Attack – SQL Injection | 6 | **0.000** |

BENIGN false-positive rate: 5.4%. The aggregate 0.996 is a weighted average dominated by the three
big classes; five attack families are effectively invisible to the model.

Figures: `reports/figures/validation_pr_curves.png` (PR curves per protocol),
`reports/figures/validation_per_class_recall.png`.

**Caveat on the cross-day result:** in CICIDS2017 each attack type occurs on exactly one day, so a
cross-day split is also a cross-attack-type split — the 0.76–0.82 measures generalization to *new
attack behavior*, not merely new time periods. That is the right question for an IDS, but it cannot
be separated from "new day" with this dataset.

---

## 3. Data representation review

### R1 — The cleaning step silently discards 51% of the dataset (cell 17) — **highest-impact issue**
Cell 3 loads 2,830,743 rows; the "drop rows with any negative value in non-IAT numeric columns"
filter leaves 1,388,089. The dominant cause is `Init_Win_bytes_forward/backward = -1`, which is a
**sentinel meaning "no TCP window observed"** (UDP flows, flows without a handshake), not corrupt
data. BENIGN falls from 2.27M to 953k rows. Consequences:

- The saved parquet under-represents benign traffic by ~58% and systematically excludes non-TCP
  and handshake-less flows — the model never learns what that (very common) benign traffic looks like.
- Class proportions in every EDA chart and both baselines reflect this filtered population, not
  the dataset. E.g., DDoS loses 36% of its rows (128k → 81.5k).

**Fix:** keep the rows; encode the sentinel explicitly (e.g., `has_init_win_fwd` flag + clamp -1 to 0),
and restrict the negative-value filter to columns where negatives are genuinely impossible.

### R2 — No deduplication
6.0% of rows (83,590) are exact feature duplicates, and 567 identical feature vectors carry
*contradictory* labels. Experiment 2 shows this doesn't inflate the headline number, but the
contradictory labels put a ceiling on achievable precision and should be resolved or dropped.

### R3 — Inf/NaN handling works by coincidence (cells 12, 16)
Cell 16 converts ±inf → NaN in *all* columns but drops NaN only via `subset=['Flow Bytes/s']`.
It happens that `Flow Packets/s` is inf in exactly the same zero-duration rows, so the saved
parquet is verifiably clean (audited: zero NaN/inf) — but the code only guarantees this for one
column. Make the dropna subset explicit over both rate columns, or assert cleanliness before saving.

### R4 — Known CICIDS2017 defects are not acknowledged
The literature (Engelen, Rimmer & Joosen 2021, "Troubleshooting an Intrusion Detection Dataset";
Lanvin et al. 2023) documents flow-construction bugs, mislabeled flows, and duplicates in the
MachineLearningCVE CSVs. None of this invalidates an EDA exercise, but a notebook positioning
models as baselines should state that label noise bounds achievable metrics. Related cosmetic
issue: web-attack labels contain a mojibake character (`Web Attack � Brute Force`) from a non-UTF-8
en dash in the source CSVs — worth normalizing during cleaning.

### EDA quality — what's good
Per-file loading with `Meta_source` provenance (cell 3) is genuinely good design — it is what made
the cross-day validation possible. The class-imbalance treatment (frequency table + relative-frequency
plot, cells 22/37), log-scaled boxplots (27/30), stratified 2k-per-label sampling for PCA/UMAP
(cell 36), and written interpretation cells after each figure all reflect a sound EDA habit.

### EDA quality — weaknesses
- **Duplicated work:** cells 27 and 39 produce the same boxplot; cells 22 and 37 the same frequency table.
- **Double standardization bug:** cell 41 scales the sample, cell 43 scales it *again* before PCA.
  Harmless numerically (re-scaling standardized data), but it signals copy-paste drift.
- The 69×69 correlation heatmap (cell 33) is unreadable — no clustering, no feature-family ordering,
  and its interpretation cell can't reference anything specific.
- The z-score outlier scan (cell 51) is computed, printed, and never used; z-scores are also a poor
  outlier criterion for heavy-tailed flow features. Either act on it or drop it.

---

## 4. Workflow review

### Done right (worth keeping)
- Column-name stripping, constant-column and duplicate-column removal (cells 9, 13–15).
- Stratified train/test split; `StandardScaler` fit on train only (cell 54).
- `class_weight='balanced'`; precision/recall/PR-AUC instead of accuracy (cells 56, 60) — the right
  metric family for 68%/32% imbalance.
- 5-fold CV with fold-level reporting; fixed seeds throughout; figures saved to `reports/figures/`.
- Isolation Forest trained on benign-only training data (cell 58) — correct protocol for
  unsupervised anomaly detection, and the honest reporting of its low recall (0.40) is a credit.

### Issues, in priority order
- **W1 — Random split is the wrong evaluation for IDS (cell 54).** Flows from the same attack burst
  land on both sides of the split. The notebook has `Meta_source` sitting in the dataframe — a
  grouped/temporal split was one line away. This is the single change that would have surfaced the
  0.76–0.82 number.
- **W2 — Prevalence inversion (cell 54).** Downsampling benign *before* the split makes the test set
  68.5% attack. That inflates the PR-AUC baseline to 0.685 and makes the headline incomparable to any
  realistic deployment. Downsample the training set only; leave the test set at natural prevalence
  (experiment 5 shows what this looks like: 0.981 vs no-skill 0.313).
- **W3 — Binary metrics hide per-class failure (cells 56/60).** One `classification_report` grouped by
  original `Label` (as in experiment 7) would have revealed the 0% recall on web attacks immediately.
- **W4 — Scaler fit before CV (cells 54/56).** The scaler is fit on all of `X_train` and the CV folds
  reuse it, so each validation fold's statistics leak into scaling. Minor here (CV matches test), but
  the idiomatic fix is `sklearn.pipeline.Pipeline(StandardScaler(), LogisticRegression())` inside the CV loop.
- **W5 — A single model instance is refit across folds (cell 56).** Works because `.fit()` resets
  state, but instantiating inside the loop (or using `cross_val_score` with a Pipeline) is cleaner.
- **W6 — `Destination Port` as a raw numeric feature.** Treating port numbers as a continuous
  magnitude is semantically wrong (port 8080 is not "16× port 443"). Experiments 4a/4b show it is
  redundant here, but for the planned multi-class model it should be dropped or encoded
  (e.g., well-known-port buckets) — alone it already achieves 0.817 PR-AUC, which is shortcut risk.

---

## 5. Prioritized recommendations

1. **Re-do cleaning without the sentinel massacre (R1):** keep `Init_Win_bytes = -1` rows via an
   indicator feature; re-generate the parquet. This changes the population every downstream model sees.
2. **Adopt cross-day evaluation as the headline metric (W1):** report random-split and cross-day
   numbers side by side (0.996 / 0.76–0.82). The gap *is* the finding.
3. **Evaluate at natural prevalence (W2)** and always print the no-skill PR-AUC baseline next to the score.
4. **Add per-attack-type recall to every binary evaluation (W3)** — reuse experiment 7 in
   `scripts/validate_eda_baseline.py`.
5. **Deduplicate (features+label) and drop contradictory-label rows (R2)** before splitting.
6. **Wrap scaling + model in a `Pipeline` for CV (W4/W5).**
7. **Tidy the EDA:** remove the duplicated boxplot/frequency cells, fix the double scaling,
   cluster the correlation heatmap, normalize the mojibake labels.

## 6. Post-fix results (dataset v2, 2026-06-12)

The recommendations above were implemented: cleaning v2 (`src/data/cleaning.py` +
`scripts/build_dataset.py`) keeps the `Init_Win_bytes = -1` sentinel rows behind indicator
columns, deduplicates, removes contradictory labels, and normalizes the mojibake labels —
**retaining 89.0% of the raw data (2,519,262 rows) vs 49% in v1**. The notebook was reworked to
split at natural prevalence (benign downsampled in training only), use a `Pipeline` for CV, and
report cross-day and per-class results. Numbers from `reports/eda_baseline_validation_v2.json`:

| Protocol | v1 PR-AUC | v2 PR-AUC | v2 no-skill |
|---|---:|---:|---:|
| Random split, benign↓ (old headline) | 0.996 | 0.995 | 0.680 |
| Natural prevalence (new headline) | 0.981 | **0.954** | 0.169 |
| Cross-day: test Friday | 0.818 | 0.799 | 0.358 |
| Cross-day: test Wed+Thu | 0.761 | 0.711 | 0.197 |
| Duplicate twins in test | 15.7% | **0.0%** | — |

Notable per-class changes (natural-prevalence test): **SSH-Patator recall 0.01 → 0.91** and
FTP-Patator 0.62 → 0.69 — the sentinel rows v1 deleted carried the brute-force signal. Web
attacks (0–3%) and Bot (~28%) remain blind spots; DoS/DDoS/PortScan stay at 0.9+. The Isolation
Forest's honest score at natural prevalence is PR-AUC ≈ 0.54 (its v1 0.859 was largely
prevalence inflation).

One implementation note: during the rework we found that
`select_dtypes(include=["int64","float64"])` silently matched only 24 of 71 numeric columns on
in-memory frames mid-pipeline (a pandas block-state quirk; parquet round-trips mask it). All
selections now use `include=[np.number]`, and the notebook asserts the feature count.

## 7. Reproducing this review

```bash
python scripts/validate_eda_baseline.py
# → reports/eda_baseline_validation.json
# → reports/figures/validation_pr_curves.png
# → reports/figures/validation_per_class_recall.png
```

Runtime ≈ 10–15 min on an M3 Pro (8 LogReg fits on up to 1.1M × 69 data).
