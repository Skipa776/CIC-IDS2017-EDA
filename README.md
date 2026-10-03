# Intrusion detection on CIC-IDS2017

An evaluation of whether flow-based intrusion detection transfers beyond its training environment.

## Key finding

The original row-random evaluation was not an adequate test of independent detection. The audit
found exact model-input overlap, extensive shared behavior profiles, and label-aware cleaning
before the split. Random-split scores are archived for reproducibility and excluded from the
headline performance claims. Reliable transfer across capture days has not been established.

## Leakage audit and corrected evaluation

[`scripts/audit_leakage.py`](scripts/audit_leakage.py) audits the old split and evaluates held-out
traffic from **all eight raw source files**. It starts from feature-only cleaning and keeps
duplicates and contradictory labels; matching profiles are assigned to one partition globally.

Confirmed findings in the original split:

- **Projected-input duplication:** cleaning deduplicated on 71 features, while the served model
  uses 20. There are 42,116 repeated 20-input rows in the cleaned data. Of 503,853 test flows,
  1,854 have an exact input twin in training. Only 47 are attacks, so this alone cannot explain
  the near-perfect ranking.
- **Related behavior across partitions:** 84.6% of attack test flows have a matching training
  profile under fixed, label-blind quarter-octave behavior bins. This demonstrates dependence
  under that proxy, not verified session identity. Exact-row deduplication missed this issue.
- **Label-informed test filtering:** the original cleaner removes contradictory labels using
  the entire dataset before splitting. That filters the evaluation population using test labels
  and removes difficult cases. The new evaluation retains them.
- **Feature-selection contamination:** the importance analysis in `attack_types.ipynb` fits on
  labeled data before making its test split. The 20-feature subset is documented as informed by
  importance analysis, without a training-only selection record. The new primary model uses all
  71 numeric inputs; the existing 20-feature subset is retained only as a diagnostic.

The corrected protocols are defined before examining their scores:

| Protocol | What is held out | Contamination control |
|---|---|---|
| Grouped profiles | 20% of global behavior groups for testing, 20% for validation | Equal profiles stay together, including across different files |
| Finer-profile sensitivity | Same protocol with eighth-octave bins | Checks dependence on the grouping resolution |
| Purged file blocks | Final 18% of raw row positions from every file | Gaps at boundaries; remove held-out profiles from earlier partitions |

Ports, flags and initial TCP windows are exact within a profile; remaining inputs use fixed
logarithmic bins. No labels or fitted statistics define those groups. These are proxies for related
flows: bin boundaries can still separate near neighbors, and file order is not verified time order.
Holding out random rows from each file would continue to share capture and campaign context.
The grouped tests therefore complement whole-day and cross-year tests rather than establish
generalization by themselves.

Models and scalers fit on training only. Thresholds use validation benign traffic and are frozen
before testing. Reports include actual test FPR, recall, precision, per-file/attack counts,
prevalence, zero-overlap assertions, source hashes, seed variation and a shuffled-label control.
The ranking metric is average precision (AP), with test attack prevalence as its no-skill reference;
operational conclusions depend on recall and realized false-positive rate together.

Results: [`reports/leakage_audit.md`](reports/leakage_audit.md), with full
[provenance and measurements](reports/leakage_audit.json).

## Corrected transfer results

The raw-derived evaluation retains 2,827,726 flows, including repeated flows and conflicting
labels previously removed from the v2 parquet. It completes 36 controlled fits plus one shuffled
training-target control: two feature sets, six protocols and three seeds. Shared behavior profiles
are purged across partitions in whole-day tests as well as file-block tests.

| Evaluation | Training days | Validation days | Test days |
|---|---|---|---|
| Friday transfer | Mon–Wed | Thu | Fri |
| Wed+Thu stress test | Mon+Tue | Fri | Wed+Thu |
| Forward Thursday | Mon+Tue | Wed | Thu |

Wed+Thu is explicitly nonchronological. Forward Thursday uses earlier capture days for training
and validation, followed by Thursday testing. Whole-day purging further removes shared profiles;
the tested population is unchanged, while training and validation populations become more restricted.

At the 1% **validation** FPR budget, the primary 71-feature LightGBM gives these ranges over seeds:

| Held-out traffic | AP | Attack prevalence | Actual test FPR | Attack recall |
|---|---:|---:|---:|---:|
| Friday | 0.632–0.680 | 41.10% | 1.99–2.32% | 21.26–45.89% |
| Wed+Thu | 0.221–0.400 | 22.08% | 0.61–0.72% | 0–0.0063% |
| Forward Thursday | 0.0046–0.0051 | 0.48% | 1.04–1.15% | 0.41–0.63% |

Friday retains some ranking signal but exceeds the intended false-alert budget in every seed.
Wed+Thu detects only 0–16 of 253,939 attack flows at the frozen operating point. Thursday ranking
is close to its prevalence baseline and only 9–14 of 2,216 attacks are detected. A moderate AP
does not rescue detection at a tight false-positive budget, and differing prevalence prevents
interpreting the AP gap as a pure model effect.

Grouped and purged-block tests within the same files still rank traffic very well after the
measured overlaps are removed. The corrected controls therefore do not establish that leakage
alone caused every high score. They also do not make shared lab captures representative of future
networks. The day tests above expose the transfer failure hidden by easier same-capture evaluations.

The shuffled-training-target control gives AP 0.2065 against prevalence 0.1985. This is a negative
control for learning from corrupted targets, not proof that target-derived features cannot exist.
Labels, file names and raw-row positions are excluded from the numeric model inputs.

Full results, per-file/attack counts and provenance: [reports/leakage_audit.md](reports/leakage_audit.md)
and [reports/leakage_audit.json](reports/leakage_audit.json). Earlier v2 comparisons are archived in
[reports/generalization_evaluation.md](reports/generalization_evaluation.md), with their population
differences documented in [reports/evaluation_audit.md](reports/evaluation_audit.md).
Seed ranges are not population confidence intervals. These explored captures cannot certify a
generalized detector; a final confirmation needs independent captures and agreed recall/FPR targets.

## Data controls and external transfer evidence

- v1 cleaning dropped 51% of rows (2,830,743 raw, 1,388,089 kept) because `Init_Win_bytes = -1` was treated as corrupt. It is a sentinel for "no TCP window observed." v2 keeps these 1,439,672 flows with indicator columns and retains 2,519,262 rows (89.0%).
- The archival v2 export removed 307,070 duplicates and 1,394 contradictory-label rows (`reports/dataset_v2_manifest.json`). The primary audit retains those examples and separates related profiles instead.
- Only training benign flows are downsampled. Test prevalence is observed within its held-out traffic, not an estimate of deployment prevalence. Scalers fit on training only.
- Earlier destination-port ablation showed feature redundancy: predictive value alone does not establish dependence or transferability. Those exploratory comparisons are archived in `reports/eda_baseline_validation_v2.json`.
- Cross-year transfer is also weak. At a **retrospective** 1% test-FPR budget on CSE-CIC-IDS2018, the 20-feature LightGBM detects 0.4% of attack flows and Isolation Forest 0.6%; a 71-feature LightGBM detects 38%. Diagnostics identify differences in header-length measurement, network traffic and class mix that may contribute to the gap. Tested corrections did not improve both transfer directions, and combining supervised and anomaly detectors did not earn an improvement under validation-based selection. Details: [`reports/cross_dataset_2018.md`](reports/cross_dataset_2018.md) and the three cross-year notebooks.
- CIC-IDS2017 has flow-construction and labeling defects documented by Engelen, Rimmer & Joosen (2021), "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study." Those underlying defects are outside the cleaning corrections here and limit interpretation of every result.

## Models

Two LightGBM classifiers on 20 flow features (`src/features/engineering.py`, `FAST_FEATURES`).
Layer 1 decides benign vs. attack. Layer 2 runs only on flows Layer 1 flags and names the attack
type (14 classes). Its random-split macro F1 of 0.910 is evaluated on **all true attack flows**,
including flows Layer 1 would miss. It is not an end-to-end score for the gated service.
Details: [`models/MODEL_CARD.txt`](models/MODEL_CARD.txt).

## Limitations

- The data is lab-generated traffic from 2017. Random-split performance does not carry over to the same lab's 2018 network; partial transfer of the 71-feature model does not establish deployment reliability.
- Labels are noisy (see Engelen et al. above).
- Some classes are tiny: Heartbleed has 11 flows, SQL injection 21, Infiltration 36. Their metrics are anecdotal.
- Flow features describe packet sizes, counts, and timing. They cannot see payloads, so attacks that differ only in content (XSS vs. SQL injection) are hard to separate.

## Methodological decisions and remaining evidence gaps

**Cleaning is part of the statistical design.** Keeping TCP-window sentinel rows avoids selecting
traffic according to protocol artifacts. Label-aware conflict removal changes the test population;
the primary audit instead retains contradictory labels and keeps duplicate/related profiles together.

**Feature importance is not transferability.** A predictive port or TCP-window value can describe
the services or systems in one capture environment. Controlled ablation tests whether removing that
information helps; it cannot assume the information is always harmful. Earlier cross-year experiments
found that the same feature removal helped in one direction and hurt in the other.

**Alert thresholds require their own validation.** A threshold meeting a 1% benign FPR on one day
can exceed that budget on another. Reporting test recall without its realized FPR hides this failure.
Recalibration on target-network benign traffic is a separate adaptation experiment, not evidence
that a frozen detector transfers unaided.

**The metadata limits the conclusions.** The cleaned MachineLearningCVE export retains capture-file
provenance, but no session IDs or timestamps. It supports file/day separation, not a verified
session-independent split or within-day chronological blocks. With only five capture days and
attack labels tied to individual days, flow-level confidence intervals would overstate the number
of independent observations. Per-day results, family counts and seed ranges summarize the available evidence.

**A generalization claim needs a fresh test.** The 1% FPR budget is a research comparison point,
not an agreed SOC operating requirement. Before promoting a detector, specify a minimum recall,
acceptable false-alert rate and important attack families; select on development captures, then
evaluate on independent future captures with repeated known families and separately identified
unseen families. Existing 2018 results add transfer evidence, but also change network, tools and
label mix. No model is promoted on the strength of a random-split score.

## Repo layout

```
api/                  FastAPI service: classify flows, return MITRE ATT&CK mapping
data/processed/       cleaned parquet files (not tracked; built by scripts/build_dataset.py)
models/               MODEL_CARD.txt, model_metadata.json, mitre_mapping.json tracked;
                      trained .joblib files are not tracked (built by scripts/train_models.py)
notebooks/
  cicids2017_eda.ipynb      EDA, logistic regression and Isolation Forest baselines
  attack_types.ipynb        per-attack analysis, random forest multiclass
  single_day_eda.ipynb      train on one day, test on another: logistic regression vs Isolation Forest
  cross_year_diagnostics.ipynb   why 2017 models fail on 2018 (format, cleaning, drift, flow tool)
  train_2017_test_2018.ipynb     train on 2017, test on 2018, with fixes for the drift
  train_2018_test_2017.ipynb     the reverse direction
  archive/day_of_the_weeks.ipynb   unfinished stub, archived (never fully run)
reports/              validation JSONs, dataset manifest, threshold analysis, SOC playbook,
                      notebook review, figures
scripts/
  build_dataset.py          raw CSVs -> cleaned parquet + manifest (--year 2018 for CSE-CIC-IDS2018)
  train_models.py           trains layers 1 and 2, writes models/ and metadata
  crossday_threshold_analysis.py   recall at fixed FPR on held-out days
  evaluate_generalization.py      controlled models, frozen thresholds, day/forward tests
  audit_leakage.py                raw-file grouped/purged holdouts, overlap and negative controls
  combined_model.py         supervised + Isolation Forest, tested cross-day and on CSE-CIC-IDS2018
  validate_eda_baseline.py  7-experiment evaluation-protocol checks
  attack_clustering.py      KMeans clusters of attack flows -> MITRE techniques
  test_overfitting*.py, smoke_test.py   diagnostics
src/                  cleaning, loading, feature, training and evaluation code
tests/                pytest suite for the API, classifier, and MITRE mapping
```

## How to reproduce

Download the CIC-IDS2017 MachineLearningCVE CSVs from the Canadian Institute for Cybersecurity and
put the 8 files in `cic-ids-eda/data/raw/MachineLearningCVE/`. For the 2018 test, download the
CSE-CIC-IDS2018 "Processed Traffic Data for ML Algorithms" CSVs (public S3 bucket `cse-cic-ids2018`,
about 6.5 GB) into `data/raw/CSE-CIC-IDS2018/`. Then:

```sh
pip install -r requirements.txt
python scripts/build_dataset.py                 # -> data/processed/cicids2017_clean_v2.parquet
python scripts/audit_leakage.py                  # primary evaluation from all eight raw files; retains conflicts
python scripts/evaluate_generalization.py --splits test_friday test_wed_thu forward_thursday --models lgbm20 lgbm71 logreg20 logreg71 lgbm71_no_network
python scripts/train_models.py                  # legacy API artifact reproduction, not independent validation
python scripts/crossday_threshold_analysis.py   # retrospective diagnostic using test-selected thresholds
python scripts/build_dataset.py --year 2018     # -> data/processed/cicids2018_clean.parquet (needs the 2018 CSVs)
python scripts/combined_model.py                # -> reports/combined_model.json
python scripts/validate_eda_baseline.py --data data/processed/cicids2017_clean_v2.parquet --tag _v2
pytest tests/
```

Original experiments use seed 42. The primary leakage audit and day evaluation use seeds 42, 43 and 44
and record library versions in their reports. Evaluation candidates do not replace the API model files.

## Other components

`api/` is a FastAPI service (`uvicorn api.main:app`) that runs both layers on a flow and returns
the attack type with its MITRE ATT&CK technique and mitigations. The mapping is in
`api/services/mitre_mapping.py`, exported to `models/mitre_mapping.json`. `tests/` covers the API
contract, the classifier, and the mapping. `reports/soc_triage_playbook.md` describes how an
analyst should read the alerts, including where not to trust them.

## Data license

CIC-IDS2017 is distributed by the Canadian Institute for Cybersecurity under its own terms. It is
not included in this repository.
