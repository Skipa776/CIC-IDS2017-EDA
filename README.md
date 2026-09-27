# Intrusion detection on CIC-IDS2017

Why near-perfect random-split scores on CIC-IDS2017 don't mean a model can detect new attacks.

## Key finding

Layer 1 (benign vs. attack, LightGBM) evaluated three ways. Precision and recall are at the
default 0.5 threshold. Source: `models/model_metadata.json`.

| Test | PR-AUC | No-skill PR-AUC | Precision | Recall |
|---|---:|---:|---:|---:|
| Random 80/20 split | 1.000 | 0.169 | 0.982 | 0.999 |
| Friday held out | 0.824 | 0.358 | 0.678 | 0.011 |
| Wed+Thu held out | 0.466 | 0.197 | 0.176 | 0.041 |

The random-split PR-AUC is 0.9997 before rounding.

In CIC-IDS2017 each attack type appears on only one capture day. Holding out days is the same as
holding out attack types. The random split tells us the model recognizes attacks it has already
seen. The held-out days test whether it can flag attacks it has never seen, which is the question
that matters for an IDS. At the default threshold it catches 1.1% and 4.1% of those attack flows.
On Wed+Thu, precision at the default threshold (0.176) is below the attack share of the test set
(0.197).

Lowering the threshold does not fix this. Recall at fixed false positive rates on held-out benign
traffic (`scripts/crossday_threshold_analysis.py`, output in
`reports/crossday_threshold_analysis.json`):

| Test | Recall at 0.1% FPR | at 1% FPR | at 5% FPR |
|---|---:|---:|---:|
| Random 80/20 split | 0.995 | 1.000 | 1.000 |
| Friday held out | 0.002 | 0.138 | 0.595 |
| Wed+Thu held out | 0.000 | 0.000 | 0.150 |

At 1% FPR on the Friday holdout, per-attack recall is DDoS 0.233, PortScan 0.005, and Bot 0.000.
On the Wed+Thu holdout every attack type (DoS variants, web attacks, Infiltration, Heartbleed) is
0.000. PR curves for all three splits: `reports/crossday_pr_curves.png`.

## What I checked, and what I found

- v1 cleaning dropped 51% of rows (2,830,743 raw, 1,388,089 kept) because `Init_Win_bytes = -1` was treated as corrupt. It is a sentinel for "no TCP window observed." v2 keeps these 1,439,672 flows with indicator columns and retains 2,519,262 rows (89.0%).
- Removed 307,070 exact duplicates and 1,394 rows whose identical feature vectors had contradictory labels (`reports/dataset_v2_manifest.json`).
- The split happens before benign downsampling, the test set keeps natural prevalence (16.9% attack), and the scaler is fit on training data only.
- Shuffled-label sanity check: test PR-AUC 0.618 against a no-skill baseline of 0.680, so the pipeline does not leak labels (`reports/eda_baseline_validation_v2.json`, experiment 6).
- Destination Port alone reaches PR-AUC 0.753 against a no-skill of 0.680 and 0.995 for all 71 features. Dropping it leaves 0.994. In v2 the port is not the shortcut it looked like in v1, where it alone reached 0.817 (experiment 4; these experiments use logistic regression on a test set that is 68% attack).
- CIC-IDS2017 has known flow-construction and labeling defects, documented by Engelen, Rimmer & Joosen (2021), "Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study." I did not correct for these, so they bound every number here.

## Models

Two LightGBM classifiers on 20 flow features (`src/features/engineering.py`, `FAST_FEATURES`).
Layer 1 decides benign vs. attack. Layer 2 runs only on flows Layer 1 flags and names the attack
type (14 classes, macro F1 0.910 on the random split). Details: `models/MODEL_CARD.txt`.

## Limitations

- The data is lab-generated traffic from 2017. Scores will not carry over to a real network.
- Labels are noisy (see Engelen et al. above).
- Some classes are tiny: Heartbleed has 11 flows, SQL injection 21, Infiltration 36. Their metrics are anecdotal.
- Flow features describe packet sizes, counts, and timing. They cannot see payloads, so attacks that differ only in content (XSS vs. SQL injection) are hard to separate.

## What I learned

TODO(josh): write this section myself, 4-6 sentences.

## Repo layout

```
api/                  FastAPI service: classify flows, return MITRE ATT&CK mapping
data/processed/       cleaned parquet files (not tracked; built by scripts/build_dataset.py)
models/               MODEL_CARD.txt, model_metadata.json, mitre_mapping.json tracked;
                      trained .joblib files are not tracked (built by scripts/train_models.py)
notebooks/
  cicids2017_eda.ipynb      EDA, logistic regression and Isolation Forest baselines
  attack_types.ipynb        per-attack analysis, random forest multiclass
  archive/day_of_the_weeks.ipynb   unfinished stub, archived (never fully run)
reports/              validation JSONs, dataset manifest, threshold analysis, SOC playbook,
                      notebook review, figures
scripts/
  build_dataset.py          raw CSVs -> data/processed/cicids2017_clean_v2.parquet + manifest
  train_models.py           trains layers 1 and 2, writes models/ and metadata
  crossday_threshold_analysis.py   recall at fixed FPR on held-out days
  validate_eda_baseline.py  7-experiment evaluation-protocol checks
  attack_clustering.py      KMeans clusters of attack flows -> MITRE techniques
  test_overfitting*.py, smoke_test.py   diagnostics
src/                  cleaning, loading, feature, training and evaluation code
tests/                pytest suite for the API, classifier, and MITRE mapping
```

## How to reproduce

Download the CIC-IDS2017 MachineLearningCVE CSVs from the Canadian Institute for Cybersecurity and
put the 8 files in `cic-ids-eda/data/raw/MachineLearningCVE/`. Then:

```sh
pip install -r requirements.txt
python scripts/build_dataset.py                 # -> data/processed/cicids2017_clean_v2.parquet
python scripts/train_models.py                  # -> models/ artifacts and model_metadata.json
python scripts/crossday_threshold_analysis.py   # -> reports/crossday_threshold_analysis.json + PR curves
python scripts/validate_eda_baseline.py --data data/processed/cicids2017_clean_v2.parquet --tag _v2
pytest tests/
```

All seeds are fixed (`random_state=42`).

## Other components

`api/` is a FastAPI service (`uvicorn api.main:app`) that runs both layers on a flow and returns
the attack type with its MITRE ATT&CK technique and mitigations. The mapping is in
`api/services/mitre_mapping.py`, exported to `models/mitre_mapping.json`. `tests/` covers the API
contract, the classifier, and the mapping. `reports/soc_triage_playbook.md` describes how an
analyst should read the alerts, including where not to trust them.

## Data license

CIC-IDS2017 is distributed by the Canadian Institute for Cybersecurity under its own terms. It is
not included in this repository.
