# CLAUDE.md - Project Context

This file provides comprehensive context to Claude Code for this project. See `PLAN.md` for the implementation plan.

---

## Project Overview

**Purpose:** Build a production-ready intrusion detection system on the CICIDS2017 dataset from the Canadian Institute for Cybersecurity.

**End Goal:** REST API that classifies network flows, identifies attack types, and maps to MITRE ATT&CK with mitigations.

---

## Directory Structure

```
cicids-eda/
├── CLAUDE.md                    # This file - project context
├── README.md                    # User documentation
│
├── data/processed/
│   ├── cicids2017_clean_v2.parquet    # 2.52M rows, 73 cols — USE THIS (v2 cleaning)
│   ├── cicids2017_clean.parquet       # v1 (1.39M rows) — kept for traceability only
│   └── cicids2017_clean.csv           # v1 CSV
│
├── cic-ids-eda/
│   └── data/raw/MachineLearningCVE/   # RAW CSVs ARE HERE (8 files)
│
├── notebooks/
│   ├── cicids2017_eda.ipynb           # COMPLETE - EDA + LogReg/IsoForest baselines (v2 protocol)
│   ├── attack_types.ipynb             # COMPLETE - attack analysis + RF multiclass (Polars)
│   └── day_of_the_weeks.ipynb         # STUB - needs implementation
│
├── src/
│   ├── data/cleaning.py               # v2 cleaning pipeline (shared with build_dataset.py)
│   ├── data/loader.py                 # parquet loading + label helpers
│   ├── features/engineering.py        # FAST_FEATURES (20) + scaler helpers
│   └── models/train.py, evaluate.py   # LGBM training/eval/save utilities
│
├── scripts/
│   ├── build_dataset.py               # raw CSVs -> cicids2017_clean_v2.parquet + manifest
│   ├── train_models.py                # trains/saves production models (cross-day metrics incl.)
│   ├── validate_eda_baseline.py       # 7-experiment evaluation-protocol validation suite
│   ├── attack_clustering.py           # KMeans attack clusters -> MITRE techniques
│   └── test_overfitting*.py           # RF overfitting diagnostics
│
├── api/                               # FastAPI service (classify + MITRE mapping)
├── models/                            # trained LGBM layer1/layer2 + scaler + metadata
├── tests/                             # pytest suite (50 tests)
└── reports/                           # review docs, validation JSONs, figures, playbook
```

---

## Dataset Details

### Source
CICIDS2017 from Canadian Institute for Cybersecurity - network flow data with labeled attacks.

### Raw Data Location
`cic-ids-eda/data/raw/MachineLearningCVE/` contains 8 CSV files:
- `Monday-WorkingHours.pcap_ISCX.csv` - Benign only
- `Tuesday-WorkingHours.pcap_ISCX.csv` - FTP-Patator, SSH-Patator
- `Wednesday-workingHours.pcap_ISCX.csv` - DoS attacks, Heartbleed
- `Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv` - Web attacks
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv` - Infiltration
- `Friday-WorkingHours-Morning.pcap_ISCX.csv` - Bot
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv` - PortScan
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv` - DDoS

### Processed Data
`data/processed/cicids2017_clean_v2.parquet` - **Use this for analysis** (built by `scripts/build_dataset.py`):
- **Rows:** 2,519,262 (89.0% of raw retained; v1 kept only 49%)
- **Columns:** 73 (71 numeric features incl. `Has_Init_Win_fwd/bwd` indicators + Label + Meta_source)
- Build manifest with per-step row counts: `reports/dataset_v2_manifest.json`
- v1 (`cicids2017_clean.parquet`, 1.39M rows) is kept only so old reports stay traceable

### Class Distribution (v2)
| Label | Count | Percentage |
|-------|-------|------------|
| BENIGN | 2,094,218 | 83.13% |
| DoS Hulk | 172,717 | 6.86% |
| DDoS | 128,011 | 5.08% |
| PortScan | 90,130 | 3.58% |
| DoS GoldenEye | 10,286 | 0.41% |
| FTP-Patator | 5,931 | 0.24% |
| DoS slowloris | 5,384 | 0.21% |
| DoS Slowhttptest | 5,228 | 0.21% |
| SSH-Patator | 3,219 | 0.13% |
| Bot | 1,948 | 0.08% |
| Web Attack - Brute Force | 1,470 | 0.06% |
| Web Attack - XSS | 652 | 0.03% |
| Infiltration | 36 | <0.01% |
| Web Attack - Sql Injection | 21 | <0.01% |
| Heartbleed | 11 | <0.01% |

### Key Columns
- **Label:** Target variable (BENIGN or attack name)
- **Meta_source:** Capture day/scenario (8 values)
- **Destination Port:** 0-65535
- **Flow Duration:** Microseconds
- **Flow Bytes/s, Flow Packets/s:** Rate features
- **Total Fwd/Backward Packets:** Packet counts
- **Packet Length Mean/Std/Variance:** Size statistics
- **TCP Flags:** SYN, FIN, RST, PSH, ACK, URG, CWE, ECE Flag Counts
- **IAT (Inter-Arrival Time):** Flow/Fwd/Bwd IAT Mean/Std/Max/Min (CAN BE NEGATIVE)
- **Init_Win_bytes_forward/backward:** TCP window sizes

---

## Notebook Status

### cicids2017_eda.ipynb - COMPLETE (v2 protocol)
- Loads raw CSVs and cleans via `src/data/cleaning.py`; saves the v2 parquet
- EDA: class distribution, clustered correlation heatmap, PCA/UMAP, pair plots
- **Logistic Regression** (Pipeline, natural-prevalence test): PR-AUC=0.955 (no-skill 0.169);
  cross-day holdout 0.72–0.81; per-attack-type recall table included
- **Isolation Forest** (benign-only training): PR-AUC=0.540 at natural prevalence
- Evaluation-protocol review: `reports/notebook_review.md` (+ `scripts/validate_eda_baseline.py`)

### attack_types.ipynb - COMPLETE
Polars-based attack analysis, RF feature importance, multiclass RF; saves
`reports/attack_*` artifacts. Runs on the v2 parquet.

### day_of_the_weeks.ipynb - STUB ONLY
Current state: Just loads parquet and displays head(). Needs full implementation.

---

## Production Models (models/, served by api/)

LightGBM two-layer pipeline trained by `scripts/train_models.py` on v2 (20 FAST_FEATURES,
metadata version 2.0.0; all metrics in `models/model_metadata.json`):
- **Layer 1 (binary, natural-prevalence test):** PR-AUC=0.9997, precision=0.982, recall=0.999;
  **cross-day holdout: 0.824 (test Friday) / 0.466 (test Wed+Thu)** — the honest
  unseen-attack-behavior estimate
- **Layer 2 (multiclass, attacks only):** macro F1=0.91, weighted F1=0.997, per-class metrics in metadata
- MITRE mapping: `api/services/mitre_mapping.py` (exported to `models/mitre_mapping.json`)
- Clustering → technique mapping: `scripts/attack_clustering.py` → `reports/attack_clusters.json`
- SOC playbook: `reports/soc_triage_playbook.md`

---

## Data Cleaning Applied (src/data/cleaning.py, v2 policy)

1. Strip whitespace from column names; normalize mojibake labels (`Web Attack � …` → `- …`)
2. Drop 8 constant columns + duplicate `Fwd Header Length.1`
3. Fill genuine `Flow Bytes/s` NaN with 0; ±inf→NaN then drop rows over BOTH rate columns
4. `Init_Win_bytes_* = -1` is a sentinel (no TCP window observed): KEEP rows, add
   `Has_Init_Win_fwd/bwd` indicators, clamp -1→0 (v1 wrongly dropped these — 51% of data)
5. Negative-value row filter only on columns where negatives are impossible (excl. IAT + sentinels)
6. Drop exact duplicates (features+Label) and contradictory-label feature vectors

---

## Feature Families

### Flow Metadata
`Destination Port`, `Flow Duration`, `Flow Bytes/s`, `Flow Packets/s`

### Packet Statistics
`Total Fwd Packets`, `Total Backward Packets`, `Total Length of Fwd/Bwd Packets`, `Fwd/Bwd Packet Length Max/Min/Mean/Std`, `Packet Length Mean/Std/Variance/Max/Min`

### TCP Flags
`FIN Flag Count`, `SYN Flag Count`, `RST Flag Count`, `PSH Flag Count`, `ACK Flag Count`, `URG Flag Count`, `CWE Flag Count`, `ECE Flag Count`

### Inter-Arrival Time (IAT) - NOTE: Can be negative
`Flow IAT Mean/Std/Max/Min`, `Fwd IAT Total/Mean/Std/Max/Min`, `Bwd IAT Total/Mean/Std/Max/Min`

### Window Sizes
`Init_Win_bytes_forward`, `Init_Win_bytes_backward`, `min_seg_size_forward`, `act_data_pkt_fwd`

### Active/Idle Times
`Active Mean/Std/Max/Min`, `Idle Mean/Std/Max/Min`

### Subflow Features
`Subflow Fwd/Bwd Packets/Bytes`

### Derived Features
`Down/Up Ratio`, `Average Packet Size`, `Avg Fwd/Bwd Segment Size`, `Fwd Header Length`

---

## Development Commands

### Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm umap-learn pyarrow jupyter plotly fastapi uvicorn pydantic pytest httpx joblib
```

### Run Jupyter
```bash
jupyter lab
# or
jupyter notebook
```

### Run API (after implementation)
```bash
uvicorn api.main:app --reload
```

### Run tests (after implementation)
```bash
pytest tests/
```

---

## Important Implementation Notes

1. **Always use the v2 parquet** (`data/processed/cicids2017_clean_v2.parquet`); rebuild with
   `python scripts/build_dataset.py`
2. **IAT features can be negative** - don't filter these as negative
3. **Class imbalance is severe** - use class_weight='balanced' and precision-recall metrics,
   and always report the no-skill PR-AUC baseline (= attack prevalence) next to scores
4. **Split BEFORE resampling**: test sets must keep natural prevalence; downsample benign in
   the training portion only
5. **Report cross-day (`Meta_source`) holdout metrics** alongside random-split metrics — random
   splits are inflated because attack sessions straddle train/test
6. **Per-attack-type recall is mandatory** for binary evaluations — aggregate scores hide
   blind spots (web attacks, Bot)
7. **StandardScaler must fit on train only** — use a sklearn `Pipeline` so CV folds stay clean
8. **Use `select_dtypes(include=[np.number])`**, never `["int64","float64"]` — string dtype
   matching silently missed 47 of 71 numeric columns on in-memory frames (pandas block quirk)
9. **Heartbleed (11) and Infiltration (36)** have very few samples - may need to group or handle specially

---

## MITRE ATT&CK Mappings (for reference)

| Attack | ATT&CK ID | Tactic |
|--------|-----------|--------|
| DoS (all variants) | T1498.001 | Impact |
| DDoS | T1498 | Impact |
| Brute Force (FTP, SSH, Web) | T1110.001 | Credential Access |
| PortScan | T1046 | Discovery |
| Web Attack - XSS | T1059.007 | Execution |
| Web Attack - SQL Injection | T1190 | Initial Access |
| Bot | T1071.001 | Command and Control |
| Infiltration | T1071 | Command and Control |
| Heartbleed | T1190 | Initial Access |

---

## Next Steps

1. Implement `day_of_the_weeks.ipynb` (last remaining stub)
2. Consider grouping ultra-rare classes (Heartbleed, Infiltration, SQLi) for layer 2
3. Threshold tuning / calibration for the API's confidence levels against v2 metrics
