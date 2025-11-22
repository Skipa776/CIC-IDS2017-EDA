# CICIDS2017 EDA & Baseline Detection

This repository contains an exploratory data analysis (EDA) and baseline detection models for the **CICIDS2017** intrusion detection dataset.

The goals of this project are to:

- Understand the structure and quality of the CICIDS2017 flow data.
- Document feature families relevant to intrusion detection (packet volume, timing, TCP flags, etc.).
- Quantify class imbalance and traffic patterns across attack types and capture days.
- Train and evaluate simple, interpretable baseline models:
  - **Logistic Regression** (supervised, benign vs attack).
  - **Isolation Forest** (unsupervised anomaly detection).
- Produce plots and metrics suitable for review and future modeling work.

> ⚠️ **Note:** The raw CICIDS2017 data is **not stored** in this repository due to size and licensing. You must download it separately and place it under `data/raw/` (see instructions below).

---

## Repository Structure

```text
.
├── data/
│   ├── raw/          # raw CICIDS2017 CSVs (NOT tracked in git)
│   └── processed/    # cleaned/derived datasets (NOT tracked in git)
│
├── notebooks/
│   ├── cicids2017_eda.ipynb       # main high-level EDA + baselines
│   ├── attack_types.ipynb         # (future) EDA & metrics per attack type
│   └── day_of_the_weeks.ipynb     # (future) EDA & metrics per capture day/source
│
├── reports/
│   └── figures/     # saved plots (PR curves, confusion matrices, etc.)
│
├── src/
│   ├── data/        # (future) data loading/processing utilities
│   ├── features/    # (future) feature engineering utilities
│   └── models/      # (future) model definitions & training scripts
│
├── .gitignore
└── README.md

```

## Instructions
## Dataset: CICIDS2017

The project uses the **CICIDS2017** dataset from the Canadian Institute for Cybersecurity.

You must download the data yourself and place the CSV files under `data/raw/`.

### 1. Download the dataset

1. Visit the official CICIDS2017 page (Canadian Institute for Cybersecurity).
2. Request/download the dataset (you’ll get a compressed archive containing multiple `.pcap_ISCX.csv` files).
3. Extract the CSV files locally.

Typical filenames include:

- `Monday-WorkingHours.pcap_ISCX.csv`
- `Tuesday-WorkingHours.pcap_ISCX.csv`
- `Wednesday-workingHours.pcap_ISCX.csv`
- `Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv`
- `Friday-WorkingHours-Morning.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv`
- `Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- etc.

### 2. Place the files in `data/raw/`

Inside this repo, create:

```text
data/raw/MachineLearningCVE/

and place all CICIDS2017 CSVs there, for example:
data/raw/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv
data/raw/MachineLearningCVE/Tuesday-WorkingHours.pcap_ISCX.csv
...
```

The main EDA notebook expects this structure:

- Root of repo: .../CIC-IDS2017-EDA/
- Raw data CSVs: data/raw/MachineLearningCVE/*.pcap_ISCX.csv

Environment & Dependencies

This project assumes:
- Python 3.9+
- Common data science stack: pandas, numpy
- matplotlib, seaborn
- scikit-learn
- pyarrow (for Parquet)
- umap-learn (for UMAP visualizations, optional)
- jupyter / notebook / jupyterlab

A minimal example using pip:

```sh
pip install \
  pandas numpy matplotlib seaborn scikit-learn \
  umap-learn pyarrow jupyter
```

If you use conda, you can create an environment and install these via conda/mamba.

Running the Main EDA Notebook

The primary analysis lives in:
notebooks/cicids2017_eda.ipynb

Steps

- Ensure raw data is placed under data/raw/MachineLearningCVE/ as described above.
- Start Jupyter:

```sh
cd /path/to/CIC-IDS2017-EDA
jupyter lab # or jupyter notebook
```

- Open notebooks/cicids2017_eda.ipynb.
- Run all cells top-to-bottom.

What this notebook does

At a high level, the notebook:

- Setup
  - Imports libraries and sets basic plotting styles.
- Load Data
  - Loads all CICIDS2017 CSV files from data/raw/MachineLearningCVE/.
  - Adds a Meta_source column indicating the source file/capture.
  - Concatenates everything into a single combined_df.
- Cleaning & Feature Engineering
  - Handles NaNs and infinities (e.g., Flow Bytes/s).
  - Drops constant or redundant columns.
  - Strips whitespace from column names.
  - Enforces basic numeric sanity checks (e.g., no negative counts for non-IAT numeric features).
  - Defines numeric_cols as the numeric feature set for EDA and modeling.
- Exploratory Data Analysis
  - Label distribution and class imbalance.
  - Feature families overview (flow metadata, TCP flags, packet stats, timing, etc.).
  - Packet volume distributions (Total Fwd/Backward Packets) by label.
  - Correlation heatmap of numeric features.
  - Low-dimensional projections (e.g., PCA + UMAP/t-SNE) on a balanced subsample.
- Baseline Models
  - Logistic Regression (supervised):
    - Binary target: benign vs attack.
    - Stratified train/test split.
    - Standardization of numeric features.
    - 5-fold cross-validated PR-AUC on the training set.
  - Isolation Forest (unsupervised):
    - Trained on benign-only traffic.
    - Applied as an anomaly detector on the test set.
- Evaluation & Metrics
  - Classification reports for both models.
  - Confusion matrices (benign vs attack).
  - Precision–Recall curves and PR-AUC for both models.
  - Comparison table summarizing:
    - Precision, recall, F1, PR-AUC for the attack class.
- Executive Summary
  - High-level bullets summarizing:
    - Data coverage and cleaning.
    - Class imbalance.
    - Key feature/structure insights.
    - Baseline model performance.
    - Next steps (per-attack analysis, time-aware splits, advanced models).

The final cell prints:

```python
print("Audit-ready: metrics saved to reports/figures/")
```

once evaluation plots have been saved.

Processed Data Artifacts

After cleaning, the notebook can save a consolidated, cleaned dataset under data/processed/, for example:

```python
from pathlib import Path
processed_dir = Path("data/processed")
processed_dir.mkdir(parents=True, exist_ok=True)
combined_df.to_parquet(processed_dir / "cicids2017_clean.parquet", index=False)
combined_df.to_csv(processed_dir / "cicids2017_clean.csv", index=False)
```

These processed files are not tracked in git (see .gitignore) but can be reused by:
- notebooks/attack_types.ipynb
- notebooks/day_of_the_weeks.ipynb
- Future modeling notebooks under notebooks/ or src/models/.

Planned / Optional Notebooks

The project roadmap includes:

- notebooks/attack_types.ipynb
  - Per-attack-type EDA and model performance:
  - Per-class counts and imbalance.
  - Per-attack precision/recall/F1 for the baseline models.
  - Identification of which attacks are hardest/easiest to detect.
- notebooks/day_of_the_weeks.ipynb
  - Day-based or scenario-based EDA:
  - Behavior by Meta_source (capture days / scenarios).
  - Time-aware or day-wise train/test splits to simulate generalization to unseen days.
- Modeling notebooks (future)
  - Additional notebooks may explore:
  - Tree-based models (Random Forest, XGBoost).
  - Autoencoder-based anomaly detection.
  - Threat context mapping to MITRE ATT&CK tactics/techniques.

Git & Data Hygiene

To keep the repository lightweight and shareable:

- Raw data and large artifacts are not committed to git.
- data/raw/, data/processed/, models/, and reports/figures/ are ignored via .gitignore.
- Check out the repo, download the data separately, and place it under data/raw/.
- If you add your own large files, make sure they go into an ignored directory or update .gitignore accordingly.

License / Usage Notes

The CICIDS2017 dataset is distributed by the Canadian Institute for Cybersecurity under its own terms. Make sure you comply with their license and usage conditions.

This code is intended for research, education, and experimentation with intrusion detection; it is not a production-ready security product.
