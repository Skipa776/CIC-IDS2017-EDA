"""Cleaning pipeline for CICIDS2017 raw CSVs (dataset v2).

Fixes the v1 cleaning issues documented in reports/notebook_review.md:
- R1: Init_Win_bytes = -1 is a sentinel (no TCP window observed), not invalid
  data. v1 dropped those rows (51% of the dataset); v2 keeps them behind
  indicator columns.
- R2: exact duplicates and contradictory-label feature vectors are removed.
- R3: inf/NaN policy is explicit over both rate columns instead of relying on
  their co-occurrence.
- Mojibake web-attack labels are normalized at the source.
"""

import glob
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
RAW_DIR = PROJECT_ROOT / "cic-ids-eda" / "data" / "raw" / "MachineLearningCVE"

# Constant in the raw data (notebook cell 13) plus the duplicated column
CONSTANT_COLUMNS = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]
DUPLICATE_COLUMNS = ["Fwd Header Length.1"]

RATE_COLUMNS = ["Flow Bytes/s", "Flow Packets/s"]

# -1 here means "no TCP window observed" (e.g., UDP, no handshake)
SENTINEL_COLUMNS = {
    "Init_Win_bytes_forward": "Has_Init_Win_fwd",
    "Init_Win_bytes_backward": "Has_Init_Win_bwd",
}


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the 8 raw CSVs with a Meta_source provenance column."""
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if len(csv_files) != 8:
        raise FileNotFoundError(
            f"Expected 8 CICIDS2017 CSVs in {raw_dir}, found {len(csv_files)}"
        )
    dfs = []
    for filepath in csv_files:
        df = pd.read_csv(filepath)
        df["Meta_source"] = os.path.basename(filepath).replace(".pcap_ISCX.csv", "")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _feature_columns(df: pd.DataFrame) -> list:
    # np.number, not ["int64", "float64"]: on frames mid-pipeline (post
    # concat/filter), string-based dtype matching can silently miss int64
    # blocks, which would scope the negative filter and dedup to a subset
    # of the features
    return [c for c in df.select_dtypes(include=[np.number]).columns]


def clean(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Apply the v2 cleaning policy. Returns (cleaned df, manifest of step counts)."""
    manifest: Dict = {"rows_raw": int(len(df))}

    # 1. Column names and labels
    df = df.rename(columns=str.strip)
    df = df.assign(Label=df["Label"].str.replace("�", "-", regex=False))

    # 2. Drop constant + duplicated columns
    df = df.drop(columns=CONSTANT_COLUMNS + DUPLICATE_COLUMNS)

    # 3. Genuine NaNs in Flow Bytes/s (zero-traffic flows) -> 0
    n_nan_rate = int(df["Flow Bytes/s"].isna().sum())
    df = df.assign(**{"Flow Bytes/s": df["Flow Bytes/s"].fillna(0)})
    manifest["flow_bytes_nan_filled"] = n_nan_rate

    # 4. inf -> NaN over both rate columns, drop those rows explicitly
    df = df.replace([np.inf, -np.inf], np.nan)
    before = len(df)
    df = df.dropna(subset=RATE_COLUMNS)
    manifest["rows_dropped_inf_rates"] = int(before - len(df))

    # 5. Sentinel handling: indicator column, then clamp -1 -> 0
    for col, flag in SENTINEL_COLUMNS.items():
        df[flag] = (df[col] >= 0).astype("int64")
        df[col] = df[col].clip(lower=0)
    manifest["sentinel_flows_kept"] = int(
        ((df["Has_Init_Win_fwd"] == 0) | (df["Has_Init_Win_bwd"] == 0)).sum()
    )

    # 6. Negative-value filter only where negatives are impossible
    numeric_cols = _feature_columns(df)
    check_cols = [
        c for c in numeric_cols
        if "IAT" not in c and c not in SENTINEL_COLUMNS
    ]
    before = len(df)
    df = df[(df[check_cols] >= 0).all(axis=1)]
    manifest["rows_dropped_negative"] = int(before - len(df))

    # 7. Deduplicate; remove contradictory-label feature vectors entirely
    feature_cols = [c for c in _feature_columns(df) if c != "Label"]
    before = len(df)
    df = df.drop_duplicates(subset=feature_cols + ["Label"])
    manifest["rows_dropped_duplicates"] = int(before - len(df))

    row_hash = pd.util.hash_pandas_object(df[feature_cols], index=False)
    labels_per_vector = df.groupby(row_hash.values)["Label"].transform("nunique")
    before = len(df)
    df = df[labels_per_vector == 1]
    manifest["rows_dropped_contradictory_labels"] = int(before - len(df))

    df = df.reset_index(drop=True)

    # 8. Sanity: nothing non-finite may survive
    values = df[_feature_columns(df)].to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite values remain after cleaning")

    manifest["rows_clean"] = int(len(df))
    manifest["columns"] = int(df.shape[1])
    manifest["label_counts"] = df["Label"].value_counts().to_dict()
    return df, manifest
