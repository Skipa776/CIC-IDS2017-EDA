#!/usr/bin/env python3
"""
Build the cleaned CICIDS2017 dataset (v2).

Usage:
    python scripts/build_dataset.py

Output:
    - data/processed/cicids2017_clean_v2.parquet
    - reports/dataset_v2_manifest.json
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cleaning import PROJECT_ROOT, clean, load_raw

PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "cicids2017_clean_v2.parquet"
MANIFEST_PATH = PROJECT_ROOT / "reports" / "dataset_v2_manifest.json"


def main():
    print("Loading raw CSVs...")
    df = load_raw()
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    print("Cleaning (v2 policy)...")
    df, manifest = clean(df)
    print(f"  {manifest['rows_raw']:,} -> {manifest['rows_clean']:,} rows "
          f"({manifest['rows_clean'] / manifest['rows_raw']:.1%} retained)")
    for key in (
        "flow_bytes_nan_filled",
        "rows_dropped_inf_rates",
        "sentinel_flows_kept",
        "rows_dropped_negative",
        "rows_dropped_duplicates",
        "rows_dropped_contradictory_labels",
    ):
        print(f"  {key}: {manifest[key]:,}")

    PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PARQUET_PATH, index=False)
    print(f"Saved {PARQUET_PATH}")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
