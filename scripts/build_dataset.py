#!/usr/bin/env python3
"""
Build the cleaned CICIDS2017 dataset (v2), or CSE-CIC-IDS2018 with the same cleaning.

Usage:
    python scripts/build_dataset.py              # CIC-IDS2017
    python scripts/build_dataset.py --year 2018  # CSE-CIC-IDS2018 (benign sampled, see src/data/cicids2018.py)

Output:
    - data/processed/cicids2017_clean_v2.parquet + reports/dataset_v2_manifest.json
    - data/processed/cicids2018_clean.parquet + reports/dataset_2018_manifest.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.cicids2018 import BENIGN_SAMPLE_RATE, load_raw_2018
from src.data.cleaning import PROJECT_ROOT, clean, load_raw

OUTPUTS = {
    "2017": (PROJECT_ROOT / "data" / "processed" / "cicids2017_clean_v2.parquet",
             PROJECT_ROOT / "reports" / "dataset_v2_manifest.json"),
    "2018": (PROJECT_ROOT / "data" / "processed" / "cicids2018_clean.parquet",
             PROJECT_ROOT / "reports" / "dataset_2018_manifest.json"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", choices=sorted(OUTPUTS), default="2017")
    year = parser.parse_args().year
    PARQUET_PATH, MANIFEST_PATH = OUTPUTS[year]

    print("Loading raw CSVs...")
    df = load_raw() if year == "2017" else load_raw_2018()
    print(f"  {len(df):,} rows, {df.shape[1]} columns")

    print("Cleaning (v2 policy)...")
    df, manifest = clean(df)
    if year == "2018":
        # rows_raw counts sampled benign; every attack flow was kept
        manifest = {"benign_sample_rate": BENIGN_SAMPLE_RATE, **manifest}
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
