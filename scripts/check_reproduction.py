#!/usr/bin/env python3
"""Gate check: the evaluation contract reproduces evaluate_generalization.py.

Re-scores one run from reports/generalization_evaluation.json with the new
contract (src/evaluation/contract.py) on the same data and partitions, and
fails if test AP, actual FPR or recall at the 1% budget differ.

Usage: python scripts/check_reproduction.py [--split forward_thursday] [--seed 42]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from scripts.evaluate_generalization import make_partitions
from scripts.train_models import DATA_PATH_V2, downsample_benign
from src.evaluation.contract import evaluate, frozen_thresholds
from src.models.cross_dataset import FAMILY
from src.models.detectors import fit_lgbm

TOLERANCE = 1e-6


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="forward_thursday")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    reference = json.loads((ROOT / "reports/generalization_evaluation.json").read_text())
    run = next(r for r in reference["runs"] if r["split"] == args.split and r["seed"] == args.seed)
    ref = run["candidates"]["lgbm71"]
    threads = reference["config"]["threads"]

    df = pd.read_parquet(DATA_PATH_V2)
    features = reference["data"]["features"]
    X = np.nan_to_num(df[features].to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = (df["Label"] != "BENIGN").to_numpy().astype(int)
    days = df["Meta_source"].str.split("-").str[0].to_numpy()
    families = df["Label"].map(FAMILY).to_numpy()

    raw_train, val, test = make_partitions(days, y, args.split, args.seed)
    train = downsample_benign(raw_train, y, seed=args.seed)
    with threadpool_limits(limits=threads):
        score = fit_lgbm(X[train], y[train], args.seed, threads)
        val_scores = score(X[val])
        thresholds = frozen_thresholds(val_scores[y[val] == 0])
        result = evaluate(y[test], families[test], score(X[test]), thresholds)

    ours = result["budgets"]["0.01"]
    theirs = ref["test_at_frozen_thresholds"]["0.01"]
    pairs = {
        "test AP": (result["average_precision"], ref["test_average_precision"]),
        "actual test FPR at 1%": (ours["fpr"], theirs["fpr"]),
        "recall at 1%": (ours["recall"], theirs["recall"]),
    }
    ok = True
    for name, (a, b) in pairs.items():
        match = abs(a - b) <= TOLERANCE
        ok &= match
        print(f"{name:<24} contract {a:.6f}  reference {b:.6f}  {'OK' if match else 'MISMATCH'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
