#!/usr/bin/env python3
"""ROC AUC of the autoencoder on every forward-in-time fold and seed.

Refits the autoencoder exactly as scripts/run_pipeline.py does (same partitions,
same benign cap, same seed), scores the test set, and reports ROC AUC. As a
check that it is the same model, it recomputes average precision and fails if
it differs from the value stored in reports/results/<fold>.json.

ROC AUC = probability that a random test attack flow scores above a random test
benign flow. Unlike average precision it ignores class balance, so it reads high
on imbalanced data; read it beside the AP and its no-skill baseline (prevalence).

Usage: python scripts/autoencoder_roc.py [--folds ...] [--seeds ...]
Writes reports/results/autoencoder_roc_auc.json
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from scripts.run_pipeline import DATA, load
from src.evaluation.contract import provenance
from src.evaluation.folds import FOLDS, FOLDS_BY_NAME, partition
from src.models.detectors import cap_benign, fit_autoencoder

RESULTS = ROOT / "reports" / "results"
TOLERANCE = 1e-9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", nargs="+", default=[f.name for f in FOLDS])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    out = {"detector": "autoencoder", "metric": "ROC AUC on the test set (benign weights do not change it: "
                                                 "benign is sampled uniformly)",
           "folds": {}}
    with threadpool_limits(limits=args.threads):
        for name in args.folds:
            fold = FOLDS_BY_NAME[name]
            stored = {r["seed"]: r["detectors"]["autoencoder"]
                      for r in json.loads((RESULTS / f"{name}.json").read_text())["runs"]}
            tr_data, te_data = load(fold.train_year), load(fold.test_year)
            cross = fold.train_year != fold.test_year
            rows = []
            for seed in args.seeds:
                tr, _, te = partition(fold, tr_data["days"], tr_data["y"], seed, te_data["days"] if cross else None)
                capped = cap_benign(tr, tr_data["y"], seed)
                score = fit_autoencoder(tr_data["X"][capped[tr_data["y"][capped] == 0]], seed, args.threads)
                y_te, w_te = te_data["y"][te], te_data["weight"][te]
                s = score(te_data["X"][te])
                ap = float(average_precision_score(y_te, s, sample_weight=w_te))
                if abs(ap - stored[seed]["average_precision"]) > TOLERANCE:
                    raise AssertionError(f"{name} seed {seed}: AP {ap} != stored {stored[seed]['average_precision']}; "
                                         "not the same model as the pipeline run")
                auc = float(roc_auc_score(y_te, s))
                rows.append({"seed": seed, "roc_auc": auc, "average_precision": ap,
                             "prevalence": stored[seed]["prevalence"]})
                print(f"{name} seed {seed}: ROC AUC {auc:.4f}  AP {ap:.4f} (matches pipeline)", flush=True)
            aucs = [r["roc_auc"] for r in rows]
            out["folds"][name] = {"question": fold.question, "runs": rows,
                                  "roc_auc": {"min": min(aucs), "median": float(np.median(aucs)), "max": max(aucs)}}
    out["provenance"] = provenance(sorted({DATA[y] for y in ("2017", "2018")}))
    (RESULTS / "autoencoder_roc_auc.json").write_text(json.dumps(out, indent=1))
    print("Wrote reports/results/autoencoder_roc_auc.json")


if __name__ == "__main__":
    main()
