#!/usr/bin/env python3
"""Audit legacy contamination and evaluate label-blind holdouts from every file.

Primary model uses all numeric features, avoiding the historical globally
chosen 20-feature subset. Profile groups are proxies, not actual sessions.
Raw CSV order is not known to be chronological. No score is forced to be low.
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from scripts.evaluate_generalization import evaluate_candidate, make_partitions
from scripts.train_models import DATA_PATH_V2, downsample_benign, random_split
from src.data.cleaning import RAW_DIR, clean
from src.data.loader import get_feature_columns, prepare_binary_labels
from src.features.engineering import FAST_FEATURES
from src.models.leakage import behavior_groups, grouped_partitions, overlap_report, purge_profiles, purged_file_blocks

PROTOCOLS = ["grouped_quarter_octave", "grouped_eighth_octave", "purged_file_blocks",
             "test_friday", "test_wed_thu", "forward_thursday"]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def legacy_audit():
    df = pd.read_parquet(DATA_PATH_V2)
    y = prepare_binary_labels(df)
    labels = df["Label"].to_numpy()
    train, test = random_split(y)
    exact = pd.util.hash_pandas_object(df[FAST_FEATURES], index=False).to_numpy()
    profiles = behavior_groups(df)
    full = pd.util.hash_pandas_object(df[get_feature_columns(df)], index=False).to_numpy()
    return {"data_sha256": file_hash(DATA_PATH_V2), "model_input_features": FAST_FEATURES,
            "duplicates_on_20_inputs": int(df.duplicated(FAST_FEATURES).sum()),
            "duplicates_on_71_inputs": int(len(full) - len(np.unique(full))),
            "random_split_seed_42": overlap_report(train, test, exact, profiles, y, labels),
            "profile_rule": "quarter-octave fixed bins; same ports, flags and initial windows"}


def load_raw_evaluation(raw_dir):
    frames, manifests, lengths, hashes = [], {}, {}, {}
    paths = sorted(raw_dir.glob("*.csv"))
    if len(paths) != 8:
        raise ValueError(f"Expected eight 2017 source files, found {len(paths)}")
    for path in paths:
        raw = pd.read_csv(path)
        source = path.name.replace(".pcap_ISCX.csv", "")
        lengths[source] = len(raw)
        raw["Meta_source"] = source
        # String provenance is kept out of the numeric feature matrix.
        raw["Meta_raw_row"] = np.arange(len(raw)).astype(str)
        frame, manifest = clean(raw, deduplicate=False)
        frames.append(frame)
        manifests[source] = manifest
        hashes[path.name] = file_hash(path)
        print(f"Loaded {source}: {len(raw):,} -> {len(frame):,}, label conflicts retained", flush=True)
    return pd.concat(frames, ignore_index=True), manifests, lengths, hashes


def partition_info(df, y, idx, profiles):
    labels = df["Label"].to_numpy()
    sources = df["Meta_source"].to_numpy()
    return {"n": len(idx), "n_behavior_groups": len(np.unique(profiles[idx])),
            "attack_prevalence": float(y[idx].mean()),
            "index_sha256": hashlib.sha256(idx.tobytes()).hexdigest(),
            "by_file": {str(s): {"n": int(np.sum(sources[idx] == s)),
                                  "attack_prevalence": float(y[idx][sources[idx] == s].mean()),
                                  "labels": {str(l): int(n) for l, n in zip(*np.unique(labels[idx][sources[idx] == s], return_counts=True))}}
                        for s in np.unique(sources[idx])}}


def write_summary(result, path):
    audit = result["legacy_audit"]["random_split_seed_42"]
    lines = ["# Leakage investigation and corrected holdouts", "",
             "## Confirmed defects", "",
             f"- {audit['exact_input_overlap_n']:,} legacy test flows have an exact 20-input twin in training "
             f"({audit['exact_input_overlap_fraction']:.2%}); {audit['attack_exact_overlap_n']} are attacks.",
             f"- {audit['attack_profile_overlap_fraction']:.2%} of legacy attack test flows share a fixed behavior "
             "profile with training. This is dependence under the declared proxy, not proof of shared sessions.",
             "- Legacy cleaning removes contradictory labels globally before splitting. The new evaluation "
             "retains conflicting labels and repeated flows, grouping them across partitions.",
             "- The feature-importance notebook fits on labels before its test split. The primary new model "
             "uses all 71 numeric inputs, avoiding that supervised feature selection.", "",
             "## New protocols", "",
             "Global label-blind behavior groups are assigned to train/validation/test (60/20/20 groups), "
             "including matching profiles across different files. Actual row proportions and prevalence may differ.", "",
             "A separate purged block test takes the tail of every raw CSV, reserves intervening gaps, and removes "
             "test/validation profiles from earlier partitions. CSV order is not verified time order. "
             "Quarter-octave and eighth-octave profile splits assess grouping sensitivity; no rule is selected by test score.", "",
             "Whole-day Friday, Wed+Thu and forward-Thursday tests are rerun on this raw-derived population "
             "with global profile purging as well. These test capture/attack-family shift; "
             "Wed+Thu remains explicitly nonchronological.", "",
             "Scalers and models fit on training only. Thresholds use validation benign flows only. "
             "Only training benign flows are capped. The original 20-feature model is a diagnostic; "
             "the 71-feature model is the primary baseline. All captures have already been explored, "
             "so these remain development evaluations.", "",
             "## Results at the 1% validation FPR budget", "",
             "| Protocol | Seed | Model | Test AP | Test prevalence | Actual test FPR | Recall |",
             "|---|---:|---|---:|---:|---:|---:|"]
    for run in result["runs"]:
        for name, report in run["candidates"].items():
            m = report["test_at_frozen_thresholds"]["0.01"]
            lines.append(f"| {run['protocol']} | {run['seed']} | {name} | {report['test_average_precision']:.4f} "
                         f"| {report['test_prevalence']:.4f} | {m['fpr']:.4f} | {m['recall']:.4f} |")
    lines += ["", "## Remaining limits", ""] + [f"- {s}" for s in result["limitations"]]
    if "shuffled_training_labels" in result:
        shuffle = result["shuffled_training_labels"]
        lines += ["", f"Shuffled-training-label control on the quarter-octave grouped split: "
                  f"test AP {shuffle['test_average_precision']:.4f}; prevalence {shuffle['test_prevalence']:.4f}. "
                  "This is a negative control for learning from corrupted training targets; "
                  "near-baseline AP does not rule out target-derived features."]
    lines += ["", "Source: [leakage_audit.json](leakage_audit.json). Historical metrics remain archived "
              "for auditability and are not promoted as generalization evidence.", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/leakage_audit.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--protocols", nargs="+", choices=PROTOCOLS, default=PROTOCOLS)
    args = parser.parse_args()
    if args.threads < 1 or not args.seeds:
        parser.error("positive threads and at least one seed are required")
    result = {"complete": False, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
              "legacy_audit": legacy_audit(),
              "config": {"seeds": args.seeds, "threads": args.threads, "primary": "lgbm71",
                         "protocols": args.protocols,
                         "profile_bin_widths": [0.25, 0.125], "threshold_source": "validation benign only"},
              "versions": {package: version(package) for package in ["numpy", "pandas", "scikit-learn", "lightgbm"]},
              "code_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in [Path(__file__),
                               ROOT / "src/models/leakage.py", ROOT / "src/data/cleaning.py",
                               ROOT / "scripts/evaluate_generalization.py", ROOT / "src/models/train.py",
                               ROOT / "src/models/evaluate.py", ROOT / "scripts/train_models.py"]},
              "limitations": ["Behavior bins do not identify real sessions; near neighbors across bin boundaries can remain.",
                               "File-order blocks cannot be called chronological without timestamps.",
                               "Same-file holdouts retain capture/campaign context; whole-day and cross-year tests remain essential.",
                               "Already explored captures cannot provide a fresh confirmatory test.",
                               "Seed variation is not a population confidence interval; no operational recall target is agreed."],
              "runs": []}
    print("Legacy overlap:", result["legacy_audit"]["random_split_seed_42"], flush=True)
    df, manifests, lengths, hashes = load_raw_evaluation(args.raw_dir)
    result["raw_files_sha256"] = hashes
    result["feature_only_cleaning"] = manifests
    features = get_feature_columns(df)
    if len(features) != 71:
        raise ValueError(f"Expected 71 feature columns, found {len(features)}")
    result["features"] = features
    result["rows"] = len(df)
    X = df[features].to_numpy(dtype=float)
    y = prepare_binary_labels(df)
    labels = df["Label"].to_numpy()
    sources = df["Meta_source"].to_numpy()
    days = df["Meta_source"].str.split("-").str[0].to_numpy()
    rows = df["Meta_raw_row"].to_numpy(dtype=int)
    exact = pd.util.hash_pandas_object(df[FAST_FEATURES], index=False).to_numpy()
    profiles = {width: behavior_groups(df, width) for width in [0.25, 0.125]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=args.threads):
        for seed in args.seeds:
            for name in args.protocols:
                width = 0.125 if name == "grouped_eighth_octave" else 0.25
                groups = profiles[width]
                if name == "purged_file_blocks":
                    raw_train, val, test = purged_file_blocks(sources, rows, groups, lengths)
                elif name.startswith("grouped_"):
                    raw_train, val, test = grouped_partitions(groups, seed)
                else:
                    raw_train, val, test = purge_profiles(*make_partitions(days, y, name, seed), groups)
                train = downsample_benign(raw_train, y, seed=seed)
                if any(len(np.unique(y[idx])) != 2 for idx in [train, val, test]):
                    raise ValueError(f"{name}: insufficient class coverage for evaluation")
                if name in PROTOCOLS[:3] and (set(sources[test]) != set(sources) or set(sources[val]) != set(sources)):
                    raise ValueError(f"{name}: validation and test must represent every source file")
                for left, right in [(train, val), (train, test), (val, test)]:
                    if np.isin(groups[left], groups[right]).any():
                        raise AssertionError("Related profile crossed a partition boundary")
                run = {"protocol": name, "seed": seed, "bin_width": width,
                       "partition_counts_before_training_cap": {"train": len(raw_train), "validation": len(val), "test": len(test),
                                                                "excluded": len(df) - len(raw_train) - len(val) - len(test)},
                       "partitions": {n: partition_info(df, y, idx, groups) for n, idx in [("train", train), ("validation", val), ("test", test)]},
                       "overlap": overlap_report(train, test, exact, groups, y, labels), "candidates": {}}
                result["runs"].append(run)
                for model_name in ["lgbm71", "lgbm20"]:
                    cols = list(range(len(features))) if model_name == "lgbm71" else [features.index(f) for f in FAST_FEATURES]
                    print(f"{name} seed={seed} {model_name}: train={len(train):,} val={len(val):,} test={len(test):,}", flush=True)
                    report = evaluate_candidate(X, y, labels, train, val, test, cols, model_name, seed, args.threads, sources)
                    report["test_by_source_at_1pct"] = report.pop("test_by_day_at_1pct")
                    report["features"] = [features[i] for i in cols]
                    run["candidates"][model_name] = report
                    m = report["test_at_frozen_thresholds"]["0.01"]
                    print(f"  AP={report['test_average_precision']:.4f} prevalence={report['test_prevalence']:.4f} FPR={m['fpr']:.4f} recall={m['recall']:.4f}", flush=True)
                    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
                if seed == args.seeds[0] and name == "grouped_quarter_octave":
                    shuffled = y.copy()
                    shuffled[train] = np.random.RandomState(seed).permutation(y[train])
                    shuffled_labels = labels.copy()
                    shuffled_labels[train] = np.where(shuffled[train], "SHUFFLED_ATTACK", "BENIGN")
                    result["shuffled_training_labels"] = evaluate_candidate(
                        X, shuffled, shuffled_labels, train, val, test, list(range(len(features))),
                        "lgbm71", seed, args.threads, sources)
        result["complete"] = True
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        write_summary(result, args.output.with_suffix(".md"))
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
