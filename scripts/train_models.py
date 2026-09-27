#!/usr/bin/env python3
"""
Train and serialize the layered IDS models.

Usage:
    python scripts/train_models.py

This script:
1. Loads the processed CICIDS2017 data
2. Trains Layer 1 (binary) and Layer 2 (multi-class) classifiers
3. Evaluates both models
4. Saves all artifacts to the models/ directory
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split

from src.data.loader import (
    load_processed_data,
    get_feature_columns,
    prepare_binary_labels,
    prepare_multiclass_labels,
    stratified_sample,
)
from src.features.engineering import FAST_FEATURES, create_scaler, filter_features_by_name
from src.models.train import (
    train_layer1_binary,
    train_layer2_multiclass,
    save_models,
)
from src.models.evaluate import (
    evaluate_binary_model,
    evaluate_multiclass_model,
    get_inference_time,
)


DATA_PATH_V2 = Path(__file__).parent.parent / "data" / "processed" / "cicids2017_clean_v2.parquet"
MAX_BENIGN = 200_000


def downsample_benign(idx, y_binary, max_benign=MAX_BENIGN, seed=42):
    """Cap benign rows in a TRAINING index set; attack rows are all kept."""
    benign, attack = idx[y_binary[idx] == 0], idx[y_binary[idx] == 1]
    if len(benign) > max_benign:
        benign = np.random.RandomState(seed).choice(benign, size=max_benign, replace=False)
    return np.concatenate([benign, attack])


def random_split(y_binary):
    """Stratified 80/20 split; benign downsampled in the training portion only."""
    idx_train, idx_test = train_test_split(
        np.arange(len(y_binary)), test_size=0.2,
        stratify=y_binary, random_state=42,
    )
    return downsample_benign(idx_train, y_binary), idx_test


def crossday_splits(df, y_binary):
    """Yield (name, test_days, train_idx, test_idx) for the two cross-day holdouts.

    Each attack type occurs on a single day, so this also measures
    generalization to unseen attack behavior.
    """
    days = df['Meta_source'].unique()
    for name, test_days in [
        ('test_friday', [d for d in days if d.startswith('Friday')]),
        ('test_wed_thu', [d for d in days if d.startswith(('Wednesday', 'Thursday'))]),
    ]:
        test_mask = df['Meta_source'].isin(test_days).values
        tr, te = np.where(~test_mask)[0], np.where(test_mask)[0]
        yield name, test_days, downsample_benign(tr, y_binary), te


def main():
    print("=" * 60)
    print("CICIDS2017 IDS Model Training")
    print("=" * 60)

    # Load data
    print("\n1. Loading data (v2)...")
    df = load_processed_data(parquet_path=DATA_PATH_V2)
    print(f"   Loaded {len(df):,} samples with {len(df.columns)} columns")

    # Get feature columns
    all_feature_cols = get_feature_columns(df)
    print(f"   Found {len(all_feature_cols)} numeric features")

    # Filter to fast features
    available_fast_features = [f for f in FAST_FEATURES if f in all_feature_cols]
    print(f"   Using {len(available_fast_features)} fast features for inference")

    # Prepare features
    print("\n2. Preparing features...")
    X_full = df[available_fast_features].values.astype(np.float64)
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)

    # Prepare labels
    y_binary = prepare_binary_labels(df)
    y_multi, label_mapping = prepare_multiclass_labels(df)

    print(f"   Binary labels: {np.sum(y_binary == 0):,} benign, {np.sum(y_binary == 1):,} attack")
    print(f"   Multi-class labels: {len(label_mapping)} classes")

    # Train/test split FIRST (stratified): the test set keeps the natural
    # benign/attack prevalence, so PR metrics reflect a realistic deployment
    print("\n3. Splitting data (natural-prevalence test set)...")
    # Benign is downsampled in the TRAINING portion only (efficiency + imbalance)
    idx_train, idx_test = random_split(y_binary)

    X_train, X_test = X_full[idx_train], X_full[idx_test]
    y_bin_train, y_bin_test = y_binary[idx_train], y_binary[idx_test]
    y_multi_train, y_multi_test = y_multi[idx_train], y_multi[idx_test]
    labels_test = df['Label'].values[idx_test]

    print(f"   Train set: {len(X_train):,} samples (attack rate {y_bin_train.mean():.3f})")
    print(f"   Test set: {len(X_test):,} samples (attack rate {y_bin_test.mean():.3f} — natural)")

    # Scale features
    print("\n5. Scaling features...")
    scaler = create_scaler(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Layer 1 (Binary)
    print("\n6. Training Layer 1 (Binary Classifier)...")
    layer1_model = train_layer1_binary(X_train_scaled, y_bin_train)
    print(f"   Model type: {type(layer1_model).__name__}")

    # Evaluate Layer 1
    print("\n7. Evaluating Layer 1...")
    layer1_metrics = evaluate_binary_model(layer1_model, X_test_scaled, y_bin_test)
    layer1_metrics['no_skill_pr_auc'] = float(y_bin_test.mean())
    print(f"   Precision: {layer1_metrics['precision']:.4f}")
    print(f"   Recall:    {layer1_metrics['recall']:.4f}")
    print(f"   F1-Score:  {layer1_metrics['f1_score']:.4f}")
    print(f"   PR-AUC:    {layer1_metrics['pr_auc']:.4f} (no-skill {layer1_metrics['no_skill_pr_auc']:.3f})")

    # Per-attack-type recall: which families does the binary layer actually catch?
    print("\n7b. Per-attack-type recall (Layer 1)...")
    y_bin_pred = (layer1_model.predict_proba(X_test_scaled)[:, 1] >= 0.5).astype(int)
    per_attack = {}
    for label in np.unique(labels_test):
        mask = labels_test == label
        flagged = float(y_bin_pred[mask].mean())  # for BENIGN this is the FP rate
        per_attack[label] = {'n_test': int(mask.sum()), 'flagged_as_attack': flagged}
        print(f"   {label:<30s} n={mask.sum():>7,}  flagged={flagged:.3f}")
    layer1_metrics['per_attack_recall'] = per_attack

    # Cross-day holdout: deployment-realistic generalization estimate.
    # Each attack type occurs on a single day, so this also measures
    # generalization to unseen attack behavior.
    print("\n7c. Cross-day holdout evaluation (Layer 1)...")
    crossday = {}
    for name, test_days, tr, te in crossday_splits(df, y_binary):
        cd_scaler = create_scaler(X_full[tr])
        cd_model = train_layer1_binary(cd_scaler.transform(X_full[tr]), y_binary[tr])
        cd_metrics = evaluate_binary_model(cd_model, cd_scaler.transform(X_full[te]), y_binary[te])
        crossday[name] = {
            'pr_auc': cd_metrics['pr_auc'],
            'precision': cd_metrics['precision'],
            'recall': cd_metrics['recall'],
            'no_skill_pr_auc': float(y_binary[te].mean()),
            'test_days': test_days,
        }
        print(f"   {name}: PR-AUC {cd_metrics['pr_auc']:.4f} (no-skill {y_binary[te].mean():.3f})")
    layer1_metrics['crossday'] = crossday

    # Train Layer 2 (Multi-class) - only on attack samples
    print("\n8. Training Layer 2 (Multi-class Classifier)...")
    # Filter to attacks only for multi-class training
    attack_mask_train = y_bin_train == 1
    attack_mask_test = y_bin_test == 1

    X_train_attacks = X_train_scaled[attack_mask_train]
    y_multi_train_attacks = y_multi_train[attack_mask_train]
    X_test_attacks = X_test_scaled[attack_mask_test]
    y_multi_test_attacks = y_multi_test[attack_mask_test]

    # Get unique classes in attack subset
    unique_classes = np.unique(y_multi_train_attacks)
    num_classes = len(unique_classes)
    print(f"   Training on {len(X_train_attacks):,} attack samples ({num_classes} classes)")

    layer2_model = train_layer2_multiclass(
        X_train_attacks, y_multi_train_attacks,
        num_classes=len(label_mapping)  # Use full class count
    )
    print(f"   Model type: {type(layer2_model).__name__}")

    # Evaluate Layer 2
    print("\n9. Evaluating Layer 2...")
    label_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    layer2_metrics = evaluate_multiclass_model(
        layer2_model, X_test_attacks, y_multi_test_attacks, label_names
    )
    print(f"   Macro F1:    {layer2_metrics['macro_f1']:.4f}")
    print(f"   Weighted F1: {layer2_metrics['weighted_f1']:.4f}")

    # Measure inference time
    print("\n10. Measuring inference time...")
    sample = X_test_scaled[0:1]

    layer1_timing = get_inference_time(layer1_model, sample)
    print(f"   Layer 1: {layer1_timing['mean_ms']:.3f}ms (p95: {layer1_timing['p95_ms']:.3f}ms)")

    layer2_timing = get_inference_time(layer2_model, sample)
    print(f"   Layer 2: {layer2_timing['mean_ms']:.3f}ms (p95: {layer2_timing['p95_ms']:.3f}ms)")

    total_time = layer1_timing['mean_ms'] + layer2_timing['mean_ms']
    print(f"   Total:   {total_time:.3f}ms")

    # Compile all metrics
    all_metrics = {
        'layer1': {
            **layer1_metrics,
            'inference_time': layer1_timing,
        },
        'layer2': {
            'macro_f1': layer2_metrics['macro_f1'],
            'weighted_f1': layer2_metrics['weighted_f1'],
            'macro_precision': layer2_metrics['macro_precision'],
            'macro_recall': layer2_metrics['macro_recall'],
            'per_class': layer2_metrics['per_class'],
            'inference_time': layer2_timing,
        },
        'total_inference_time_ms': total_time,
    }

    # Save models
    print("\n11. Saving models...")
    save_models(
        layer1_model=layer1_model,
        layer2_model=layer2_model,
        scaler=scaler,
        feature_columns=available_fast_features,
        label_mapping=label_mapping,
        metrics=all_metrics,
        dataset=str(DATA_PATH_V2.name),
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    # Summary
    print("\nModel Summary:")
    print(f"  Layer 1 (Binary):     PR-AUC = {layer1_metrics['pr_auc']:.4f}")
    print(f"  Layer 2 (Multi-class): Macro F1 = {layer2_metrics['macro_f1']:.4f}")
    print(f"  Total inference time: {total_time:.3f}ms")
    print(f"\nArtifacts saved to: models/")


if __name__ == "__main__":
    main()
