#!/usr/bin/env python3
"""
Lightweight Overfitting Test Suite for CICIDS2017 Attack Classifier

Memory-efficient version that runs sequentially to avoid RAM exhaustion.
Designed for laptops with limited RAM (8-16GB).

Tests:
1. Train/Test Gap - Simple overfitting check
2. Temporal Split - Train Mon-Thu, test Friday
3. Learning Curve - Incremental training sizes
4. Shuffle Test - Sanity check with random labels

Usage:
    python scripts/test_overfitting_light.py
"""

import os
import gc
import json
from datetime import datetime

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score

# === MEMORY-SAFE SETTINGS ===
# All operations run sequentially (n_jobs=1)
# Smaller sample sizes and fewer trees
SAMPLE_SIZE = 20_000  # Reduced from 40k
N_ESTIMATORS = 50     # Reduced from 100
MAX_DEPTH = 12        # Reduced from 20
N_FOLDS = 3           # Reduced from 5

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data/processed/cicids2017_clean.parquet')
REPORT_PATH = os.path.join(PROJECT_ROOT, 'models/overfitting_report_light.json')
FIGURES_PATH = os.path.join(PROJECT_ROOT, 'reports/figures')


def cleanup():
    """Force garbage collection to free memory."""
    gc.collect()


def load_data():
    """Load dataset with minimal memory footprint."""
    print("Loading data...")
    df = pl.read_parquet(DATA_PATH)
    print(f"  Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    return df


def get_feature_cols(df):
    """Get numeric feature columns."""
    exclude = ['Label', 'Meta_source', 'Attack_Family']
    return [c for c in df.columns if c not in exclude
            and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]


def prepare_data(df, feature_cols, sample_size=SAMPLE_SIZE, min_class_samples=5):
    """Sample and prepare features/labels with stratification."""
    # Count samples per class
    label_counts = df.group_by('Label').len()

    # Filter out classes with too few samples for stratification
    valid_labels = label_counts.filter(
        pl.col('len') >= min_class_samples * 2  # Need enough for train/test split
    )['Label'].to_list()

    df_filtered = df.filter(pl.col('Label').is_in(valid_labels))

    # Stratified sample: take proportional samples from each class
    samples_per_class = max(10, sample_size // len(valid_labels))
    df_sample = df_filtered.group_by('Label').agg(
        pl.all().sample(n=samples_per_class, with_replacement=True, seed=42)
    ).explode(pl.all().exclude('Label'))

    # Cap total size
    if df_sample.height > sample_size:
        df_sample = df_sample.sample(n=sample_size, seed=42)

    X = df_sample.select(feature_cols).to_numpy()
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = df_sample['Label'].to_numpy()

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    print(f"  Sampled {df_sample.height:,} rows, {len(valid_labels)} classes")

    return X, y_enc, le


def create_model():
    """Create a lightweight RandomForest model."""
    return RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight='balanced',
        n_jobs=1,  # SEQUENTIAL - key for memory safety
        random_state=42
    )


def test_train_test_gap(df, feature_cols):
    """
    Test 1: Basic Train/Test Performance Gap

    Simple check: if train score >> test score, model is overfitting.
    """
    print("\n" + "="*60)
    print("TEST 1: TRAIN/TEST PERFORMANCE GAP")
    print("="*60)

    X, y, _ = prepare_data(df, feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = create_model()
    model.fit(X_train_s, y_train)

    train_f1 = f1_score(y_train, model.predict(X_train_s), average='macro', zero_division=0)
    test_f1 = f1_score(y_test, model.predict(X_test_s), average='macro', zero_division=0)
    gap = train_f1 - test_f1

    print(f"\n  Train F1: {train_f1:.4f}")
    print(f"  Test F1:  {test_f1:.4f}")
    print(f"  Gap:      {gap:.4f}")

    if gap > 0.1:
        print(f"\n  ⚠️  WARNING: Large gap suggests overfitting")
    elif gap > 0.05:
        print(f"\n  ⚡ CAUTION: Moderate gap, monitor closely")
    else:
        print(f"\n  ✓ OK: Small gap, model generalizes well")

    del model, X_train_s, X_test_s
    cleanup()

    return {'train_f1': train_f1, 'test_f1': test_f1, 'gap': gap}


def test_temporal_split(df, feature_cols):
    """
    Test 2: Temporal Generalization

    Train on Mon-Thu, test on Friday. Tests if model generalizes across time.
    """
    print("\n" + "="*60)
    print("TEST 2: TEMPORAL SPLIT (Mon-Thu → Friday)")
    print("="*60)

    friday_sources = [
        'Friday-WorkingHours-Morning',
        'Friday-WorkingHours-Afternoon-DDos',
        'Friday-WorkingHours-Afternoon-PortScan'
    ]

    df_train = df.filter(~pl.col('Meta_source').is_in(friday_sources))
    df_test = df.filter(pl.col('Meta_source').is_in(friday_sources))

    # Find common labels between train and test
    train_labels = set(df_train['Label'].unique().to_list())
    test_labels = set(df_test['Label'].unique().to_list())
    common_labels = list(train_labels & test_labels)

    # Filter to common labels only
    df_train = df_train.filter(pl.col('Label').is_in(common_labels))
    df_test = df_test.filter(pl.col('Label').is_in(common_labels))

    # Sample each split
    sample_size = SAMPLE_SIZE // 2
    df_train = df_train.sample(n=min(sample_size, df_train.height), seed=42)
    df_test = df_test.sample(n=min(sample_size, df_test.height), seed=42)

    print(f"\n  Common classes: {len(common_labels)}")
    print(f"  Train (Mon-Thu): {df_train.height:,} samples")
    print(f"  Test (Friday):   {df_test.height:,} samples")

    # Prepare features
    X_train = df_train.select(feature_cols).to_numpy()
    X_test = df_test.select(feature_cols).to_numpy()
    X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
    X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

    # Fit encoder on all common labels to ensure consistency
    le = LabelEncoder()
    le.fit(common_labels)
    y_train = le.transform(df_train['Label'].to_numpy())
    y_test = le.transform(df_test['Label'].to_numpy())

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = create_model()
    model.fit(X_train_s, y_train)

    train_f1 = f1_score(y_train, model.predict(X_train_s), average='macro', zero_division=0)
    test_f1 = f1_score(y_test, model.predict(X_test_s), average='macro', zero_division=0)
    gap = train_f1 - test_f1

    print(f"\n  Train F1: {train_f1:.4f}")
    print(f"  Test F1:  {test_f1:.4f}")
    print(f"  Gap:      {gap:.4f}")

    if gap > 0.1:
        print(f"\n  ⚠️  WARNING: Model doesn't generalize well across time")
    else:
        print(f"\n  ✓ OK: Model generalizes across time periods")

    del model, X_train_s, X_test_s, df_train, df_test
    cleanup()

    return {'train_f1': train_f1, 'test_f1': test_f1, 'gap': gap}


def test_learning_curve(df, feature_cols):
    """
    Test 3: Learning Curve (Sequential)

    Train with increasing data sizes and compare train vs validation.
    Runs sequentially to minimize memory usage.
    """
    print("\n" + "="*60)
    print("TEST 3: LEARNING CURVE (Sequential)")
    print("="*60)

    X, y, _ = prepare_data(df, feature_cols, sample_size=15_000)

    train_sizes_pct = [0.2, 0.4, 0.6, 0.8, 1.0]
    results = {'sizes': [], 'train_scores': [], 'val_scores': []}

    # Single train/val split for consistency
    X_full, X_val, y_full, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_val_s = scaler.fit_transform(X_val)  # Will refit for each size

    print(f"\n{'Size':>8} {'Train F1':>10} {'Val F1':>10} {'Gap':>8}")
    print("-" * 40)

    for pct in train_sizes_pct:
        n_samples = int(len(X_full) * pct)
        X_train = X_full[:n_samples]
        y_train = y_full[:n_samples]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = create_model()
        model.fit(X_train_s, y_train)

        train_f1 = f1_score(y_train, model.predict(X_train_s), average='macro', zero_division=0)
        val_f1 = f1_score(y_val, model.predict(X_val_s), average='macro', zero_division=0)

        results['sizes'].append(n_samples)
        results['train_scores'].append(train_f1)
        results['val_scores'].append(val_f1)

        print(f"{n_samples:>8,} {train_f1:>10.4f} {val_f1:>10.4f} {train_f1-val_f1:>8.4f}")

        del model
        cleanup()

    # Plot
    os.makedirs(FIGURES_PATH, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results['sizes'], results['train_scores'], 'o-', label='Train F1')
    ax.plot(results['sizes'], results['val_scores'], 'o-', label='Validation F1')
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('F1 Score (macro)')
    ax.set_title('Learning Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'learning_curve_light.png'), dpi=100)
    plt.close()

    final_gap = results['train_scores'][-1] - results['val_scores'][-1]
    print(f"\n  Plot saved: reports/figures/learning_curve_light.png")

    if final_gap > 0.05:
        print(f"\n  ⚠️  WARNING: Gap at full data ({final_gap:.4f}) indicates overfitting")
    else:
        print(f"\n  ✓ OK: Converging train/val scores")

    results['final_gap'] = final_gap
    return results


def test_cv_sequential(df, feature_cols):
    """
    Test 4: Cross-Validation (Sequential)

    Manual K-fold CV running one fold at a time to minimize memory.
    """
    print("\n" + "="*60)
    print(f"TEST 4: {N_FOLDS}-FOLD CROSS-VALIDATION (Sequential)")
    print("="*60)

    X, y, _ = prepare_data(df, feature_cols)

    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    scores = []

    print(f"\n  Running {N_FOLDS} folds sequentially...")

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = create_model()
        model.fit(X_train_s, y_train)

        f1 = f1_score(y_val, model.predict(X_val_s), average='macro', zero_division=0)
        scores.append(f1)
        print(f"    Fold {fold}: {f1:.4f}")

        del model, X_train_s, X_val_s
        cleanup()

    scores = np.array(scores)
    print(f"\n  Mean:  {scores.mean():.4f}")
    print(f"  Std:   {scores.std():.4f}")

    if scores.std() > 0.05:
        print(f"\n  ⚠️  WARNING: High variance across folds")
    else:
        print(f"\n  ✓ OK: Consistent performance across folds")

    return {'scores': scores.tolist(), 'mean': scores.mean(), 'std': scores.std()}


def test_shuffle(df, feature_cols):
    """
    Test 5: Shuffle Test (Sanity Check)

    Train on random labels - should perform near chance level.
    """
    print("\n" + "="*60)
    print("TEST 5: SHUFFLE TEST (Sanity Check)")
    print("="*60)

    X, y, _ = prepare_data(df, feature_cols, sample_size=10_000)

    n_classes = len(np.unique(y))
    expected_random = 1.0 / n_classes

    # Shuffle labels
    np.random.seed(123)
    y_shuffled = np.random.permutation(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_shuffled, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = create_model()
    model.fit(X_train_s, y_train)

    acc = accuracy_score(y_test, model.predict(X_test_s))

    print(f"\n  Classes:         {n_classes}")
    print(f"  Expected random: {expected_random:.4f} ({expected_random*100:.1f}%)")
    print(f"  Actual accuracy: {acc:.4f} ({acc*100:.1f}%)")

    if acc > expected_random * 2:
        print(f"\n  ⚠️  WARNING: Much higher than random - possible data leakage!")
    else:
        print(f"\n  ✓ OK: Near random as expected (sanity check passed)")

    del model
    cleanup()

    return {'n_classes': n_classes, 'expected': expected_random, 'actual': acc}


def main():
    """Run all tests sequentially."""
    print("\n" + "#"*60)
    print("# LIGHTWEIGHT OVERFITTING TEST SUITE")
    print("# (Memory-safe sequential execution)")
    print("#"*60)

    df = load_data()
    feature_cols = get_feature_cols(df)
    print(f"  Using {len(feature_cols)} features")

    results = {}

    # Run tests one at a time with cleanup
    results['train_test_gap'] = test_train_test_gap(df, feature_cols)
    cleanup()

    results['temporal_split'] = test_temporal_split(df, feature_cols)
    cleanup()

    results['learning_curve'] = test_learning_curve(df, feature_cols)
    cleanup()

    results['cross_validation'] = test_cv_sequential(df, feature_cols)
    cleanup()

    results['shuffle_test'] = test_shuffle(df, feature_cols)
    cleanup()

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    issues = []
    if results['train_test_gap']['gap'] > 0.1:
        issues.append("Large train/test gap")
    if results['temporal_split']['gap'] > 0.1:
        issues.append("Poor temporal generalization")
    if results['learning_curve']['final_gap'] > 0.05:
        issues.append("Learning curve gap")
    if results['cross_validation']['std'] > 0.05:
        issues.append("High CV variance")
    if results['shuffle_test']['actual'] > results['shuffle_test']['expected'] * 2:
        issues.append("Shuffle test failed")

    risk = 'HIGH' if len(issues) >= 2 else ('MEDIUM' if issues else 'LOW')

    print(f"\n  Overfitting Risk: {risk}")
    if issues:
        print("  Issues:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("  No significant overfitting detected.")

    # Save report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report = {
        'generated_at': datetime.now().isoformat(),
        'risk_level': risk,
        'issues': issues,
        'tests': results
    }
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n  Report saved: {REPORT_PATH}")
    print("\n" + "#"*60)
    print("# COMPLETE")
    print("#"*60 + "\n")


if __name__ == '__main__':
    main()
