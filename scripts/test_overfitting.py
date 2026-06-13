#!/usr/bin/env python3
"""
Overfitting Test Suite for CICIDS2017 Attack Classifier

This script evaluates potential overfitting in the trained RandomForest model
through multiple diagnostic tests:

1. Temporal Split Validation - Train on early days, test on later days
2. Cross-Validation - Measure variance across folds
3. Learning Curve Analysis - Compare train vs validation error
4. Feature Ablation - Test impact of removing top features
5. Shuffle Test - Sanity check with randomized labels

Usage:
    python scripts/test_overfitting.py

Output:
    - Console report with all test results
    - models/overfitting_report.json with detailed metrics
    - reports/figures/learning_curve.png
"""

import os
import json
import warnings
from datetime import datetime

import numpy as np
import polars as pl
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    learning_curve,
    cross_val_score,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
)

warnings.filterwarnings('ignore')

# Resource limits for MacBook M3 Pro (18GB RAM)
# Avoid nested parallelism: use n_jobs on outer CV loops, not inner estimators
N_JOBS_CV = 6  # Parallel CV folds (leave cores for system)
N_JOBS_EST = 1  # Single-threaded estimators (prevents nested parallelism)
SAMPLE_SIZE_LARGE = 40000  # For most tests
SAMPLE_SIZE_SMALL = 25000  # For learning curve (most memory-intensive)

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, 'data/processed/cicids2017_clean.parquet')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models/rf_multiclass_attack_classifier.joblib')
REPORT_PATH = os.path.join(PROJECT_ROOT, 'models/overfitting_report.json')
FIGURES_PATH = os.path.join(PROJECT_ROOT, 'reports/figures')


def load_model_and_data():
    """Load the trained model and dataset."""
    print("Loading model and data...")

    # Load model artifacts
    if os.path.exists(MODEL_PATH):
        model_artifacts = joblib.load(MODEL_PATH)
        print(f"  Model loaded from {MODEL_PATH}")
    else:
        print(f"  WARNING: Model not found at {MODEL_PATH}")
        print("  Run the attack_types.ipynb notebook first to train and save the model.")
        model_artifacts = None

    # Load data
    df = pl.read_parquet(DATA_PATH)
    print(f"  Data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

    return model_artifacts, df


def prepare_features(df, feature_cols):
    """Prepare feature matrix and labels from dataframe."""
    X = df.select(feature_cols).to_numpy()
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    y = df['Label'].to_numpy()
    return X, y


def test_temporal_split(df, feature_cols):
    """
    Test 1: Temporal Split Validation

    Train on Monday-Thursday data, test on Friday data.
    This tests if the model generalizes across time.
    """
    print("\n" + "="*80)
    print("TEST 1: TEMPORAL SPLIT VALIDATION")
    print("="*80)
    print("Training on Monday-Thursday, testing on Friday\n")

    # Define temporal groups based on Meta_source
    friday_sources = [
        'Friday-WorkingHours-Morning',
        'Friday-WorkingHours-Afternoon-DDos',
        'Friday-WorkingHours-Afternoon-PortScan'
    ]

    # Split by time period
    df_train_temporal = df.filter(~pl.col('Meta_source').is_in(friday_sources))
    df_test_temporal = df.filter(pl.col('Meta_source').is_in(friday_sources))

    print(f"Train set (Mon-Thu): {df_train_temporal.height:,} samples")
    print(f"Test set (Friday): {df_test_temporal.height:,} samples")

    # Sample for computational efficiency
    sample_size = SAMPLE_SIZE_LARGE
    if df_train_temporal.height > sample_size:
        df_train_temporal = df_train_temporal.sample(n=sample_size, seed=42)
    if df_test_temporal.height > sample_size:
        df_test_temporal = df_test_temporal.sample(n=sample_size, seed=42)

    X_train, y_train = prepare_features(df_train_temporal, feature_cols)
    X_test, y_test = prepare_features(df_test_temporal, feature_cols)

    # Train-test split on train data for random comparison
    le = LabelEncoder()
    le.fit(np.concatenate([y_train, y_test]))
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20,
        class_weight='balanced', n_jobs=N_JOBS_EST, random_state=42
    )
    rf.fit(X_train_scaled, y_train_enc)

    # Evaluate
    y_pred_train = rf.predict(X_train_scaled)
    y_pred_test = rf.predict(X_test_scaled)

    train_f1 = f1_score(y_train_enc, y_pred_train, average='macro', zero_division=0)
    test_f1 = f1_score(y_test_enc, y_pred_test, average='macro', zero_division=0)
    train_acc = accuracy_score(y_train_enc, y_pred_train)
    test_acc = accuracy_score(y_test_enc, y_pred_test)

    print(f"\nResults:")
    print(f"  Train F1 (macro): {train_f1:.4f}")
    print(f"  Test F1 (macro):  {test_f1:.4f}")
    print(f"  F1 Drop:          {train_f1 - test_f1:.4f}")
    print(f"\n  Train Accuracy:   {train_acc:.4f}")
    print(f"  Test Accuracy:    {test_acc:.4f}")
    print(f"  Accuracy Drop:    {train_acc - test_acc:.4f}")

    gap = train_f1 - test_f1
    if gap > 0.1:
        print(f"\n  WARNING: Large performance gap ({gap:.2f}) indicates temporal overfitting!")
    elif gap > 0.05:
        print(f"\n  CAUTION: Moderate performance gap ({gap:.2f}) suggests some temporal bias.")
    else:
        print(f"\n  OK: Small performance gap ({gap:.2f}) - model generalizes across time.")

    return {
        'train_f1': train_f1,
        'test_f1': test_f1,
        'f1_gap': train_f1 - test_f1,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_samples': df_train_temporal.height,
        'test_samples': df_test_temporal.height
    }


def test_cross_validation(df, feature_cols):
    """
    Test 2: Cross-Validation with Stratified K-Fold

    5-fold cross-validation to measure variance in performance.
    High variance indicates overfitting to specific data splits.
    """
    print("\n" + "="*80)
    print("TEST 2: STRATIFIED 5-FOLD CROSS-VALIDATION")
    print("="*80)
    print("Measuring variance across folds\n")

    # Sample for efficiency
    sample_size = SAMPLE_SIZE_LARGE
    df_sample = df.sample(n=min(sample_size, df.height), seed=42)

    X, y = prepare_features(df_sample, feature_cols)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20,
        class_weight='balanced', n_jobs=N_JOBS_EST, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(rf, X_scaled, y_enc, cv=cv, scoring='f1_macro', n_jobs=N_JOBS_CV)

    print(f"Results (F1 macro):")
    for i, score in enumerate(scores):
        print(f"  Fold {i+1}: {score:.4f}")

    print(f"\n  Mean:  {scores.mean():.4f}")
    print(f"  Std:   {scores.std():.4f}")
    print(f"  Range: [{scores.min():.4f}, {scores.max():.4f}]")

    if scores.std() > 0.05:
        print(f"\n  WARNING: High variance ({scores.std():.4f}) suggests overfitting to specific splits!")
    else:
        print(f"\n  OK: Low variance ({scores.std():.4f}) - consistent performance across folds.")

    return {
        'fold_scores': scores.tolist(),
        'mean': scores.mean(),
        'std': scores.std(),
        'min': scores.min(),
        'max': scores.max()
    }


def test_learning_curve(df, feature_cols):
    """
    Test 3: Learning Curve Analysis

    Train with varying amounts of data (10%, 25%, 50%, 75%, 100%).
    Plot train vs validation error to identify overfitting.
    """
    print("\n" + "="*80)
    print("TEST 3: LEARNING CURVE ANALYSIS")
    print("="*80)
    print("Training with varying data sizes\n")

    # Sample for efficiency (learning curve is most memory-intensive)
    sample_size = SAMPLE_SIZE_SMALL
    df_sample = df.sample(n=min(sample_size, df.height), seed=42)

    X, y = prepare_features(df_sample, feature_cols)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Learning curve
    rf = RandomForestClassifier(
        n_estimators=50, max_depth=15,  # Smaller for speed
        class_weight='balanced', n_jobs=N_JOBS_EST, random_state=42
    )

    train_sizes_abs, train_scores, val_scores = learning_curve(
        rf, X_scaled, y_enc,
        train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
        cv=3,
        scoring='f1_macro',
        n_jobs=N_JOBS_CV,
        random_state=42
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    print(f"{'Size':>10} {'Train F1':>12} {'Val F1':>12} {'Gap':>10}")
    print("-" * 46)
    for size, tm, vm in zip(train_sizes_abs, train_mean, val_mean):
        print(f"{size:>10,} {tm:>12.4f} {vm:>12.4f} {tm-vm:>10.4f}")

    # Plot
    os.makedirs(FIGURES_PATH, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.2, color='orange')
    ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training F1')
    ax.plot(train_sizes_abs, val_mean, 'o-', color='orange', label='Validation F1')

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('F1 Score (macro)')
    ax.set_title('Learning Curve: Train vs Validation Performance')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, 'learning_curve.png'), dpi=150)
    plt.close()
    print(f"\n  Plot saved to: reports/figures/learning_curve.png")

    final_gap = train_mean[-1] - val_mean[-1]
    if final_gap > 0.05:
        print(f"\n  WARNING: Gap at full data ({final_gap:.4f}) indicates overfitting!")
    else:
        print(f"\n  OK: Small gap at full data ({final_gap:.4f}).")

    return {
        'train_sizes': train_sizes_abs.tolist(),
        'train_scores': train_mean.tolist(),
        'val_scores': val_mean.tolist(),
        'final_gap': final_gap
    }


def test_feature_ablation(df, feature_cols, model_artifacts):
    """
    Test 4: Feature Ablation

    Remove top features (Destination Port, Init_Win_bytes_backward)
    and re-evaluate to see if model over-relies on potentially leaky features.
    """
    print("\n" + "="*80)
    print("TEST 4: FEATURE ABLATION")
    print("="*80)
    print("Testing impact of removing top features\n")

    # Features to ablate (potentially problematic)
    features_to_remove = ['Destination Port', 'Init_Win_bytes_backward']

    # Sample for efficiency
    sample_size = SAMPLE_SIZE_LARGE
    df_sample = df.sample(n=min(sample_size, df.height), seed=42)

    results = {}

    # Baseline with all features
    X_full, y = prepare_features(df_sample, feature_cols)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y_enc, test_size=0.2, stratify=y_enc, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=20,
        class_weight='balanced', n_jobs=N_JOBS_CV, random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)
    baseline_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"Baseline (all {len(feature_cols)} features):")
    print(f"  F1 macro: {baseline_f1:.4f}\n")
    results['baseline'] = {'f1': baseline_f1, 'n_features': len(feature_cols)}

    # Ablate features one by one
    for feat_to_remove in features_to_remove:
        if feat_to_remove not in feature_cols:
            print(f"  Feature '{feat_to_remove}' not found, skipping")
            continue

        # Create reduced feature set
        reduced_cols = [f for f in feature_cols if f != feat_to_remove]
        feat_idx = [feature_cols.index(f) for f in reduced_cols]

        X_reduced = X_full[:, feat_idx]

        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_reduced, y_enc, test_size=0.2, stratify=y_enc, random_state=42
        )

        scaler_r = StandardScaler()
        X_train_r_scaled = scaler_r.fit_transform(X_train_r)
        X_test_r_scaled = scaler_r.transform(X_test_r)

        rf_r = RandomForestClassifier(
            n_estimators=100, max_depth=20,
            class_weight='balanced', n_jobs=N_JOBS_CV, random_state=42
        )
        rf_r.fit(X_train_r_scaled, y_train_r)
        y_pred_r = rf_r.predict(X_test_r_scaled)
        ablated_f1 = f1_score(y_test_r, y_pred_r, average='macro', zero_division=0)

        drop = baseline_f1 - ablated_f1
        print(f"Without '{feat_to_remove}':")
        print(f"  F1 macro: {ablated_f1:.4f}")
        print(f"  Drop:     {drop:.4f} ({drop/baseline_f1*100:.1f}%)\n")

        results[f'without_{feat_to_remove}'] = {
            'f1': ablated_f1,
            'drop': drop,
            'drop_pct': drop/baseline_f1*100
        }

    # Check for over-reliance
    max_drop = max([r.get('drop', 0) for k, r in results.items() if k != 'baseline'])
    if max_drop > 0.1:
        print(f"  WARNING: Large F1 drop ({max_drop:.4f}) when removing features!")
        print(f"  Model may be over-reliant on potentially leaky features.")
    else:
        print(f"  OK: Moderate impact from feature removal (max drop: {max_drop:.4f})")

    return results


def test_label_shuffle(df, feature_cols):
    """
    Test 5: Shuffle Test (Sanity Check)

    Train model on randomly shuffled labels.
    A good model should perform near random chance (~6.7% for 15 classes).
    """
    print("\n" + "="*80)
    print("TEST 5: LABEL SHUFFLE TEST (SANITY CHECK)")
    print("="*80)
    print("Training on randomly shuffled labels\n")

    # Sample for efficiency
    sample_size = SAMPLE_SIZE_SMALL
    df_sample = df.sample(n=min(sample_size, df.height), seed=42)

    X, y = prepare_features(df_sample, feature_cols)
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_classes = len(np.unique(y_enc))
    expected_random = 1.0 / n_classes

    # Shuffle labels
    np.random.seed(123)
    y_shuffled = np.random.permutation(y_enc)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_shuffled, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=15,
        class_weight='balanced', n_jobs=N_JOBS_CV, random_state=42
    )
    rf.fit(X_train_scaled, y_train)
    y_pred = rf.predict(X_test_scaled)

    shuffle_acc = accuracy_score(y_test, y_pred)
    shuffle_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    print(f"Number of classes: {n_classes}")
    print(f"Expected random accuracy: {expected_random:.4f} ({expected_random*100:.1f}%)")
    print(f"\nShuffled label results:")
    print(f"  Accuracy: {shuffle_acc:.4f} ({shuffle_acc*100:.1f}%)")
    print(f"  F1 macro: {shuffle_f1:.4f}")

    if shuffle_acc > expected_random * 2:
        print(f"\n  WARNING: Performance ({shuffle_acc:.4f}) much higher than random!")
        print(f"  This could indicate data leakage or memorization issues.")
    else:
        print(f"\n  OK: Performance near random as expected (sanity check passed).")

    return {
        'n_classes': n_classes,
        'expected_random': expected_random,
        'shuffled_accuracy': shuffle_acc,
        'shuffled_f1': shuffle_f1
    }


def generate_report(results):
    """Generate and save the overfitting analysis report."""
    report = {
        'generated_at': datetime.now().isoformat(),
        'model': 'rf_multiclass_attack_classifier',
        'dataset': 'CICIDS2017',
        'tests': results,
        'summary': {
            'temporal_gap': results.get('temporal_split', {}).get('f1_gap', None),
            'cv_variance': results.get('cross_validation', {}).get('std', None),
            'learning_curve_gap': results.get('learning_curve', {}).get('final_gap', None),
            'shuffle_test_passed': results.get('shuffle_test', {}).get('shuffled_accuracy', 1) < 0.2
        }
    }

    # Determine overall risk
    risk_factors = []
    if report['summary']['temporal_gap'] and report['summary']['temporal_gap'] > 0.1:
        risk_factors.append('High temporal performance gap')
    if report['summary']['cv_variance'] and report['summary']['cv_variance'] > 0.05:
        risk_factors.append('High cross-validation variance')
    if report['summary']['learning_curve_gap'] and report['summary']['learning_curve_gap'] > 0.05:
        risk_factors.append('Large train-validation gap')
    if not report['summary']['shuffle_test_passed']:
        risk_factors.append('Shuffle test indicates potential issues')

    report['overall_risk'] = 'HIGH' if len(risk_factors) >= 2 else ('MEDIUM' if len(risk_factors) >= 1 else 'LOW')
    report['risk_factors'] = risk_factors

    # Save report
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2)

    print("\n" + "="*80)
    print("OVERFITTING ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nOverall Risk Level: {report['overall_risk']}")
    if risk_factors:
        print("\nRisk Factors:")
        for rf in risk_factors:
            print(f"  - {rf}")
    else:
        print("\nNo significant overfitting indicators detected.")

    print(f"\nDetailed report saved to: {REPORT_PATH}")

    return report


def main():
    """Run all overfitting tests."""
    print("\n" + "#"*80)
    print("# OVERFITTING TEST SUITE FOR CICIDS2017 ATTACK CLASSIFIER")
    print("#"*80)

    # Load model and data
    model_artifacts, df = load_model_and_data()

    if model_artifacts is None:
        # Define feature columns manually if model not found
        exclude_cols = ['Label', 'Meta_source', 'Attack_Family']
        feature_cols = [c for c in df.columns if c not in exclude_cols
                       and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    else:
        feature_cols = model_artifacts['feature_columns']

    print(f"Using {len(feature_cols)} features for testing")

    # Run all tests
    results = {}

    results['temporal_split'] = test_temporal_split(df, feature_cols)
    results['cross_validation'] = test_cross_validation(df, feature_cols)
    results['learning_curve'] = test_learning_curve(df, feature_cols)
    results['feature_ablation'] = test_feature_ablation(df, feature_cols, model_artifacts)
    results['shuffle_test'] = test_label_shuffle(df, feature_cols)

    # Generate report
    report = generate_report(results)

    print("\n" + "#"*80)
    print("# TESTING COMPLETE")
    print("#"*80 + "\n")

    return report


if __name__ == '__main__':
    main()
