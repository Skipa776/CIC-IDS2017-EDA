"""Model evaluation utilities."""

import time
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    precision_recall_fscore_support,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)


def evaluate_binary_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate binary classification model.

    Args:
        model: Trained binary classifier
        X_test: Test features (scaled)
        y_test: True binary labels
        threshold: Classification threshold

    Returns:
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', zero_division=0
    )
    pr_auc = average_precision_score(y_test, y_proba)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'pr_auc': float(pr_auc),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
        'test_size': len(y_test),
        'positive_rate': float(y_test.mean()),
    }


def threshold_at_fpr(y_true, scores, target_fpr):
    """Lowest score threshold whose FPR on benign stays at or below target_fpr.

    Returns (threshold, actual_fpr, recall).
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=False)
    i = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(thresholds[i]), float(fpr[i]), float(tpr[i])


def threshold_on_benign(scores, target_fpr):
    """Choose a >= threshold on validation benign scores only, respecting ties.

    The empirical validation FPR is bounded; the FPR on future traffic is not.
    No attack scores or outer-test labels are needed to choose this threshold.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Expected a nonempty vector of finite benign scores")
    if not 0 <= target_fpr < 1:
        raise ValueError("target_fpr must be in [0, 1)")
    allowed = int(np.floor(target_fpr * len(scores)))
    # Exclude the first score that cannot fit within the budget. nextafter
    # excludes all ties at that boundary for the >= decision rule.
    boundary = np.partition(scores, len(scores) - allowed - 1)[len(scores) - allowed - 1]
    return float(np.nextafter(boundary, np.inf))


def evaluate_multiclass_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    label_names: list = None
) -> Dict[str, Any]:
    """
    Evaluate multi-class classification model.

    Args:
        model: Trained multi-class classifier
        X_test: Test features (scaled)
        y_test: True multi-class labels (encoded)
        label_names: List of class names for reporting

    Returns:
        Dictionary of evaluation metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)

    # Overall metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    # Per-class metrics. Pass the present class codes explicitly: y_test may
    # not contain every code (e.g., layer 2 sees attacks only, never BENIGN),
    # and positional indexing into label_names would shift every name by one.
    present_classes = np.unique(np.concatenate([y_test, y_pred]))
    precision_per_class, recall_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_test, y_pred, labels=present_classes, average=None, zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=present_classes)

    # Build per-class metrics dict keyed by the true class code's name
    per_class = {}
    for i, code in enumerate(present_classes):
        class_name = label_names[int(code)] if label_names else str(code)
        per_class[class_name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1_score': float(f1_per_class[i]),
            'support': int(support[i]),
        }

    return {
        'macro_precision': float(precision_macro),
        'macro_recall': float(recall_macro),
        'macro_f1': float(f1_macro),
        'weighted_precision': float(precision_weighted),
        'weighted_recall': float(recall_weighted),
        'weighted_f1': float(f1_weighted),
        'per_class': per_class,
        'confusion_matrix': cm.tolist(),
        'test_size': len(y_test),
        'num_classes': len(np.unique(y_test)),
    }


def get_inference_time(
    model: Any,
    X_sample: np.ndarray,
    n_iterations: int = 1000
) -> Dict[str, float]:
    """
    Measure model inference time.

    Args:
        model: Trained model
        X_sample: Single sample for inference (shape: (1, n_features))
        n_iterations: Number of iterations for timing

    Returns:
        Dictionary with timing statistics in milliseconds
    """
    if X_sample.ndim == 1:
        X_sample = X_sample.reshape(1, -1)

    times = []

    # Warmup
    for _ in range(10):
        model.predict_proba(X_sample)

    # Timed iterations
    for _ in range(n_iterations):
        start = time.perf_counter()
        model.predict_proba(X_sample)
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    times = np.array(times)

    return {
        'mean_ms': float(times.mean()),
        'std_ms': float(times.std()),
        'min_ms': float(times.min()),
        'max_ms': float(times.max()),
        'p50_ms': float(np.percentile(times, 50)),
        'p95_ms': float(np.percentile(times, 95)),
        'p99_ms': float(np.percentile(times, 99)),
    }


def print_classification_report(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    label_names: list = None
) -> str:
    """
    Generate and print a formatted classification report.

    Args:
        y_test: True labels
        y_pred: Predicted labels
        label_names: List of class names

    Returns:
        Classification report string
    """
    report = classification_report(
        y_test, y_pred,
        target_names=label_names,
        zero_division=0
    )
    print(report)
    return report
