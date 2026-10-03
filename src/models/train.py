"""Model training utilities for the layered IDS."""

import json
import joblib
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Try to import LightGBM, fall back to sklearn if not available
try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import RandomForestClassifier

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODEL_DIR = PROJECT_ROOT / "models"


def train_layer1_binary(
    X_train: np.ndarray,
    y_train: np.ndarray,
    use_lightgbm: bool = True,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Any:
    """
    Train Layer 1 binary classifier (benign vs attack).

    Optimized for fast inference with good accuracy.

    Args:
        X_train: Training features (scaled)
        y_train: Binary labels (0=benign, 1=attack)
        use_lightgbm: Use LightGBM if available (faster inference)
        random_state: Random seed
        n_jobs: Worker threads for training

    Returns:
        Trained classifier
    """
    if use_lightgbm and HAS_LIGHTGBM:
        model = LGBMClassifier(
            objective='binary',
            boosting_type='gbdt',
            num_leaves=31,
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            class_weight='balanced',
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=-1,
        )
    else:
        # Fallback to Logistic Regression (very fast, good baseline)
        model = LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    model.fit(X_train, y_train)
    return model


def train_layer2_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int = 15,
    use_lightgbm: bool = True,
    random_state: int = 42
) -> Any:
    """
    Train Layer 2 multi-class classifier (attack type identification).

    Args:
        X_train: Training features (scaled)
        y_train: Multi-class labels (encoded integers)
        num_classes: Number of classes
        use_lightgbm: Use LightGBM if available
        random_state: Random seed

    Returns:
        Trained classifier
    """
    if use_lightgbm and HAS_LIGHTGBM:
        model = LGBMClassifier(
            objective='multiclass',
            num_class=num_classes,
            boosting_type='gbdt',
            num_leaves=63,
            max_depth=8,
            learning_rate=0.05,
            n_estimators=200,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )
    else:
        # Fallback to Random Forest
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            class_weight='balanced',
            n_jobs=-1,
            random_state=random_state,
        )

    model.fit(X_train, y_train)
    return model


def save_models(
    layer1_model: Any,
    layer2_model: Any,
    scaler: StandardScaler,
    feature_columns: List[str],
    label_mapping: Dict[int, str],
    metrics: Dict[str, Any],
    output_dir: Optional[Path] = None,
    dataset: Optional[str] = None
) -> Path:
    """
    Save all model artifacts to disk.

    Args:
        layer1_model: Trained binary classifier
        layer2_model: Trained multi-class classifier
        scaler: Fitted StandardScaler
        feature_columns: Ordered list of feature names
        label_mapping: {code: label_name} mapping for multi-class
        metrics: Dictionary of evaluation metrics
        output_dir: Directory to save models. Defaults to models/
        dataset: Name of the training dataset file, recorded in metadata

    Returns:
        Path to output directory
    """
    if output_dir is None:
        output_dir = MODEL_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save models
    joblib.dump(layer1_model, output_dir / 'layer1_binary_classifier.joblib')
    joblib.dump(layer2_model, output_dir / 'layer2_multiclass_classifier.joblib')
    joblib.dump(scaler, output_dir / 'feature_scaler.joblib')

    # Save feature columns
    with open(output_dir / 'feature_columns.json', 'w') as f:
        json.dump(feature_columns, f, indent=2)

    # Save label mapping (convert int keys to strings for JSON)
    with open(output_dir / 'label_mapping.json', 'w') as f:
        json.dump({str(k): v for k, v in label_mapping.items()}, f, indent=2)

    # Save metadata
    metadata = {
        'version': '2.0.0',
        'dataset': dataset,
        'trained_at': datetime.now().isoformat(),
        'layer1_type': type(layer1_model).__name__,
        'layer2_type': type(layer2_model).__name__,
        'feature_count': len(feature_columns),
        'num_classes': len(label_mapping),
        'metrics': metrics,
        'has_lightgbm': HAS_LIGHTGBM,
    }
    with open(output_dir / 'model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Models saved to {output_dir}")
    return output_dir


def load_models(
    model_dir: Optional[Path] = None
) -> Tuple[Any, Any, StandardScaler, List[str], Dict[int, str], Dict[str, Any]]:
    """
    Load all model artifacts from disk.

    Args:
        model_dir: Directory containing saved models. Defaults to models/

    Returns:
        Tuple of (layer1_model, layer2_model, scaler, feature_columns, label_mapping, metadata)
    """
    if model_dir is None:
        model_dir = MODEL_DIR

    # Load models
    layer1 = joblib.load(model_dir / 'layer1_binary_classifier.joblib')
    layer2 = joblib.load(model_dir / 'layer2_multiclass_classifier.joblib')
    scaler = joblib.load(model_dir / 'feature_scaler.joblib')

    # Load feature columns
    with open(model_dir / 'feature_columns.json', 'r') as f:
        feature_columns = json.load(f)

    # Load label mapping (convert string keys back to int)
    with open(model_dir / 'label_mapping.json', 'r') as f:
        label_mapping_str = json.load(f)
        label_mapping = {int(k): v for k, v in label_mapping_str.items()}

    # Load metadata
    with open(model_dir / 'model_metadata.json', 'r') as f:
        metadata = json.load(f)

    return layer1, layer2, scaler, feature_columns, label_mapping, metadata
