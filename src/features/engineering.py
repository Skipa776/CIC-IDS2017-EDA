"""Feature engineering and selection utilities."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from typing import List, Tuple, Optional

# Top 20 features selected for fast inference
# These were identified through feature importance analysis
FAST_FEATURES = [
    'Destination Port',
    'Flow Duration',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Packet Length Mean',
    'Packet Length Std',
    'SYN Flag Count',
    'FIN Flag Count',
    'RST Flag Count',
    'ACK Flag Count',
    'Init_Win_bytes_forward',
    'Init_Win_bytes_backward',
    'Flow IAT Mean',
    'Flow IAT Std',
    'Fwd IAT Mean',
    'Bwd IAT Mean',
]


def create_scaler(X_train: np.ndarray) -> StandardScaler:
    """
    Create and fit a StandardScaler on training data.

    Args:
        X_train: Training feature matrix

    Returns:
        Fitted StandardScaler
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def select_top_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    k: int = 20
) -> Tuple[np.ndarray, List[str], SelectKBest]:
    """
    Select top k features using ANOVA F-test.

    Args:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature column names
        k: Number of features to select

    Returns:
        Tuple of (selected features array, selected feature names, fitted selector)
    """
    selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)

    # Get selected feature names
    selected_mask = selector.get_support()
    selected_names = [f for f, s in zip(feature_names, selected_mask) if s]

    return X_selected, selected_names, selector


def get_feature_selector(
    feature_names: List[str],
    selected_features: Optional[List[str]] = None
) -> List[int]:
    """
    Get indices of selected features from full feature list.

    Args:
        feature_names: Full list of feature names
        selected_features: Features to select. Defaults to FAST_FEATURES.

    Returns:
        List of indices for selected features
    """
    if selected_features is None:
        selected_features = FAST_FEATURES

    indices = []
    for feat in selected_features:
        if feat in feature_names:
            indices.append(feature_names.index(feat))

    return indices


def filter_features_by_name(
    X: np.ndarray,
    feature_names: List[str],
    selected_features: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Filter feature matrix to only include selected features.

    Args:
        X: Full feature matrix
        feature_names: Full list of feature names
        selected_features: Features to keep. Defaults to FAST_FEATURES.

    Returns:
        Tuple of (filtered feature matrix, filtered feature names)
    """
    if selected_features is None:
        selected_features = FAST_FEATURES

    indices = get_feature_selector(feature_names, selected_features)
    filtered_names = [feature_names[i] for i in indices]

    return X[:, indices], filtered_names


def validate_features(
    feature_names: List[str],
    required_features: Optional[List[str]] = None
) -> bool:
    """
    Validate that all required features are present.

    Args:
        feature_names: Available feature names
        required_features: Features that must be present. Defaults to FAST_FEATURES.

    Returns:
        True if all required features are present
    """
    if required_features is None:
        required_features = FAST_FEATURES

    missing = set(required_features) - set(feature_names)
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return True
