"""Data loading utilities for CICIDS2017 dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# All attack labels in the dataset
ATTACK_LABELS = [
    "BENIGN",
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
    "Web Attack - Brute Force",
    "Web Attack - Sql Injection",
    "Web Attack - XSS",
]


def load_processed_data(
    parquet_path: Optional[Path] = None,
    sample_size: Optional[int] = None,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Load cleaned CICIDS2017 data from parquet file.

    Args:
        parquet_path: Path to parquet file. Defaults to data/processed/cicids2017_clean.parquet
        sample_size: If provided, randomly sample this many rows
        random_state: Random seed for sampling

    Returns:
        DataFrame with cleaned CICIDS2017 data
    """
    if parquet_path is None:
        parquet_path = DATA_DIR / "processed" / "cicids2017_clean.parquet"

    df = pd.read_parquet(parquet_path)

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Get numeric feature columns (excluding Label and Meta_source).

    Args:
        df: DataFrame with CICIDS2017 data

    Returns:
        List of feature column names
    """
    exclude = {'Label', 'Meta_source', 'Attack_Family', 'Day', 'Is_Attack'}
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype in ['int64', 'float64']
    ]


def prepare_binary_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Convert Label to binary classification target.

    Args:
        df: DataFrame with 'Label' column

    Returns:
        numpy array with 0 (benign) or 1 (attack)
    """
    return (df['Label'] != 'BENIGN').astype(int).values


def prepare_multiclass_labels(df: pd.DataFrame) -> Tuple[np.ndarray, Dict[int, str]]:
    """
    Convert Label to integer codes with mapping dictionary.

    Args:
        df: DataFrame with 'Label' column

    Returns:
        Tuple of (encoded labels array, {code: label_name} mapping)
    """
    # Filter to attacks only (exclude BENIGN for multi-class)
    labels = df['Label'].astype('category')
    codes = labels.cat.codes.values
    mapping = {i: cat for i, cat in enumerate(labels.cat.categories)}

    return codes, mapping


def get_attack_labels() -> List[str]:
    """Get list of all attack labels (excluding BENIGN)."""
    return [label for label in ATTACK_LABELS if label != "BENIGN"]


def prepare_features(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None
) -> np.ndarray:
    """
    Prepare feature matrix from DataFrame.

    Args:
        df: DataFrame with features
        feature_cols: List of column names to use. If None, auto-detect.

    Returns:
        numpy array of features with inf/nan handled
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    X = df[feature_cols].values.astype(np.float64)

    # Handle inf and nan values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X


def stratified_sample(
    df: pd.DataFrame,
    n_per_class: int = 10000,
    label_col: str = 'Label',
    random_state: int = 42
) -> pd.DataFrame:
    """
    Sample data with stratification by label.

    Args:
        df: DataFrame to sample from
        n_per_class: Maximum samples per class
        label_col: Column to stratify by
        random_state: Random seed

    Returns:
        Sampled DataFrame
    """
    return df.groupby(label_col).apply(
        lambda x: x.sample(min(len(x), n_per_class), random_state=random_state)
    ).reset_index(drop=True)
