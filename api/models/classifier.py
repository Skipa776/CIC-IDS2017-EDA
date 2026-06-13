"""Model loading and classifier wrapper."""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from api.config import settings


class IDSClassifier:
    """Wrapper for the layered IDS classification models."""

    def __init__(self, model_dir: Optional[Path] = None):
        """
        Initialize the classifier.

        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = model_dir or settings.model_dir
        self.layer1_model = None
        self.layer2_model = None
        self.scaler = None
        self.feature_columns = None
        self.label_mapping = None
        self.metadata = None
        self._loaded = False

    def load_models(self) -> None:
        """Load all model artifacts from disk."""
        if self._loaded:
            return

        model_dir = self.model_dir

        # Load models
        self.layer1_model = joblib.load(model_dir / 'layer1_binary_classifier.joblib')
        self.layer2_model = joblib.load(model_dir / 'layer2_multiclass_classifier.joblib')
        self.scaler = joblib.load(model_dir / 'feature_scaler.joblib')

        # Load feature columns
        with open(model_dir / 'feature_columns.json', 'r') as f:
            self.feature_columns = json.load(f)

        # Load label mapping
        with open(model_dir / 'label_mapping.json', 'r') as f:
            label_mapping_str = json.load(f)
            self.label_mapping = {int(k): v for k, v in label_mapping_str.items()}

        # Load metadata
        with open(model_dir / 'model_metadata.json', 'r') as f:
            self.metadata = json.load(f)

        self._loaded = True

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._loaded

    def features_to_array(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert feature dictionary to numpy array in correct order.

        Args:
            features: Dictionary of feature name -> value

        Returns:
            numpy array of features
        """
        # Map API field names to model feature names
        field_mapping = {
            'destination_port': 'Destination Port',
            'flow_duration': 'Flow Duration',
            'flow_bytes_per_sec': 'Flow Bytes/s',
            'flow_packets_per_sec': 'Flow Packets/s',
            'total_fwd_packets': 'Total Fwd Packets',
            'total_backward_packets': 'Total Backward Packets',
            'fwd_packet_length_mean': 'Fwd Packet Length Mean',
            'bwd_packet_length_mean': 'Bwd Packet Length Mean',
            'packet_length_mean': 'Packet Length Mean',
            'packet_length_std': 'Packet Length Std',
            'syn_flag_count': 'SYN Flag Count',
            'fin_flag_count': 'FIN Flag Count',
            'rst_flag_count': 'RST Flag Count',
            'ack_flag_count': 'ACK Flag Count',
            'init_win_bytes_forward': 'Init_Win_bytes_forward',
            'init_win_bytes_backward': 'Init_Win_bytes_backward',
            'flow_iat_mean': 'Flow IAT Mean',
            'flow_iat_std': 'Flow IAT Std',
            'fwd_iat_mean': 'Fwd IAT Mean',
            'bwd_iat_mean': 'Bwd IAT Mean',
        }

        # Build feature array in correct order
        feature_array = []
        for col in self.feature_columns:
            # Find the API field that maps to this column
            api_field = None
            for field, model_col in field_mapping.items():
                if model_col == col:
                    api_field = field
                    break

            if api_field and api_field in features:
                feature_array.append(float(features[api_field]))
            else:
                # Default to 0 if feature not provided
                feature_array.append(0.0)

        return np.array(feature_array).reshape(1, -1)

    def predict_binary(self, features: np.ndarray) -> Tuple[bool, float]:
        """
        Run Layer 1 binary classification.

        Args:
            features: Scaled feature array

        Returns:
            Tuple of (is_attack, probability)
        """
        proba = self.layer1_model.predict_proba(features)[0]
        attack_prob = proba[1]
        is_attack = attack_prob >= settings.binary_threshold

        return is_attack, float(attack_prob)

    def predict_multiclass(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Run Layer 2 multi-class classification.

        Args:
            features: Scaled feature array

        Returns:
            Tuple of (attack_type, probability)
        """
        proba = self.layer2_model.predict_proba(features)[0]
        pred_class = int(np.argmax(proba))
        pred_prob = float(proba[pred_class])

        attack_type = self.label_mapping.get(pred_class, f"Unknown-{pred_class}")

        return attack_type, pred_prob

    def classify(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run full classification pipeline.

        Args:
            features: Dictionary of feature values

        Returns:
            Classification result dictionary
        """
        import time

        start_time = time.perf_counter()

        # Convert to array and scale
        feature_array = self.features_to_array(features)
        scaled_features = self.scaler.transform(feature_array)

        # Layer 1: Binary classification
        is_attack, attack_prob = self.predict_binary(scaled_features)

        attack_type = None
        attack_type_prob = None

        # Layer 2: Multi-class (only if attack)
        if is_attack:
            attack_type, attack_type_prob = self.predict_multiclass(scaled_features)

        inference_time = (time.perf_counter() - start_time) * 1000

        return {
            'is_attack': is_attack,
            'attack_probability': attack_prob,
            'attack_type': attack_type,
            'attack_type_probability': attack_type_prob,
            'inference_time_ms': inference_time,
        }


# Global classifier instance
classifier = IDSClassifier()
