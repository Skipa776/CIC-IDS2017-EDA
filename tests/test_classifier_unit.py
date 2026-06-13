"""Unit tests for the classifier model wrapper."""

import json
import numpy as np
import pytest

from api.models.classifier import IDSClassifier


@pytest.fixture(scope="module")
def loaded_classifier():
    """Load classifier once for all unit tests."""
    clf = IDSClassifier()
    clf.load_models()
    return clf


def test_feature_array_ordering(loaded_classifier):
    """Feature array must match feature_columns.json order."""
    with open(loaded_classifier.model_dir / "feature_columns.json") as f:
        expected_columns = json.load(f)

    features = {
        "destination_port": 443,
        "flow_duration": 10000,
        "flow_bytes_per_sec": 50000.0,
        "flow_packets_per_sec": 100.0,
        "total_fwd_packets": 5,
        "total_backward_packets": 3,
        "fwd_packet_length_mean": 100.0,
        "bwd_packet_length_mean": 500.0,
        "packet_length_mean": 250.0,
        "packet_length_std": 150.0,
        "syn_flag_count": 1,
        "fin_flag_count": 1,
        "rst_flag_count": 0,
        "ack_flag_count": 5,
        "init_win_bytes_forward": 65535,
        "init_win_bytes_backward": 65535,
        "flow_iat_mean": 1000.0,
        "flow_iat_std": 500.0,
        "fwd_iat_mean": 2000.0,
        "bwd_iat_mean": 1500.0,
    }

    arr = loaded_classifier.features_to_array(features)
    assert arr.shape == (1, len(expected_columns))

    # Verify specific positional values
    field_mapping = {
        "destination_port": "Destination Port",
        "flow_duration": "Flow Duration",
        "flow_bytes_per_sec": "Flow Bytes/s",
        "flow_packets_per_sec": "Flow Packets/s",
        "total_fwd_packets": "Total Fwd Packets",
        "total_backward_packets": "Total Backward Packets",
        "fwd_packet_length_mean": "Fwd Packet Length Mean",
        "bwd_packet_length_mean": "Bwd Packet Length Mean",
        "packet_length_mean": "Packet Length Mean",
        "packet_length_std": "Packet Length Std",
        "syn_flag_count": "SYN Flag Count",
        "fin_flag_count": "FIN Flag Count",
        "rst_flag_count": "RST Flag Count",
        "ack_flag_count": "ACK Flag Count",
        "init_win_bytes_forward": "Init_Win_bytes_forward",
        "init_win_bytes_backward": "Init_Win_bytes_backward",
        "flow_iat_mean": "Flow IAT Mean",
        "flow_iat_std": "Flow IAT Std",
        "fwd_iat_mean": "Fwd IAT Mean",
        "bwd_iat_mean": "Bwd IAT Mean",
    }

    for api_field, model_col in field_mapping.items():
        idx = expected_columns.index(model_col)
        assert arr[0, idx] == float(features[api_field])


def test_api_field_names_map_to_model_columns(loaded_classifier):
    """All 20 feature columns must be reachable from API fields."""
    with open(loaded_classifier.model_dir / "feature_columns.json") as f:
        expected_columns = json.load(f)

    features = {
        "destination_port": 1,
        "flow_duration": 1,
        "flow_bytes_per_sec": 1.0,
        "flow_packets_per_sec": 1.0,
        "total_fwd_packets": 1,
        "total_backward_packets": 1,
        "fwd_packet_length_mean": 1.0,
        "bwd_packet_length_mean": 1.0,
        "packet_length_mean": 1.0,
        "packet_length_std": 1.0,
        "syn_flag_count": 1,
        "fin_flag_count": 1,
        "rst_flag_count": 1,
        "ack_flag_count": 1,
        "init_win_bytes_forward": 1,
        "init_win_bytes_backward": 1,
        "flow_iat_mean": 1.0,
        "flow_iat_std": 1.0,
        "fwd_iat_mean": 1.0,
        "bwd_iat_mean": 1.0,
    }

    arr = loaded_classifier.features_to_array(features)
    # All values should be 1.0 (none defaulted to 0)
    assert np.all(arr == 1.0)


def test_binary_prediction_returns_tuple(loaded_classifier):
    """predict_binary returns (bool-like, float)."""
    features = np.zeros((1, 20))
    scaled = loaded_classifier.scaler.transform(features)
    is_attack, prob = loaded_classifier.predict_binary(scaled)
    assert is_attack in (True, False)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_multiclass_returns_valid_label(loaded_classifier):
    """predict_multiclass returns a label from label_mapping."""
    features = np.zeros((1, 20))
    scaled = loaded_classifier.scaler.transform(features)
    attack_type, prob = loaded_classifier.predict_multiclass(scaled)
    all_labels = set(loaded_classifier.label_mapping.values())
    assert attack_type in all_labels or attack_type.startswith("Unknown-")
    assert 0.0 <= prob <= 1.0


def test_all_zero_input_does_not_crash(loaded_classifier):
    """Classification should not crash on all-zero input."""
    features = {
        "destination_port": 0,
        "flow_duration": 0,
        "flow_bytes_per_sec": 0.0,
        "flow_packets_per_sec": 0.0,
        "total_fwd_packets": 0,
        "total_backward_packets": 0,
        "fwd_packet_length_mean": 0.0,
        "bwd_packet_length_mean": 0.0,
        "packet_length_mean": 0.0,
        "packet_length_std": 0.0,
        "syn_flag_count": 0,
        "fin_flag_count": 0,
        "rst_flag_count": 0,
        "ack_flag_count": 0,
        "init_win_bytes_forward": 0,
        "init_win_bytes_backward": 0,
        "flow_iat_mean": 0.0,
        "flow_iat_std": 0.0,
        "fwd_iat_mean": 0.0,
        "bwd_iat_mean": 0.0,
    }

    result = loaded_classifier.classify(features)
    assert "is_attack" in result
    assert "attack_probability" in result
    assert "inference_time_ms" in result
