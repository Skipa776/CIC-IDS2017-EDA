"""Tests for MITRE ATT&CK mapping service."""

import json
from pathlib import Path

import pytest

from api.services.mitre_mapping import get_mitre_mapping, MITRE_MAPPINGS

EXPORT_PATH = Path(__file__).parent.parent / "models" / "mitre_mapping.json"


# All attack types from CICIDS2017 (excluding BENIGN)
ATTACK_TYPES = [
    "DoS Hulk",
    "DoS GoldenEye",
    "DoS Slowhttptest",
    "DoS slowloris",
    "DDoS",
    "FTP-Patator",
    "SSH-Patator",
    "PortScan",
    "Web Attack - Brute Force",
    "Web Attack - XSS",
    "Web Attack - Sql Injection",
    "Bot",
    "Infiltration",
    "Heartbleed",
]


@pytest.mark.parametrize("attack_type", ATTACK_TYPES)
def test_all_attack_types_have_mitre_entry(attack_type):
    """Every attack type must have a MITRE mapping."""
    result = get_mitre_mapping(attack_type)
    assert result is not None, f"No MITRE mapping for: {attack_type}"


@pytest.mark.parametrize("attack_type", ATTACK_TYPES)
def test_mapping_has_required_fields(attack_type):
    """Each MITRE mapping must contain all required fields."""
    mapping = get_mitre_mapping(attack_type)
    assert "technique_id" in mapping
    assert "tactic" in mapping
    assert "url" in mapping
    assert "mitigations" in mapping
    assert mapping["technique_id"].startswith("T")
    assert len(mapping["mitigations"]) > 0


def test_unknown_attack_returns_none():
    """Unknown attack type should return None."""
    assert get_mitre_mapping("FakeAttack-9999") is None


def test_exported_json_matches_mappings():
    """models/mitre_mapping.json must exist and mirror MITRE_MAPPINGS exactly."""
    assert EXPORT_PATH.exists(), (
        f"Missing {EXPORT_PATH} - regenerate with: python -m api.services.mitre_mapping"
    )
    with open(EXPORT_PATH) as f:
        exported = json.load(f)
    assert exported == MITRE_MAPPINGS


def test_web_attack_with_unicode_resolves_after_normalization():
    """Web Attack labels with replacement char resolve after normalization."""
    # Simulate labels as stored in label_mapping.json (with \ufffd)
    unicode_labels = [
        "Web Attack \ufffd Brute Force",
        "Web Attack \ufffd XSS",
        "Web Attack \ufffd Sql Injection",
    ]
    expected_keys = [
        "Web Attack - Brute Force",
        "Web Attack - XSS",
        "Web Attack - Sql Injection",
    ]

    for unicode_label, expected_key in zip(unicode_labels, expected_keys):
        # Direct lookup fails
        assert get_mitre_mapping(unicode_label) is None
        # After normalization it succeeds
        normalized = unicode_label.replace('\ufffd', '-')
        result = get_mitre_mapping(normalized)
        assert result is not None, f"Normalization failed for: {unicode_label}"
        assert result == MITRE_MAPPINGS[expected_key]
