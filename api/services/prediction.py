"""Prediction service for flow classification."""

import uuid
from typing import Dict, Any

from api.config import settings
from api.models.classifier import classifier
from api.models.schemas import (
    FlowFeatures,
    ClassificationResult,
    MitreMapping,
)
from api.services.mitre_mapping import get_mitre_mapping


def get_confidence_level(probability: float) -> str:
    """
    Determine confidence level based on probability.

    Args:
        probability: Classification probability

    Returns:
        Confidence level string
    """
    if probability >= settings.confidence_high_threshold:
        return "high"
    elif probability >= settings.confidence_medium_threshold:
        return "medium"
    else:
        return "low"


def get_classification(features: FlowFeatures) -> ClassificationResult:
    """
    Classify a network flow.

    Args:
        features: FlowFeatures object with network flow data

    Returns:
        ClassificationResult with classification details
    """
    # Convert pydantic model to dict
    features_dict = features.model_dump()

    # Run classification
    result = classifier.classify(features_dict)

    # Get MITRE mapping if attack
    mitre = None
    if result['is_attack'] and result['attack_type']:
        # Normalize unicode replacement chars from label_mapping.json
        attack_type_normalized = result['attack_type'].replace('\ufffd', '-')
        mitre_data = get_mitre_mapping(attack_type_normalized)
        if mitre_data and mitre_data['technique_id'] != 'N/A':
            mitre = MitreMapping(
                technique_id=mitre_data['technique_id'],
                technique_name=mitre_data['technique_name'],
                tactic=mitre_data['tactic'],
                url=mitre_data['url'],
                mitigations=mitre_data['mitigations']
            )

    # Determine confidence level
    if result['is_attack']:
        confidence_prob = result['attack_type_probability'] or result['attack_probability']
    else:
        confidence_prob = 1 - result['attack_probability']

    confidence = get_confidence_level(confidence_prob)

    return ClassificationResult(
        is_attack=result['is_attack'],
        attack_probability=result['attack_probability'],
        attack_type=result['attack_type'],
        attack_type_probability=result['attack_type_probability'],
        confidence_level=confidence,
        mitre_mapping=mitre,
        inference_time_ms=result['inference_time_ms'],
        model_version=classifier.metadata.get('version', '1.0.0') if classifier.metadata else '1.0.0'
    )
