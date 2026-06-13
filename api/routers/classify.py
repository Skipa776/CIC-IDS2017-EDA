"""Classification endpoints."""

import time
import uuid
from typing import List

from fastapi import APIRouter, HTTPException

from api.models.classifier import classifier
from api.models.schemas import (
    FlowFeatures,
    ClassificationResponse,
    ClassificationResult,
    BatchClassificationRequest,
    BatchClassificationResponse,
)
from api.services.prediction import get_classification

router = APIRouter(tags=["Classification"])


@router.post("/classify", response_model=ClassificationResponse)
async def classify_flow(features: FlowFeatures):
    """
    Classify a single network flow.

    This endpoint runs the layered classification:
    1. Layer 1: Binary classification (benign vs attack)
    2. Layer 2: Multi-class classification (attack type identification)
    3. MITRE ATT&CK mapping with mitigations

    Args:
        features: Network flow features for classification

    Returns:
        Classification result with attack type and MITRE mapping
    """
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Service is starting up."
        )

    try:
        result = get_classification(features)

        return ClassificationResponse(
            success=True,
            result=result,
            request_id=str(uuid.uuid4())
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )


@router.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_batch(request: BatchClassificationRequest):
    """
    Classify multiple network flows in a single request.

    Args:
        request: Batch of flow features to classify

    Returns:
        Classification results for all flows
    """
    if not classifier.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Service is starting up."
        )

    start_time = time.perf_counter()
    results: List[ClassificationResult] = []

    try:
        for flow in request.flows:
            result = get_classification(flow)
            results.append(result)

        total_time = (time.perf_counter() - start_time) * 1000

        return BatchClassificationResponse(
            success=True,
            results=results,
            request_id=str(uuid.uuid4()),
            total_inference_time_ms=total_time
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch classification failed: {str(e)}"
        )
