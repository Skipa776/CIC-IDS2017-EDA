"""Health check endpoints."""

from fastapi import APIRouter, HTTPException

from api.config import settings
from api.models.classifier import classifier
from api.models.schemas import HealthResponse, ModelInfoResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check API health status.

    Returns:
        Health status including model loading state
    """
    return HealthResponse(
        status="healthy" if classifier.is_loaded else "degraded",
        models_loaded=classifier.is_loaded,
        version=settings.app_version
    )


@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """
    Get information about the loaded models.

    Returns:
        Model metadata and performance metrics
    """
    if not classifier.is_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded")

    metadata = classifier.metadata

    return ModelInfoResponse(
        version=metadata.get('version', '1.0.0'),
        layer1_type=metadata.get('layer1_type', 'Unknown'),
        layer2_type=metadata.get('layer2_type', 'Unknown'),
        feature_count=metadata.get('feature_count', 0),
        num_classes=metadata.get('num_classes', 0),
        metrics=metadata.get('metrics', {}),
        trained_at=metadata.get('trained_at', 'Unknown')
    )
