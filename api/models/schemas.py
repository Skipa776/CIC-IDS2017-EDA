"""Pydantic schemas for API request/response models."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class FlowFeatures(BaseModel):
    """Network flow features for classification."""

    destination_port: int = Field(..., ge=0, le=65535, description="Destination port number")
    flow_duration: float = Field(..., ge=0, description="Flow duration in microseconds")
    flow_bytes_per_sec: float = Field(..., ge=0, description="Flow bytes per second")
    flow_packets_per_sec: float = Field(..., ge=0, description="Flow packets per second")
    total_fwd_packets: int = Field(..., ge=0, description="Total forward packets")
    total_backward_packets: int = Field(..., ge=0, description="Total backward packets")
    fwd_packet_length_mean: float = Field(..., ge=0, description="Mean forward packet length")
    bwd_packet_length_mean: float = Field(..., ge=0, description="Mean backward packet length")
    packet_length_mean: float = Field(..., ge=0, description="Mean packet length")
    packet_length_std: float = Field(..., ge=0, description="Packet length standard deviation")
    syn_flag_count: int = Field(..., ge=0, description="SYN flag count")
    fin_flag_count: int = Field(..., ge=0, description="FIN flag count")
    rst_flag_count: int = Field(..., ge=0, description="RST flag count")
    ack_flag_count: int = Field(..., ge=0, description="ACK flag count")
    init_win_bytes_forward: int = Field(..., ge=0, description="Initial window bytes forward")
    init_win_bytes_backward: int = Field(..., ge=0, description="Initial window bytes backward")
    flow_iat_mean: float = Field(..., description="Flow inter-arrival time mean (can be negative)")
    flow_iat_std: float = Field(..., ge=0, description="Flow inter-arrival time std")
    fwd_iat_mean: float = Field(..., description="Forward inter-arrival time mean")
    bwd_iat_mean: float = Field(..., description="Backward inter-arrival time mean")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class MitreMapping(BaseModel):
    """MITRE ATT&CK mapping for an attack type."""

    technique_id: str = Field(..., description="ATT&CK technique ID (e.g., T1498.001)")
    technique_name: str = Field(..., description="ATT&CK technique name")
    tactic: str = Field(..., description="ATT&CK tactic (e.g., Impact, Credential Access)")
    url: str = Field(..., description="Link to ATT&CK page")
    mitigations: List[str] = Field(..., description="Recommended mitigations")


class ClassificationResult(BaseModel):
    """Classification result for a single flow."""

    is_attack: bool = Field(..., description="Whether the flow is classified as an attack")
    attack_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of being an attack")
    attack_type: Optional[str] = Field(None, description="Specific attack type if classified as attack")
    attack_type_probability: Optional[float] = Field(None, ge=0.0, le=1.0, description="Probability of the attack type")
    confidence_level: str = Field(..., description="Confidence level: high, medium, or low")
    mitre_mapping: Optional[MitreMapping] = Field(None, description="MITRE ATT&CK mapping if attack")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    model_version: str = Field(..., description="Model version used for classification")


class ClassificationResponse(BaseModel):
    """API response for classification endpoint."""

    success: bool = Field(..., description="Whether the request was successful")
    result: ClassificationResult = Field(..., description="Classification result")
    request_id: str = Field(..., description="Unique request identifier")


class BatchClassificationRequest(BaseModel):
    """Request for batch classification."""

    flows: List[FlowFeatures] = Field(..., min_length=1, max_length=1000, description="List of flows to classify")


class BatchClassificationResponse(BaseModel):
    """Response for batch classification."""

    success: bool = Field(..., description="Whether the request was successful")
    results: List[ClassificationResult] = Field(..., description="Classification results for each flow")
    request_id: str = Field(..., description="Unique request identifier")
    total_inference_time_ms: float = Field(..., description="Total inference time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    version: str = Field(..., description="Model version")
    layer1_type: str = Field(..., description="Layer 1 model type")
    layer2_type: str = Field(..., description="Layer 2 model type")
    feature_count: int = Field(..., description="Number of features used")
    num_classes: int = Field(..., description="Number of attack classes")
    metrics: Dict[str, Any] = Field(..., description="Model performance metrics")
    trained_at: str = Field(..., description="Model training timestamp")
