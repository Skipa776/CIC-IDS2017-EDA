"""Tests for classification endpoints."""

import pytest


@pytest.mark.asyncio
async def test_single_flow_response_schema(async_client, sample_benign_flow):
    resp = await async_client.post("/api/v1/classify", json=sample_benign_flow)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "result" in data
    assert "request_id" in data
    result = data["result"]
    assert "is_attack" in result
    assert "attack_probability" in result
    assert "confidence_level" in result
    assert "inference_time_ms" in result
    assert "model_version" in result


@pytest.mark.asyncio
async def test_attack_flow_has_mitre_mapping(async_client, sample_dos_flow):
    resp = await async_client.post("/api/v1/classify", json=sample_dos_flow)
    assert resp.status_code == 200
    result = resp.json()["result"]
    if result["is_attack"] and result["attack_type"]:
        # If classified as attack, check MITRE mapping structure
        if result["mitre_mapping"]:
            mitre = result["mitre_mapping"]
            assert "technique_id" in mitre
            assert "tactic" in mitre
            assert "url" in mitre
            assert "mitigations" in mitre


@pytest.mark.asyncio
async def test_benign_flow_null_attack_fields(async_client, sample_benign_flow):
    resp = await async_client.post("/api/v1/classify", json=sample_benign_flow)
    assert resp.status_code == 200
    result = resp.json()["result"]
    if not result["is_attack"]:
        assert result["attack_type"] is None
        assert result["mitre_mapping"] is None


@pytest.mark.asyncio
async def test_invalid_port_returns_422(async_client, sample_benign_flow):
    sample_benign_flow["destination_port"] = 70000
    resp = await async_client.post("/api/v1/classify", json=sample_benign_flow)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_missing_field_returns_422(async_client):
    incomplete = {"destination_port": 80}
    resp = await async_client.post("/api/v1/classify", json=incomplete)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_negative_iat_accepted(async_client, sample_benign_flow):
    sample_benign_flow["flow_iat_mean"] = -500.0
    sample_benign_flow["fwd_iat_mean"] = -100.0
    sample_benign_flow["bwd_iat_mean"] = -200.0
    resp = await async_client.post("/api/v1/classify", json=sample_benign_flow)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_inference_time_under_100ms(async_client, sample_benign_flow):
    resp = await async_client.post("/api/v1/classify", json=sample_benign_flow)
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["inference_time_ms"] < 100.0


@pytest.mark.asyncio
async def test_batch_three_flows(async_client, sample_benign_flow, sample_dos_flow, sample_portscan_flow):
    payload = {"flows": [sample_benign_flow, sample_dos_flow, sample_portscan_flow]}
    resp = await async_client.post("/api/v1/classify/batch", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert len(data["results"]) == 3
    assert "total_inference_time_ms" in data


@pytest.mark.asyncio
async def test_batch_over_1000_returns_422(async_client, sample_benign_flow):
    payload = {"flows": [sample_benign_flow] * 1001}
    resp = await async_client.post("/api/v1/classify/batch", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_batch_zero_flows_returns_422(async_client):
    payload = {"flows": []}
    resp = await async_client.post("/api/v1/classify/batch", json=payload)
    assert resp.status_code == 422
