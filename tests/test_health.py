"""Tests for health and model-info endpoints."""

import pytest


@pytest.mark.asyncio
async def test_health_returns_200(async_client):
    resp = await async_client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert "models_loaded" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_models_loaded(async_client):
    resp = await async_client.get("/api/v1/health")
    data = resp.json()
    assert data["models_loaded"] is True
    assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_model_info_returns_metadata(async_client):
    resp = await async_client.get("/api/v1/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert data["layer1_type"] == "LGBMClassifier"
    assert data["layer2_type"] == "LGBMClassifier"
    assert data["feature_count"] == 20
    assert data["num_classes"] == 15
    assert "metrics" in data
    assert "trained_at" in data


@pytest.mark.asyncio
async def test_root_endpoint(async_client):
    resp = await async_client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "CICIDS2017 IDS API"
    assert "version" in data
    assert "docs" in data
    assert "health" in data
    assert "classify" in data
