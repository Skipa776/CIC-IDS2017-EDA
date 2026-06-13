# Skill: REST API Implementation

## Description
This skill builds Phase 3 of the PLAN.md: Setting up the FastAPI project structure, endpoints, schemas, and services for the IDS classification API.

## Instructions for Claude
You are a code assistant implementing Phase 3 of the CICIDS2017 IDS plan. Use file tools to create the api/ directory structure.

Create api/__init__.py (empty).

Create api/main.py: FastAPI app, load models on startup using joblib from models/.

Create api/config.py: Define settings (e.g., model paths).

Create api/models/schemas.py: Pydantic models for FlowFeatures and ClassificationResult (match JSON schema in plan).

Create api/models/classifier.py: Wrapper class for layered inference (load scaler, models; predict binary then multi-class).

Create api/routers/classify.py: POST /api/v1/classify (single flow), POST /api/v1/classify/batch (multiple).

Create api/routers/health.py: GET /api/v1/health, GET /api/v1/model-info.

Create api/services/prediction.py: Inference logic (scale features, predict, calculate probs/confidence, inference time).

Create api/services/mitre_mapping.py: Static dict from Phase 4 table.

Integrate MITRE mapping. Use bash to set up virtual env if needed, but assume requirements.txt is installed.

Test locally: Suggest running 'uvicorn api.main:app --reload'. Output created file paths.

## Dependencies
- fastapi, uvicorn, pydantic, joblib, lightgbm
- Model artifacts from Phase 2