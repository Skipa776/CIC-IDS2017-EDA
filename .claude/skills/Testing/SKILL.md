# Skill: Testing

## Description
This skill covers Phase 5: Implementing tests for models, API, and mappings, ensuring performance targets.

## Instructions for Claude
You are a code assistant implementing Phase 5 of the CICIDS2017 IDS plan. Use pytest for tests.

Create tests/__init__.py (empty).

Create tests/test_models.py: Test training, inference speed (<5ms), metrics (Layer 1 PR-AUC >0.95, Recall >0.90; Layer 2 Macro F1 >0.80).

Create tests/test_api.py: Use httpx to test endpoints (classify, health); validate schemas.

Create tests/test_mitre_mapping.py: Check all attacks mapped, valid ATT&CK IDs.

Run tests via bash: 'pytest tests/'. Fix failures.

Output test results and coverage.

## Dependencies
- pytest, httpx, lightgbm, fastapi (for API tests)
- All prior phases completed