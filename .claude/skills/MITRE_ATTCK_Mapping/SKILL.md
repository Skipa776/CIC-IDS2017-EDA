# Skill: MITRE ATT&CK Mapping

## Description
This skill implements Phase 4: Creating a static mapping of CICIDS attacks to MITRE ATT&CK techniques, tactics, and mitigations.

## Instructions for Claude
You are a code assistant implementing Phase 4 of the CICIDS2017 IDS plan.

In api/services/mitre_mapping.py (or standalone if Phase 3 not done):
- Define a dict mapping CICIDS attacks to ATT&CK: Use the table from PLAN.md (e.g., 'DoS Hulk': {'technique_id': 'T1498.001', 'technique_name': 'Network DoS: Direct Network Flood', 'tactic': 'Impact', 'mitigations': ['M1037: Rate limiting', ...]}).
- Expand each with 5+ specific mitigations (research via knowledge or suggest using Browser MCP if enabled).
- Ensure all 14-15 attack types are covered.
- Save as Python dict; export to models/mitre_mapping.json if needed.

Validate: Write a test script to check all keys and valid IDs.

Output the mapping dict and any additions.

## Dependencies
- None (static), but integrate with api/services/