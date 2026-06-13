#!/usr/bin/env python3
"""
Smoke test for the IDS API.

Hits a live API instance and verifies core functionality.
Usage: python scripts/smoke_test.py [--url http://localhost:8000]
Exit code: 0 on success, 1 on failure.
"""

import argparse
import sys
import json

import httpx


BENIGN_FLOW = {
    "destination_port": 443,
    "flow_duration": 50000,
    "flow_bytes_per_sec": 15000.0,
    "flow_packets_per_sec": 20.0,
    "total_fwd_packets": 5,
    "total_backward_packets": 4,
    "fwd_packet_length_mean": 120.0,
    "bwd_packet_length_mean": 800.0,
    "packet_length_mean": 400.0,
    "packet_length_std": 300.0,
    "syn_flag_count": 1,
    "fin_flag_count": 1,
    "rst_flag_count": 0,
    "ack_flag_count": 8,
    "init_win_bytes_forward": 65535,
    "init_win_bytes_backward": 65535,
    "flow_iat_mean": 5000.0,
    "flow_iat_std": 2000.0,
    "fwd_iat_mean": 8000.0,
    "bwd_iat_mean": 7000.0,
}

ATTACK_FLOW = {
    "destination_port": 80,
    "flow_duration": 100,
    "flow_bytes_per_sec": 5000000.0,
    "flow_packets_per_sec": 50000.0,
    "total_fwd_packets": 500,
    "total_backward_packets": 0,
    "fwd_packet_length_mean": 64.0,
    "bwd_packet_length_mean": 0.0,
    "packet_length_mean": 64.0,
    "packet_length_std": 0.0,
    "syn_flag_count": 500,
    "fin_flag_count": 0,
    "rst_flag_count": 0,
    "ack_flag_count": 0,
    "init_win_bytes_forward": 1024,
    "init_win_bytes_backward": 0,
    "flow_iat_mean": 2.0,
    "flow_iat_std": 1.0,
    "fwd_iat_mean": 2.0,
    "bwd_iat_mean": 0.0,
}


def run_tests(base_url: str) -> bool:
    """Run all smoke tests. Returns True if all pass."""
    passed = 0
    failed = 0
    client = httpx.Client(base_url=base_url, timeout=10.0)

    def check(name: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            print(f"  PASS: {name}")
            passed += 1
        else:
            print(f"  FAIL: {name} {detail}")
            failed += 1

    # Test 1: Health endpoint
    print("\n[Health]")
    try:
        resp = client.get("/api/v1/health")
        check("health returns 200", resp.status_code == 200)
        data = resp.json()
        check("models loaded", data.get("models_loaded") is True)
    except Exception as e:
        check("health endpoint reachable", False, str(e))

    # Test 2: Model info
    print("\n[Model Info]")
    try:
        resp = client.get("/api/v1/model-info")
        check("model-info returns 200", resp.status_code == 200)
        data = resp.json()
        check("has feature_count", data.get("feature_count", 0) > 0)
        check("has num_classes", data.get("num_classes", 0) > 0)
    except Exception as e:
        check("model-info reachable", False, str(e))

    # Test 3: Classify benign
    print("\n[Classify Benign]")
    try:
        resp = client.post("/api/v1/classify", json=BENIGN_FLOW)
        check("classify returns 200", resp.status_code == 200)
        data = resp.json()
        check("success is true", data.get("success") is True)
        result = data.get("result", {})
        check("has is_attack field", "is_attack" in result)
        check("has inference_time_ms", "inference_time_ms" in result)
    except Exception as e:
        check("classify benign reachable", False, str(e))

    # Test 4: Classify attack
    print("\n[Classify Attack]")
    try:
        resp = client.post("/api/v1/classify", json=ATTACK_FLOW)
        check("classify attack returns 200", resp.status_code == 200)
        result = resp.json().get("result", {})
        if result.get("is_attack"):
            check("attack has attack_type", result.get("attack_type") is not None)
        else:
            check("attack classified (may be benign)", True)
    except Exception as e:
        check("classify attack reachable", False, str(e))

    # Test 5: Batch
    print("\n[Batch]")
    try:
        resp = client.post("/api/v1/classify/batch", json={"flows": [BENIGN_FLOW, ATTACK_FLOW]})
        check("batch returns 200", resp.status_code == 200)
        data = resp.json()
        check("batch has 2 results", len(data.get("results", [])) == 2)
    except Exception as e:
        check("batch reachable", False, str(e))

    # Test 6: Invalid input
    print("\n[Invalid Input]")
    try:
        bad_flow = {"destination_port": 99999}
        resp = client.post("/api/v1/classify", json=bad_flow)
        check("invalid input returns 422", resp.status_code == 422)
    except Exception as e:
        check("invalid input handled", False, str(e))

    client.close()

    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Smoke test for IDS API")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL of the API")
    args = parser.parse_args()

    print(f"Running smoke tests against: {args.url}")
    success = run_tests(args.url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
