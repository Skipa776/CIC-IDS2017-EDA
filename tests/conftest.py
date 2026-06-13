"""Shared test fixtures."""

import pytest
import httpx

from api.models.classifier import classifier
from api.main import app


@pytest.fixture(scope="session", autouse=True)
def load_models():
    """Load classifier models once for the entire test session."""
    classifier.load_models()


@pytest.fixture
def sample_benign_flow():
    """A typical benign HTTPS flow."""
    return {
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


@pytest.fixture
def sample_dos_flow():
    """A DoS-like flow with high packet rate and short duration."""
    return {
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


@pytest.fixture
def sample_portscan_flow():
    """A port scan flow with many SYN flags and no data."""
    return {
        "destination_port": 22,
        "flow_duration": 0,
        "flow_bytes_per_sec": 0.0,
        "flow_packets_per_sec": 0.0,
        "total_fwd_packets": 1,
        "total_backward_packets": 1,
        "fwd_packet_length_mean": 0.0,
        "bwd_packet_length_mean": 0.0,
        "packet_length_mean": 0.0,
        "packet_length_std": 0.0,
        "syn_flag_count": 1,
        "fin_flag_count": 0,
        "rst_flag_count": 1,
        "ack_flag_count": 0,
        "init_win_bytes_forward": 1024,
        "init_win_bytes_backward": 0,
        "flow_iat_mean": 0.0,
        "flow_iat_std": 0.0,
        "fwd_iat_mean": 0.0,
        "bwd_iat_mean": 0.0,
    }


@pytest.fixture
async def async_client():
    """Async HTTP client for testing the FastAPI app."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
