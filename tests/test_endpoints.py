"""
FastAPI Endpoint Tests
"""

import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.backend.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns service info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["service"] == "Bank Transaction Anomaly Detection API"


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_single_valid():
    """Test single prediction with valid input."""
    payload = {
        "customer_id": "CUST_000123",
        "amount": 100.0,
        "merchant_category": "grocery",
        "merchant_id": "MERCH_01234",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "is_online": 0,
        "hour": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_night": 0
    }

    response = client.post("/predict_single", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "fraud_probability" in data
    assert "is_fraud_predicted" in data
    assert "explanation" in data
    assert 0 <= data["fraud_probability"] <= 1


def test_predict_single_invalid_category():
    """Test single prediction with invalid merchant category."""
    payload = {
        "customer_id": "CUST_000123",
        "amount": 100.0,
        "merchant_category": "invalid_category",  # Invalid
        "merchant_id": "MERCH_01234",
        "latitude": 40.7128,
        "longitude": -74.0060,
        "is_online": 0,
        "hour": 14,
        "day_of_week": 2,
        "is_weekend": 0,
        "is_night": 0
    }

    response = client.post("/predict_single", json=payload)
    assert response.status_code == 422  # Validation error


def test_investigate_endpoint():
    """Test investigation endpoint."""
    payload = {
        "customer_id": "CUST_000123",
        "query": "Why was this customer flagged?"
    }

    response = client.post("/agent_investigate", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "customer_id" in data
    assert "risk_score" in data
    assert "key_findings" in data
    assert isinstance(data["key_findings"], list)


def test_metrics_endpoint():
    """Test Prometheus metrics endpoint."""
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
