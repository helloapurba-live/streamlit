from fastapi.testclient import TestClient
from api import app
import pytest

client = TestClient(app)

def test_read_main():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Iris API. Go to /docs for Swagger UI."}

def test_predict_endpoint():
    """Test the prediction endpoint with valid data."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
        
    assert response.status_code == 200
    data = response.json()
    assert "species" in data
    assert "probabilities" in data
    assert data["species"] == "setosa" # Deterministic with fixed seed

def test_predict_invalid_data():
    """Test validation error."""
    payload = {
        "sepal_length": "invalid", # String instead of float
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Unprocessable Entity
