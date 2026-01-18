"""Pytest fixtures for ml-service tests."""

import pytest
from fastapi.testclient import TestClient

from ml_service.main import app
from ml_service.services.inference import get_inference_service


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    # Ensure model is loaded before tests
    service = get_inference_service()
    service.load_model()
    return TestClient(app)


@pytest.fixture
def inference_service():
    """Get the inference service instance."""
    service = get_inference_service()
    if not service.is_ready:
        service.load_model()
    return service
