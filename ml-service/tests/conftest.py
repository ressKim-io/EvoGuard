"""Pytest fixtures for ml-service tests."""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ml_service.core.config import Settings, get_settings
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


@pytest.fixture
def settings_override() -> dict[str, Any]:
    """Override settings for tests.

    Returns a dictionary of settings to override.
    Tests can modify this dictionary to customize settings.
    """
    return {
        "low_confidence_threshold": 0.5,
        "prediction_buffer_size": 100,
        "prediction_sample_rate": 0.1,
        "text_features_ttl_seconds": 3600,
        "battle_features_ttl_seconds": 1800,
        "user_features_ttl_seconds": 7200,
        "default_features_ttl_seconds": 3600,
    }


@pytest.fixture
def test_settings(settings_override: dict[str, Any]) -> Settings:
    """Create a Settings instance with test-specific overrides.

    Args:
        settings_override: Dictionary of settings to override.

    Returns:
        Settings instance with overrides applied.
    """
    return Settings(**settings_override)


@pytest.fixture
def mock_settings(test_settings: Settings) -> Generator[Settings, None, None]:
    """Mock get_settings() to return test settings.

    This fixture patches get_settings() to return the test_settings
    fixture, allowing tests to use custom configuration.

    Args:
        test_settings: The test settings to use.

    Yields:
        The test settings instance.
    """
    with patch("ml_service.core.config.get_settings", return_value=test_settings):
        # Clear the LRU cache to ensure our mock is used
        get_settings.cache_clear()
        yield test_settings
    # Restore the cache after test
    get_settings.cache_clear()


@pytest.fixture
def mock_redis() -> Generator[MagicMock, None, None]:
    """Mock Redis client for testing.

    Yields a MagicMock that can be configured to simulate Redis operations.
    """
    with patch("redis.asyncio.Redis") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        # Configure common async methods
        mock_instance.ping = MagicMock(return_value=True)
        mock_instance.close = MagicMock()
        yield mock_instance


@pytest.fixture
def mock_metrics_collector() -> Generator[MagicMock, None, None]:
    """Mock MetricsCollector for testing.

    Yields a MagicMock that simulates the MetricsCollector interface.
    """
    mock = MagicMock()
    mock.record_prediction = MagicMock()
    mock.record_latency = MagicMock()
    with patch(
        "ml_service.monitoring.metrics.collector.MetricsCollector.get_instance",
        return_value=mock,
    ):
        yield mock
