"""Tests for inference service."""

import pytest

from ml_service.services.inference import InferenceService


class TestInferenceService:
    """Tests for InferenceService."""

    @pytest.fixture
    def service(self) -> InferenceService:
        """Create a fresh inference service instance."""
        return InferenceService()

    def test_initial_state(self, service: InferenceService) -> None:
        """Test service initial state."""
        assert service.is_ready is False
        assert service.model_version == "not-loaded"

    def test_load_model(self, service: InferenceService) -> None:
        """Test loading the model."""
        service.load_model()
        assert service.is_ready is True
        assert service.model_version != "not-loaded"

    def test_unload_model(self, service: InferenceService) -> None:
        """Test unloading the model."""
        service.load_model()
        assert service.is_ready is True

        service.unload_model()
        assert service.is_ready is False

    def test_classify_without_model_raises(self, service: InferenceService) -> None:
        """Test that classify raises when model not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            service.classify("test")

    def test_classify_batch_without_model_raises(self, service: InferenceService) -> None:
        """Test that classify_batch raises when model not loaded."""
        with pytest.raises(RuntimeError, match="Model not loaded"):
            service.classify_batch(["test"])

    def test_classify_with_model(self, service: InferenceService) -> None:
        """Test classification with loaded model."""
        service.load_model()
        result = service.classify("Hello world")

        assert result.label in ["toxic", "non-toxic"]
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_batch_with_model(self, service: InferenceService) -> None:
        """Test batch classification with loaded model."""
        service.load_model()
        results = service.classify_batch(["Hello", "World"])

        assert len(results) == 2
        for result in results:
            assert result.label in ["toxic", "non-toxic"]
