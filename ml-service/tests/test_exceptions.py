"""Tests for custom exception hierarchy."""

import pytest

from ml_service.core.exceptions import (
    ClassificationError,
    ConfigurationError,
    FeatureStoreConnectionError,
    FeatureStoreError,
    MLServiceException,
    ModelNotLoadedError,
)


class TestMLServiceException:
    """Tests for base MLServiceException."""

    def test_default_message(self) -> None:
        """Test exception with default empty message."""
        exc = MLServiceException()
        assert exc.message == ""
        assert str(exc) == ""

    def test_custom_message(self) -> None:
        """Test exception with custom message."""
        exc = MLServiceException("Custom error message")
        assert exc.message == "Custom error message"
        assert str(exc) == "Custom error message"

    def test_is_exception(self) -> None:
        """Test that MLServiceException inherits from Exception."""
        exc = MLServiceException("test")
        assert isinstance(exc, Exception)


class TestModelNotLoadedError:
    """Tests for ModelNotLoadedError."""

    def test_default_message(self) -> None:
        """Test exception with default message."""
        exc = ModelNotLoadedError()
        assert exc.message == "Model not loaded"
        assert str(exc) == "Model not loaded"

    def test_custom_message(self) -> None:
        """Test exception with custom message."""
        exc = ModelNotLoadedError("Model 'foo' not loaded")
        assert exc.message == "Model 'foo' not loaded"

    def test_inheritance(self) -> None:
        """Test that ModelNotLoadedError inherits from MLServiceException."""
        exc = ModelNotLoadedError()
        assert isinstance(exc, MLServiceException)
        assert isinstance(exc, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        """Test that the exception can be raised and caught."""
        with pytest.raises(ModelNotLoadedError, match="Model not loaded"):
            raise ModelNotLoadedError()


class TestFeatureStoreError:
    """Tests for FeatureStoreError."""

    def test_default_message(self) -> None:
        """Test exception with default message."""
        exc = FeatureStoreError()
        assert exc.message == "Feature store operation failed"

    def test_custom_message(self) -> None:
        """Test exception with custom message."""
        exc = FeatureStoreError("Redis connection timeout")
        assert exc.message == "Redis connection timeout"

    def test_inheritance(self) -> None:
        """Test that FeatureStoreError inherits from MLServiceException."""
        exc = FeatureStoreError()
        assert isinstance(exc, MLServiceException)


class TestFeatureStoreConnectionError:
    """Tests for FeatureStoreConnectionError."""

    def test_default_message(self) -> None:
        """Test exception with default message."""
        exc = FeatureStoreConnectionError()
        assert exc.message == "Not connected to feature store"

    def test_inheritance(self) -> None:
        """Test that FeatureStoreConnectionError inherits from FeatureStoreError."""
        exc = FeatureStoreConnectionError()
        assert isinstance(exc, FeatureStoreError)
        assert isinstance(exc, MLServiceException)

    def test_can_catch_as_feature_store_error(self) -> None:
        """Test that FeatureStoreConnectionError can be caught as FeatureStoreError."""
        with pytest.raises(FeatureStoreError):
            raise FeatureStoreConnectionError()


class TestConfigurationError:
    """Tests for ConfigurationError."""

    def test_default_message(self) -> None:
        """Test exception with default message."""
        exc = ConfigurationError()
        assert exc.message == "Invalid configuration"

    def test_custom_message(self) -> None:
        """Test exception with custom message."""
        exc = ConfigurationError("Missing REDIS_URL environment variable")
        assert exc.message == "Missing REDIS_URL environment variable"

    def test_inheritance(self) -> None:
        """Test that ConfigurationError inherits from MLServiceException."""
        exc = ConfigurationError()
        assert isinstance(exc, MLServiceException)


class TestClassificationError:
    """Tests for ClassificationError."""

    def test_default_message(self) -> None:
        """Test exception with default message."""
        exc = ClassificationError()
        assert exc.message == "Classification failed"

    def test_custom_message(self) -> None:
        """Test exception with custom message."""
        exc = ClassificationError("Input text too long")
        assert exc.message == "Input text too long"

    def test_inheritance(self) -> None:
        """Test that ClassificationError inherits from MLServiceException."""
        exc = ClassificationError()
        assert isinstance(exc, MLServiceException)


class TestExceptionHierarchy:
    """Tests for exception hierarchy behavior."""

    def test_catch_all_with_base_exception(self) -> None:
        """Test that all custom exceptions can be caught with MLServiceException."""
        exceptions = [
            ModelNotLoadedError(),
            FeatureStoreError(),
            FeatureStoreConnectionError(),
            ConfigurationError(),
            ClassificationError(),
        ]
        for exc in exceptions:
            with pytest.raises(MLServiceException):
                raise exc

    def test_different_exception_types(self) -> None:
        """Test that different exceptions have different types."""
        exc1 = ModelNotLoadedError()
        exc2 = FeatureStoreError()

        assert type(exc1) is not type(exc2)
        assert isinstance(exc1, ModelNotLoadedError)
        assert not isinstance(exc1, FeatureStoreError)
        assert isinstance(exc2, FeatureStoreError)
        assert not isinstance(exc2, ModelNotLoadedError)
