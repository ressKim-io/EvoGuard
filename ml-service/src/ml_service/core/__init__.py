"""Core module for ml-service.

This module provides core functionality including:
- Configuration management (Settings, get_settings)
- Custom exceptions (MLServiceException and subclasses)
- Protocol definitions for dependency injection
- Logging utilities
"""

from ml_service.core.config import Settings, get_settings
from ml_service.core.exceptions import (
    ClassificationError,
    ConfigurationError,
    FeatureStoreConnectionError,
    FeatureStoreError,
    MLServiceException,
    ModelNotLoadedError,
)
from ml_service.core.logging import get_logger
from ml_service.core.protocols import (
    FeatureStoreProtocol,
    InferenceServiceProtocol,
    MetricsCollectorProtocol,
    RepositoryProtocol,
)

__all__ = [
    "ClassificationError",
    "ConfigurationError",
    "FeatureStoreConnectionError",
    "FeatureStoreError",
    "FeatureStoreProtocol",
    "InferenceServiceProtocol",
    "MLServiceException",
    "MetricsCollectorProtocol",
    "ModelNotLoadedError",
    "RepositoryProtocol",
    "Settings",
    "get_logger",
    "get_settings",
]
