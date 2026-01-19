"""Custom exception hierarchy for ml-service.

This module defines domain-specific exceptions that replace generic
RuntimeError exceptions throughout the codebase for better error handling.
"""


class MLServiceException(Exception):  # noqa: N818
    """Base exception for ml-service.

    All custom exceptions in ml-service should inherit from this class.
    This allows callers to catch all ml-service specific exceptions with
    a single except clause if needed.
    """

    def __init__(self, message: str = "") -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
        """
        super().__init__(message)
        self.message = message


class ModelNotLoadedError(MLServiceException):
    """Model is not loaded in inference service.

    Raised when attempting to perform inference operations
    before the model has been loaded.
    """

    def __init__(self, message: str = "Model not loaded") -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
        """
        super().__init__(message)


class FeatureStoreError(MLServiceException):
    """Feature store operation failed.

    Raised when a feature store operation (read, write, delete)
    fails due to connection issues or other problems.
    """

    def __init__(self, message: str = "Feature store operation failed") -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
        """
        super().__init__(message)


class FeatureStoreConnectionError(FeatureStoreError):
    """Feature store connection failed.

    Raised when the feature store cannot establish a connection
    to the underlying storage (Redis, PostgreSQL, etc.).
    """

    def __init__(self, message: str = "Not connected to feature store") -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
        """
        super().__init__(message)


class ConfigurationError(MLServiceException):
    """Configuration is invalid.

    Raised when the application configuration is invalid or missing
    required values.
    """

    def __init__(self, message: str = "Invalid configuration") -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
        """
        super().__init__(message)


class ClassificationError(MLServiceException):
    """Classification operation failed.

    Raised when the model fails to classify input data due to
    unexpected errors during inference.
    """

    def __init__(self, message: str = "Classification failed") -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
        """
        super().__init__(message)


class DatabaseNotInitializedError(MLServiceException):
    """Database is not initialized.

    Raised when attempting to perform database operations
    before the database has been initialized.
    """

    def __init__(
        self, message: str = "Database not initialized. Call init_database() first."
    ) -> None:
        """Initialize the exception.

        Args:
            message: Human-readable error message.
        """
        super().__init__(message)
