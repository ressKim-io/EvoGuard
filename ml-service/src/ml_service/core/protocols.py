"""Protocol definitions for ml-service.

This module defines abstract interfaces (protocols) that establish
contracts for repositories and services, enabling dependency injection
and loose coupling between components.
"""

from typing import Any, Protocol, TypeVar
from uuid import UUID

T = TypeVar("T")


class RepositoryProtocol(Protocol[T]):
    """Generic repository interface for CRUD operations.

    This protocol defines the standard interface that all repositories
    should implement. It provides a consistent API for data access
    across different storage backends.

    Type Parameters:
        T: The entity type managed by the repository.
    """

    async def create(self, data: Any) -> T:
        """Create a new entity.

        Args:
            data: Data to create the entity from.

        Returns:
            The created entity.
        """
        ...

    async def get_by_id(self, entity_id: UUID) -> T | None:
        """Get an entity by its ID.

        Args:
            entity_id: The unique identifier of the entity.

        Returns:
            The entity if found, None otherwise.
        """
        ...

    async def list_all(self, **filters: Any) -> list[T]:
        """List all entities matching the given filters.

        Args:
            **filters: Optional filter criteria.

        Returns:
            List of matching entities.
        """
        ...

    async def update(self, entity_id: UUID, data: Any) -> T | None:
        """Update an existing entity.

        Args:
            entity_id: The unique identifier of the entity.
            data: Data to update the entity with.

        Returns:
            The updated entity if found, None otherwise.
        """
        ...

    async def delete(self, entity_id: UUID) -> bool:
        """Delete an entity by its ID.

        Args:
            entity_id: The unique identifier of the entity.

        Returns:
            True if deleted, False if not found.
        """
        ...


class FeatureStoreProtocol(Protocol):
    """Protocol for feature store implementations.

    Defines the interface for online and offline feature stores,
    allowing interchangeable storage backends.
    """

    async def connect(self) -> None:
        """Establish connection to the feature store."""
        ...

    async def close(self) -> None:
        """Close the connection to the feature store."""
        ...

    async def set_features(
        self,
        entity_type: str,
        entity_id: str,
        feature_group: str,
        features: dict[str, Any],
        version: int = 1,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store features for an entity.

        Args:
            entity_type: Type of entity (e.g., "text", "battle").
            entity_id: Unique entity identifier.
            feature_group: Name of the feature group.
            features: Dictionary of feature name -> value.
            version: Feature group version.
            ttl_seconds: Optional TTL for the features.
        """
        ...

    async def get_features(
        self,
        entity_type: str,
        entity_id: str,
        feature_group: str,
        version: int = 1,
    ) -> dict[str, Any] | None:
        """Retrieve features for an entity.

        Args:
            entity_type: Type of entity.
            entity_id: Unique entity identifier.
            feature_group: Name of the feature group.
            version: Feature group version.

        Returns:
            Dictionary of features or None if not found.
        """
        ...

    async def delete_features(
        self,
        entity_type: str,
        entity_id: str,
        feature_group: str,
        version: int = 1,
    ) -> bool:
        """Delete features for an entity.

        Args:
            entity_type: Type of entity.
            entity_id: Unique entity identifier.
            feature_group: Name of the feature group.
            version: Feature group version.

        Returns:
            True if deleted, False if not found.
        """
        ...


class InferenceServiceProtocol(Protocol):
    """Protocol for inference service implementations.

    Defines the interface for model inference services,
    enabling different model backends.
    """

    @property
    def is_ready(self) -> bool:
        """Check if the model is loaded and ready for inference."""
        ...

    @property
    def model_version(self) -> str:
        """Get the current model version."""
        ...

    def load_model(self) -> None:
        """Load the classification model."""
        ...

    def unload_model(self) -> None:
        """Unload the current model."""
        ...


class MetricsCollectorProtocol(Protocol):
    """Protocol for metrics collection.

    Defines the interface for collecting and recording metrics
    from various service components.
    """

    def record_prediction(
        self,
        prediction: int,
        confidence: float,
        low_confidence_threshold: float,
    ) -> None:
        """Record a prediction metric.

        Args:
            prediction: The prediction class (0 or 1).
            confidence: The prediction confidence score.
            low_confidence_threshold: Threshold for low confidence.
        """
        ...

    def record_latency(self, latency_seconds: float) -> None:
        """Record a latency metric.

        Args:
            latency_seconds: The latency in seconds.
        """
        ...
