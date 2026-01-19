"""Base class for feature transformers."""

from abc import ABC, abstractmethod
from typing import Any


class FeatureTransformer(ABC):
    """Abstract base class for feature computation.

    All feature transformers must inherit from this class and implement
    the required abstract methods.

    Example:
        >>> class MyTransformer(FeatureTransformer):
        ...     @property
        ...     def feature_names(self) -> list[str]:
        ...         return ["feature_a", "feature_b"]
        ...
        ...     @property
        ...     def entity_type(self) -> str:
        ...         return "my_entity"
        ...
        ...     def transform(self, data: Any) -> dict[str, Any]:
        ...         return {"feature_a": 1, "feature_b": 2}
    """

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """List of feature names produced by this transformer.

        Returns:
            List of feature names that will be in the output dictionary.
        """
        ...

    @property
    @abstractmethod
    def entity_type(self) -> str:
        """Entity type this transformer operates on.

        Returns:
            Entity type string (e.g., 'text', 'battle', 'user').
        """
        ...

    @property
    def version(self) -> str:
        """Version of the transformer.

        Returns:
            Version string for tracking transformer changes.
        """
        return "1.0.0"

    @abstractmethod
    def transform(self, data: Any) -> dict[str, Any]:
        """Transform a single data point into features.

        Args:
            data: Input data to transform. Type depends on the specific
                transformer implementation.

        Returns:
            Dictionary mapping feature names to computed values.
        """
        ...

    def transform_batch(self, data_list: list[Any]) -> list[dict[str, Any]]:
        """Transform a batch of data points into features.

        Default implementation applies `transform` to each item.
        Override for optimized batch processing.

        Args:
            data_list: List of input data points.

        Returns:
            List of feature dictionaries, one per input.
        """
        return [self.transform(data) for data in data_list]

    def validate_output(self, features: dict[str, Any]) -> bool:
        """Validate that output contains all expected features.

        Args:
            features: Feature dictionary to validate.

        Returns:
            True if all expected features are present.

        Raises:
            ValueError: If required features are missing.
        """
        missing = set(self.feature_names) - set(features.keys())
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return True

    def get_feature_schema(self) -> dict[str, str]:
        """Get schema describing feature names and types.

        Returns:
            Dictionary mapping feature names to their data types.
        """
        # Default implementation - subclasses should override
        return dict.fromkeys(self.feature_names, "any")

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(entity_type={self.entity_type}, features={len(self.feature_names)})>"
