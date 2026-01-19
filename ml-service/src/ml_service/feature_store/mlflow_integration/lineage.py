"""Feature Lineage Tracking - Track model-feature relationships."""

import logging
from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import UUID

logger = logging.getLogger(__name__)


class FeatureMissingError(Exception):
    """Raised when a required feature is missing during validation."""

    pass


class FeatureTypeMismatchError(Exception):
    """Raised when a feature has an unexpected type."""

    pass


class FeatureLineageTracker:
    """Track lineage between ML models and feature groups.

    This class provides utilities for creating and querying lineage records
    that connect ML models to the features they were trained on.

    Note: This is a standalone utility that doesn't require database access.
    For persistence, use the ModelLineageRepository with the registry database.

    Example:
        >>> tracker = FeatureLineageTracker()
        >>> lineage_record = tracker.create_lineage_record(
        ...     mlflow_run_id="abc123",
        ...     mlflow_model_name="text_classifier",
        ...     mlflow_model_version=1,
        ...     feature_group_id=uuid4(),
        ...     feature_group_version=1,
        ...     feature_schema={"text_length": "int", "word_count": "int"},
        ... )
    """

    def create_lineage_record(
        self,
        mlflow_run_id: str,
        feature_group_id: UUID | str,
        feature_group_version: int,
        feature_schema: dict[str, str],
        mlflow_model_name: str | None = None,
        mlflow_model_version: int | None = None,
        feature_names: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create a lineage record for model-feature relationship.

        Args:
            mlflow_run_id: MLflow run ID.
            feature_group_id: UUID of the feature group.
            feature_group_version: Version of the feature group.
            feature_schema: Schema mapping feature names to types.
            mlflow_model_name: Optional registered model name.
            mlflow_model_version: Optional model version.
            feature_names: Optional list of feature names used (subset of schema).

        Returns:
            Lineage record dictionary ready for storage.
        """
        schema_snapshot = {
            "features": [
                {"name": name, "data_type": dtype}
                for name, dtype in feature_schema.items()
            ],
            "version": feature_group_version,
            "captured_at": datetime.now(UTC).isoformat(),
        }

        if feature_names:
            schema_snapshot["used_features"] = feature_names

        return {
            "mlflow_run_id": mlflow_run_id,
            "mlflow_model_name": mlflow_model_name,
            "mlflow_model_version": mlflow_model_version,
            "feature_group_id": str(feature_group_id),
            "feature_group_version": feature_group_version,
            "feature_schema_snapshot": schema_snapshot,
            "created_at": datetime.now(UTC).isoformat(),
        }

    def extract_schema_from_lineage(
        self,
        lineage_record: dict[str, Any],
    ) -> dict[str, str]:
        """Extract feature schema from a lineage record.

        Args:
            lineage_record: Lineage record dictionary.

        Returns:
            Dictionary mapping feature names to types.
        """
        schema_snapshot = lineage_record.get("feature_schema_snapshot", {})
        features = schema_snapshot.get("features", [])

        return {f["name"]: f["data_type"] for f in features}

    def get_used_features(
        self,
        lineage_record: dict[str, Any],
    ) -> list[str]:
        """Get list of features used from a lineage record.

        Args:
            lineage_record: Lineage record dictionary.

        Returns:
            List of feature names used by the model.
        """
        schema_snapshot = lineage_record.get("feature_schema_snapshot", {})

        # Check for explicit used_features list
        if "used_features" in schema_snapshot:
            return schema_snapshot["used_features"]

        # Otherwise return all features
        features = schema_snapshot.get("features", [])
        return [f["name"] for f in features]


class FeatureSchemaValidator:
    """Validate feature schemas during model serving.

    Ensures that features provided for inference match the schema
    that the model was trained on, preventing silent failures.

    Example:
        >>> validator = FeatureSchemaValidator(expected_schema)
        >>> validator.validate(current_features)  # Raises if mismatch
    """

    # Type compatibility mapping
    TYPE_COMPATIBILITY: ClassVar[dict[str, set[str]]] = {
        "int": {"int", "int32", "int64", "integer"},
        "float": {"float", "float32", "float64", "double", "number"},
        "string": {"str", "string", "text"},
        "bool": {"bool", "boolean"},
    }

    def __init__(self, expected_schema: dict[str, str]) -> None:
        """Initialize the validator with expected schema.

        Args:
            expected_schema: Dictionary mapping feature names to expected types.
        """
        self.expected_schema = expected_schema

    def validate(
        self,
        features: dict[str, Any],
        strict: bool = True,
    ) -> bool:
        """Validate features against expected schema.

        Args:
            features: Dictionary of feature values to validate.
            strict: If True, raise exceptions on mismatch. If False, log warnings.

        Returns:
            True if validation passes.

        Raises:
            FeatureMissingError: If a required feature is missing (strict mode).
            FeatureTypeMismatchError: If a feature has wrong type (strict mode).
        """
        errors: list[str] = []

        # Check for missing features
        for name in self.expected_schema:
            if name not in features:
                error_msg = f"Missing feature: {name}"
                if strict:
                    raise FeatureMissingError(error_msg)
                errors.append(error_msg)

        # Check types for present features
        for name, expected_type in self.expected_schema.items():
            if name not in features:
                continue

            actual_value = features[name]
            actual_type = type(actual_value).__name__

            if not self._types_compatible(expected_type, actual_type):
                error_msg = (
                    f"Feature '{name}': expected {expected_type}, got {actual_type}"
                )
                if strict:
                    raise FeatureTypeMismatchError(error_msg)
                errors.append(error_msg)

        if errors:
            logger.warning(f"Schema validation warnings: {errors}")

        return len(errors) == 0

    def validate_batch(
        self,
        features_batch: list[dict[str, Any]],
        strict: bool = True,
    ) -> list[bool]:
        """Validate a batch of feature dictionaries.

        Args:
            features_batch: List of feature dictionaries.
            strict: If True, raise on first error.

        Returns:
            List of validation results for each record.
        """
        results = []
        for i, features in enumerate(features_batch):
            try:
                results.append(self.validate(features, strict=strict))
            except (FeatureMissingError, FeatureTypeMismatchError) as e:
                logger.error(f"Validation failed for record {i}: {e}")
                if strict:
                    raise
                results.append(False)
        return results

    def get_missing_features(self, features: dict[str, Any]) -> list[str]:
        """Get list of missing features.

        Args:
            features: Feature dictionary to check.

        Returns:
            List of missing feature names.
        """
        return [name for name in self.expected_schema if name not in features]

    def get_extra_features(self, features: dict[str, Any]) -> list[str]:
        """Get list of unexpected extra features.

        Args:
            features: Feature dictionary to check.

        Returns:
            List of extra feature names not in schema.
        """
        return [name for name in features if name not in self.expected_schema]

    def _types_compatible(self, expected: str, actual: str) -> bool:
        """Check if actual type is compatible with expected type.

        Args:
            expected: Expected type string.
            actual: Actual type string.

        Returns:
            True if types are compatible.
        """
        expected_lower = expected.lower()
        actual_lower = actual.lower()

        # Direct match
        if expected_lower == actual_lower:
            return True

        # Check compatibility groups
        for _base_type, compatible_types in self.TYPE_COMPATIBILITY.items():
            if expected_lower in compatible_types and actual_lower in compatible_types:
                return True

        # Special case: int is compatible with float
        if expected_lower in self.TYPE_COMPATIBILITY.get("float", set()):
            if actual_lower in self.TYPE_COMPATIBILITY.get("int", set()):
                return True

        return False

    @classmethod
    def from_lineage_record(
        cls,
        lineage_record: dict[str, Any],
    ) -> "FeatureSchemaValidator":
        """Create validator from a lineage record.

        Args:
            lineage_record: Lineage record with schema snapshot.

        Returns:
            FeatureSchemaValidator instance.
        """
        tracker = FeatureLineageTracker()
        schema = tracker.extract_schema_from_lineage(lineage_record)
        return cls(schema)
