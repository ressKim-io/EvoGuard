"""MLflow Feature Logger - Log feature metadata to MLflow runs."""

import json
import logging
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

logger = logging.getLogger(__name__)


class FeatureLogger:
    """Log feature metadata to MLflow experiments.

    This class provides utilities for logging feature group information,
    schemas, and statistics to MLflow runs for experiment tracking.

    Note: This implementation doesn't require mlflow as a dependency.
    It prepares the metadata in a format compatible with MLflow logging.
    The actual logging should be done with mlflow client when available.

    Example:
        >>> feature_logger = FeatureLogger()
        >>> metadata = feature_logger.prepare_feature_metadata(
        ...     feature_group="text_features",
        ...     feature_group_version=1,
        ...     feature_schema={"text_length": "int", "word_count": "int"},
        ... )
        >>> # Use with mlflow:
        >>> # mlflow.log_params(metadata["params"])
        >>> # mlflow.log_dict(metadata["schema"], "feature_schema.json")
    """

    def prepare_feature_metadata(
        self,
        feature_group: str,
        feature_group_version: int,
        feature_schema: dict[str, str],
        feature_group_id: UUID | str | None = None,
        entity_type: str | None = None,
        transformer_version: str | None = None,
    ) -> dict[str, Any]:
        """Prepare feature metadata for MLflow logging.

        Args:
            feature_group: Name of the feature group.
            feature_group_version: Version of the feature group.
            feature_schema: Dictionary mapping feature names to types.
            feature_group_id: Optional UUID of the feature group.
            entity_type: Optional entity type (e.g., "text", "battle").
            transformer_version: Optional version of the transformer used.

        Returns:
            Dictionary with 'params', 'schema', and 'tags' for MLflow logging.
        """
        params = {
            "feature_group": feature_group,
            "feature_group_version": str(feature_group_version),
            "feature_count": str(len(feature_schema)),
        }

        if entity_type:
            params["entity_type"] = entity_type
        if transformer_version:
            params["transformer_version"] = transformer_version
        if feature_group_id:
            params["feature_group_id"] = str(feature_group_id)

        tags = {
            "mlflow.note.content": f"Features: {feature_group} v{feature_group_version}",
            "feature_store.group": feature_group,
            "feature_store.version": str(feature_group_version),
        }

        schema_artifact = {
            "feature_group": feature_group,
            "version": feature_group_version,
            "features": feature_schema,
            "feature_names": list(feature_schema.keys()),
            "logged_at": datetime.now(UTC).isoformat(),
        }

        return {
            "params": params,
            "schema": schema_artifact,
            "tags": tags,
        }

    def prepare_training_features_info(
        self,
        feature_group: str,
        feature_group_version: int,
        row_count: int,
        entity_count: int,
        date_range: tuple[str, str] | None = None,
        statistics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Prepare training feature statistics for MLflow logging.

        Args:
            feature_group: Name of the feature group.
            feature_group_version: Version of the feature group.
            row_count: Total number of feature rows used.
            entity_count: Number of unique entities.
            date_range: Optional tuple of (start_date, end_date).
            statistics: Optional additional statistics.

        Returns:
            Dictionary with metrics and artifacts for MLflow logging.
        """
        metrics = {
            "feature_row_count": row_count,
            "feature_entity_count": entity_count,
        }

        artifact = {
            "feature_group": feature_group,
            "version": feature_group_version,
            "row_count": row_count,
            "entity_count": entity_count,
            "logged_at": datetime.now(UTC).isoformat(),
        }

        if date_range:
            artifact["date_range"] = {
                "start": date_range[0],
                "end": date_range[1],
            }
            metrics["feature_date_span_days"] = self._calculate_date_span(date_range)

        if statistics:
            artifact["statistics"] = statistics

        return {
            "metrics": metrics,
            "artifact": artifact,
        }

    def prepare_feature_importance(
        self,
        feature_names: list[str],
        importance_values: list[float],
        method: str = "default",
    ) -> dict[str, Any]:
        """Prepare feature importance for MLflow logging.

        Args:
            feature_names: List of feature names.
            importance_values: List of importance values (same order as names).
            method: Method used to calculate importance.

        Returns:
            Dictionary with metrics and artifact for MLflow logging.
        """
        if len(feature_names) != len(importance_values):
            raise ValueError("Feature names and importance values must have same length")

        # Create metrics for top features
        sorted_importance = sorted(
            zip(feature_names, importance_values, strict=True),
            key=lambda x: x[1],
            reverse=True,
        )

        metrics = {}
        for i, (_name, value) in enumerate(sorted_importance[:10]):
            metrics[f"feature_importance_top{i + 1}"] = value

        artifact = {
            "method": method,
            "features": [
                {"name": name, "importance": value}
                for name, value in sorted_importance
            ],
            "logged_at": datetime.now(UTC).isoformat(),
        }

        return {
            "metrics": metrics,
            "artifact": artifact,
        }

    def _calculate_date_span(self, date_range: tuple[str, str]) -> int:
        """Calculate number of days between date range.

        Args:
            date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format.

        Returns:
            Number of days.
        """
        try:
            start = datetime.strptime(date_range[0], "%Y-%m-%d")
            end = datetime.strptime(date_range[1], "%Y-%m-%d")
            return (end - start).days
        except (ValueError, TypeError):
            return 0

    def format_for_mlflow_log_dict(self, data: dict[str, Any]) -> str:
        """Format dictionary for mlflow.log_dict compatibility.

        Args:
            data: Dictionary to format.

        Returns:
            JSON string.
        """
        return json.dumps(data, indent=2, default=str)
