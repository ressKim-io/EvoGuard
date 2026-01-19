"""Offline Store Writer - Write features to Parquet files."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


class OfflineStoreWriter:
    """Write feature data to Parquet files.

    Features are stored in a hierarchical directory structure:
        {base_path}/{feature_group}/v{version}/{date}.parquet

    Example:
        >>> writer = OfflineStoreWriter("/data/features")
        >>> writer.write_features(
        ...     feature_group="text_features",
        ...     features=[
        ...         {"entity_id": "123", "text_length": 100, "word_count": 20}
        ...     ],
        ...     schema=text_features_schema,
        ... )
    """

    def __init__(self, base_path: str | Path) -> None:
        """Initialize the offline store writer.

        Args:
            base_path: Root directory for feature storage.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def write_features(
        self,
        feature_group: str,
        features: list[dict[str, Any]],
        schema: pa.Schema,
        version: int = 1,
        partition_date: datetime | None = None,
    ) -> Path:
        """Write features to a Parquet file.

        Args:
            feature_group: Name of the feature group (e.g., "text_features").
            features: List of feature dictionaries.
            schema: PyArrow schema for the features.
            version: Feature group version.
            partition_date: Date for partitioning. Defaults to today.

        Returns:
            Path to the written Parquet file.

        Raises:
            ValueError: If features list is empty.
        """
        if not features:
            raise ValueError("Features list cannot be empty")

        partition_date = partition_date or datetime.now(UTC)
        date_str = partition_date.strftime("%Y-%m-%d")

        # Create directory structure
        output_dir = self.base_path / feature_group / f"v{version}"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"{date_str}.parquet"

        # Add metadata timestamps if not present
        for feature in features:
            if "event_timestamp" not in feature:
                feature["event_timestamp"] = partition_date
            if "created_at" not in feature:
                feature["created_at"] = datetime.now(UTC)

        # Create PyArrow table and write
        table = pa.Table.from_pylist(features, schema=schema)
        pq.write_table(
            table,
            output_path,
            compression="snappy",
            write_statistics=True,
        )

        return output_path

    def append_features(
        self,
        feature_group: str,
        features: list[dict[str, Any]],
        schema: pa.Schema,
        version: int = 1,
        partition_date: datetime | None = None,
    ) -> Path:
        """Append features to an existing Parquet file or create new.

        If the file exists, reads existing data and appends new features.

        Args:
            feature_group: Name of the feature group.
            features: List of feature dictionaries to append.
            schema: PyArrow schema for the features.
            version: Feature group version.
            partition_date: Date for partitioning.

        Returns:
            Path to the updated Parquet file.
        """
        if not features:
            raise ValueError("Features list cannot be empty")

        partition_date = partition_date or datetime.now(UTC)
        date_str = partition_date.strftime("%Y-%m-%d")

        output_dir = self.base_path / feature_group / f"v{version}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{date_str}.parquet"

        # Add metadata timestamps
        for feature in features:
            if "event_timestamp" not in feature:
                feature["event_timestamp"] = partition_date
            if "created_at" not in feature:
                feature["created_at"] = datetime.now(UTC)

        new_table = pa.Table.from_pylist(features, schema=schema)

        # If file exists, concatenate with existing data
        if output_path.exists():
            existing_table = pq.read_table(output_path)
            combined_table = pa.concat_tables([existing_table, new_table])
        else:
            combined_table = new_table

        pq.write_table(
            combined_table,
            output_path,
            compression="snappy",
            write_statistics=True,
        )

        return output_path

    def get_feature_path(
        self,
        feature_group: str,
        version: int = 1,
        partition_date: datetime | None = None,
    ) -> Path:
        """Get the path for a feature file.

        Args:
            feature_group: Name of the feature group.
            version: Feature group version.
            partition_date: Date partition.

        Returns:
            Path to the feature file.
        """
        partition_date = partition_date or datetime.now(UTC)
        date_str = partition_date.strftime("%Y-%m-%d")
        return self.base_path / feature_group / f"v{version}" / f"{date_str}.parquet"

    def list_partitions(
        self,
        feature_group: str,
        version: int = 1,
    ) -> list[str]:
        """List available date partitions for a feature group.

        Args:
            feature_group: Name of the feature group.
            version: Feature group version.

        Returns:
            List of date strings (YYYY-MM-DD) for available partitions.
        """
        version_dir = self.base_path / feature_group / f"v{version}"
        if not version_dir.exists():
            return []

        partitions = []
        for file in version_dir.glob("*.parquet"):
            # Extract date from filename (YYYY-MM-DD.parquet)
            date_str = file.stem
            partitions.append(date_str)

        return sorted(partitions)

    def list_versions(self, feature_group: str) -> list[int]:
        """List available versions for a feature group.

        Args:
            feature_group: Name of the feature group.

        Returns:
            List of version numbers.
        """
        group_dir = self.base_path / feature_group
        if not group_dir.exists():
            return []

        versions = []
        for dir_path in group_dir.iterdir():
            if dir_path.is_dir() and dir_path.name.startswith("v"):
                try:
                    version = int(dir_path.name[1:])
                    versions.append(version)
                except ValueError:
                    continue

        return sorted(versions)


# Predefined schemas for EvoGuard features
TEXT_FEATURES_SCHEMA = pa.schema([
    pa.field("entity_id", pa.string()),
    pa.field("event_timestamp", pa.timestamp("us", tz="UTC")),
    pa.field("text_length", pa.int32()),
    pa.field("word_count", pa.int32()),
    pa.field("unicode_ratio", pa.float32()),
    pa.field("special_char_ratio", pa.float32()),
    pa.field("repeated_char_ratio", pa.float32()),
    pa.field("created_at", pa.timestamp("us", tz="UTC")),
])

BATTLE_FEATURES_SCHEMA = pa.schema([
    pa.field("entity_id", pa.string()),
    pa.field("event_timestamp", pa.timestamp("us", tz="UTC")),
    pa.field("detection_rate", pa.float32()),
    pa.field("evasion_rate", pa.float32()),
    pa.field("avg_confidence", pa.float32()),
    pa.field("total_rounds", pa.int32()),
    pa.field("created_at", pa.timestamp("us", tz="UTC")),
])
