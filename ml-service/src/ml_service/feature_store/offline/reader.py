"""Offline Store Reader - Read features with DuckDB for efficient queries."""

from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


class OfflineStoreReader:
    """Read feature data from Parquet files using DuckDB.

    Supports point-in-time correct joins to prevent data leakage during
    training by ensuring only features available at each event time are used.

    Example:
        >>> reader = OfflineStoreReader("/data/features")
        >>> df = reader.get_features(
        ...     feature_group="text_features",
        ...     entity_ids=["123", "456"],
        ...     start_date=datetime(2024, 1, 1),
        ...     end_date=datetime(2024, 1, 31),
        ... )
    """

    def __init__(self, base_path: str | Path) -> None:
        """Initialize the offline store reader.

        Args:
            base_path: Root directory for feature storage.
        """
        self.base_path = Path(base_path)
        self._conn = duckdb.connect(":memory:")

    def get_features(
        self,
        feature_group: str,
        version: int = 1,
        entity_ids: list[str] | None = None,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        columns: list[str] | None = None,
    ) -> pa.Table:
        """Read features from Parquet files.

        Args:
            feature_group: Name of the feature group.
            version: Feature group version.
            entity_ids: Optional list of entity IDs to filter.
            start_date: Optional start date for filtering.
            end_date: Optional end date for filtering.
            columns: Optional list of columns to select.

        Returns:
            PyArrow Table with the requested features.

        Raises:
            FileNotFoundError: If no feature files are found.
        """
        version_dir = self.base_path / feature_group / f"v{version}"
        if not version_dir.exists():
            raise FileNotFoundError(f"Feature group not found: {feature_group} v{version}")

        # Find relevant parquet files
        parquet_files = self._find_parquet_files(
            version_dir,
            start_date,
            end_date,
        )

        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found for {feature_group} v{version} "
                f"between {start_date} and {end_date}"
            )

        # Build and execute query
        return self._query_parquet_files(
            parquet_files,
            entity_ids=entity_ids,
            columns=columns,
        )

    def point_in_time_join(
        self,
        entity_timestamps: list[dict[str, Any]],
        feature_group: str,
        version: int = 1,
    ) -> pa.Table:
        """Perform point-in-time correct join.

        For each entity and timestamp, retrieves the most recent feature values
        that were available at that time (event_timestamp <= entity_timestamp).
        This prevents data leakage from future features.

        Args:
            entity_timestamps: List of dicts with 'entity_id' and 'event_timestamp'.
            feature_group: Name of the feature group.
            version: Feature group version.

        Returns:
            PyArrow Table with features joined at the correct point in time.

        Example:
            >>> entities = [
            ...     {"entity_id": "123", "event_timestamp": datetime(2024, 1, 15)},
            ...     {"entity_id": "456", "event_timestamp": datetime(2024, 1, 20)},
            ... ]
            >>> result = reader.point_in_time_join(entities, "text_features")
        """
        version_dir = self.base_path / feature_group / f"v{version}"
        if not version_dir.exists():
            raise FileNotFoundError(f"Feature group not found: {feature_group} v{version}")

        # Get all parquet files
        parquet_files = list(version_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found for {feature_group} v{version}")

        # Create entity dataframe
        entity_df = pa.Table.from_pylist(entity_timestamps)

        # Read all feature files
        feature_tables = [pq.read_table(f) for f in parquet_files]
        feature_df = pa.concat_tables(feature_tables)

        # Register tables with DuckDB
        self._conn.register("entity_df", entity_df)
        self._conn.register("feature_df", feature_df)

        # Perform ASOF JOIN using DuckDB
        # This joins each entity with the most recent feature before its timestamp
        query = """
            SELECT
                e.entity_id,
                e.event_timestamp AS request_timestamp,
                f.* EXCLUDE (entity_id, event_timestamp)
            FROM entity_df e
            ASOF JOIN feature_df f
                ON e.entity_id = f.entity_id
                AND e.event_timestamp >= f.event_timestamp
            ORDER BY e.entity_id, e.event_timestamp
        """

        result = self._conn.execute(query).fetch_arrow_table()

        # Cleanup
        self._conn.unregister("entity_df")
        self._conn.unregister("feature_df")

        return result

    def get_latest_features(
        self,
        feature_group: str,
        entity_ids: list[str],
        version: int = 1,
    ) -> pa.Table:
        """Get the most recent features for each entity.

        Args:
            feature_group: Name of the feature group.
            entity_ids: List of entity IDs.
            version: Feature group version.

        Returns:
            PyArrow Table with the latest features for each entity.
        """
        version_dir = self.base_path / feature_group / f"v{version}"
        if not version_dir.exists():
            raise FileNotFoundError(f"Feature group not found: {feature_group} v{version}")

        parquet_files = list(version_dir.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found for {feature_group} v{version}")

        # Read all files
        tables = [pq.read_table(f) for f in parquet_files]
        combined = pa.concat_tables(tables)

        self._conn.register("features", combined)

        # Build entity filter
        entity_list = ", ".join(f"'{eid}'" for eid in entity_ids)

        query = f"""
            SELECT *
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY entity_id
                        ORDER BY event_timestamp DESC
                    ) as rn
                FROM features
                WHERE entity_id IN ({entity_list})
            )
            WHERE rn = 1
        """

        result = self._conn.execute(query).fetch_arrow_table()

        # Remove the row number column
        if "rn" in result.column_names:
            rn_idx = result.column_names.index("rn")
            result = result.remove_column(rn_idx)

        self._conn.unregister("features")

        return result

    def get_feature_statistics(
        self,
        feature_group: str,
        version: int = 1,
    ) -> dict[str, Any]:
        """Get statistics for a feature group.

        Args:
            feature_group: Name of the feature group.
            version: Feature group version.

        Returns:
            Dictionary with statistics (count, date range, etc.).
        """
        version_dir = self.base_path / feature_group / f"v{version}"
        if not version_dir.exists():
            raise FileNotFoundError(f"Feature group not found: {feature_group} v{version}")

        parquet_files = list(version_dir.glob("*.parquet"))
        if not parquet_files:
            return {
                "row_count": 0,
                "file_count": 0,
                "date_range": None,
                "entity_count": 0,
            }

        tables = [pq.read_table(f) for f in parquet_files]
        combined = pa.concat_tables(tables)

        self._conn.register("features", combined)

        stats_query = """
            SELECT
                COUNT(*) as row_count,
                COUNT(DISTINCT entity_id) as entity_count,
                MIN(event_timestamp) as min_timestamp,
                MAX(event_timestamp) as max_timestamp
            FROM features
        """

        result = self._conn.execute(stats_query).fetchone()
        self._conn.unregister("features")

        return {
            "row_count": result[0],
            "entity_count": result[1],
            "min_timestamp": result[2],
            "max_timestamp": result[3],
            "file_count": len(parquet_files),
            "date_range": sorted([f.stem for f in parquet_files]),
        }

    def _find_parquet_files(
        self,
        version_dir: Path,
        start_date: datetime | None,
        end_date: datetime | None,
    ) -> list[Path]:
        """Find parquet files within the date range.

        Args:
            version_dir: Directory containing version's parquet files.
            start_date: Optional start date filter.
            end_date: Optional end date filter.

        Returns:
            List of paths to relevant parquet files.
        """
        all_files = list(version_dir.glob("*.parquet"))

        if start_date is None and end_date is None:
            return all_files

        filtered_files = []
        for file in all_files:
            date_str = file.stem  # YYYY-MM-DD
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d")

                if start_date and file_date.date() < start_date.date():
                    continue
                if end_date and file_date.date() > end_date.date():
                    continue

                filtered_files.append(file)
            except ValueError:
                # Skip files that don't match date format
                continue

        return filtered_files

    def _query_parquet_files(
        self,
        files: list[Path],
        entity_ids: list[str] | None = None,
        columns: list[str] | None = None,
    ) -> pa.Table:
        """Query parquet files with optional filtering.

        Args:
            files: List of parquet file paths.
            entity_ids: Optional entity ID filter.
            columns: Optional column selection.

        Returns:
            PyArrow Table with query results.
        """
        # Read and combine tables
        tables = []
        for file in files:
            table = pq.read_table(file, columns=columns)
            tables.append(table)

        if not tables:
            raise FileNotFoundError("No data found in parquet files")

        combined = pa.concat_tables(tables)

        # Filter by entity_ids if specified
        if entity_ids:
            self._conn.register("data", combined)
            entity_list = ", ".join(f"'{eid}'" for eid in entity_ids)
            query = f"SELECT * FROM data WHERE entity_id IN ({entity_list})"
            combined = self._conn.execute(query).fetch_arrow_table()
            self._conn.unregister("data")

        return combined

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()

    def __enter__(self) -> "OfflineStoreReader":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
