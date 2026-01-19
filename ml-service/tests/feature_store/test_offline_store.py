"""Tests for Offline Store (Parquet + DuckDB)."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from ml_service.feature_store.offline import OfflineStoreReader, OfflineStoreWriter
from ml_service.feature_store.offline.writer import TEXT_FEATURES_SCHEMA


class TestOfflineStoreWriter:
    """Tests for OfflineStoreWriter."""

    def test_write_features(self, tmp_path: Path) -> None:
        """Test writing features to Parquet."""
        writer = OfflineStoreWriter(tmp_path)

        features = [
            {
                "entity_id": "entity-1",
                "text_length": 100,
                "word_count": 20,
                "unicode_ratio": 0.1,
                "special_char_ratio": 0.05,
                "repeated_char_ratio": 0.02,
            },
            {
                "entity_id": "entity-2",
                "text_length": 200,
                "word_count": 40,
                "unicode_ratio": 0.15,
                "special_char_ratio": 0.08,
                "repeated_char_ratio": 0.03,
            },
        ]

        partition_date = datetime(2024, 1, 15, tzinfo=UTC)
        output_path = writer.write_features(
            feature_group="text_features",
            features=features,
            schema=TEXT_FEATURES_SCHEMA,
            version=1,
            partition_date=partition_date,
        )

        assert output_path.exists()
        assert output_path.name == "2024-01-15.parquet"
        assert "text_features/v1" in str(output_path)

    def test_write_features_empty_raises(self, tmp_path: Path) -> None:
        """Test that empty features list raises ValueError."""
        writer = OfflineStoreWriter(tmp_path)

        with pytest.raises(ValueError, match="empty"):
            writer.write_features(
                feature_group="text_features",
                features=[],
                schema=TEXT_FEATURES_SCHEMA,
            )

    def test_append_features(self, tmp_path: Path) -> None:
        """Test appending features to existing file."""
        writer = OfflineStoreWriter(tmp_path)
        partition_date = datetime(2024, 1, 15, tzinfo=UTC)

        # Write first batch
        features1 = [
            {
                "entity_id": "entity-1",
                "text_length": 100,
                "word_count": 20,
                "unicode_ratio": 0.1,
                "special_char_ratio": 0.05,
                "repeated_char_ratio": 0.02,
            },
        ]
        writer.write_features(
            feature_group="text_features",
            features=features1,
            schema=TEXT_FEATURES_SCHEMA,
            partition_date=partition_date,
        )

        # Append second batch
        features2 = [
            {
                "entity_id": "entity-2",
                "text_length": 200,
                "word_count": 40,
                "unicode_ratio": 0.15,
                "special_char_ratio": 0.08,
                "repeated_char_ratio": 0.03,
            },
        ]
        writer.append_features(
            feature_group="text_features",
            features=features2,
            schema=TEXT_FEATURES_SCHEMA,
            partition_date=partition_date,
        )

        # Read and verify
        reader = OfflineStoreReader(tmp_path)
        table = reader.get_features("text_features", version=1)
        assert table.num_rows == 2

    def test_list_partitions(self, tmp_path: Path) -> None:
        """Test listing available partitions."""
        writer = OfflineStoreWriter(tmp_path)

        features = [
            {
                "entity_id": "entity-1",
                "text_length": 100,
                "word_count": 20,
                "unicode_ratio": 0.1,
                "special_char_ratio": 0.05,
                "repeated_char_ratio": 0.02,
            },
        ]

        # Write to multiple dates
        for day in [10, 15, 20]:
            partition_date = datetime(2024, 1, day, tzinfo=UTC)
            writer.write_features(
                feature_group="text_features",
                features=features,
                schema=TEXT_FEATURES_SCHEMA,
                partition_date=partition_date,
            )

        partitions = writer.list_partitions("text_features", version=1)
        assert partitions == ["2024-01-10", "2024-01-15", "2024-01-20"]

    def test_list_versions(self, tmp_path: Path) -> None:
        """Test listing available versions."""
        writer = OfflineStoreWriter(tmp_path)

        features = [
            {
                "entity_id": "entity-1",
                "text_length": 100,
                "word_count": 20,
                "unicode_ratio": 0.1,
                "special_char_ratio": 0.05,
                "repeated_char_ratio": 0.02,
            },
        ]

        # Write to multiple versions
        for version in [1, 2, 3]:
            writer.write_features(
                feature_group="text_features",
                features=features,
                schema=TEXT_FEATURES_SCHEMA,
                version=version,
            )

        versions = writer.list_versions("text_features")
        assert versions == [1, 2, 3]


class TestOfflineStoreReader:
    """Tests for OfflineStoreReader."""

    @pytest.fixture
    def populated_store(self, tmp_path: Path) -> Path:
        """Create a populated feature store for testing."""
        writer = OfflineStoreWriter(tmp_path)

        # Create features for multiple entities and dates
        for day in range(1, 11):
            partition_date = datetime(2024, 1, day, tzinfo=UTC)
            features = [
                {
                    "entity_id": f"entity-{i}",
                    "event_timestamp": partition_date,
                    "text_length": 100 * i + day,
                    "word_count": 20 * i,
                    "unicode_ratio": 0.1 * i,
                    "special_char_ratio": 0.05,
                    "repeated_char_ratio": 0.02,
                    "created_at": partition_date,
                }
                for i in range(1, 4)
            ]
            writer.write_features(
                feature_group="text_features",
                features=features,
                schema=TEXT_FEATURES_SCHEMA,
                partition_date=partition_date,
            )

        return tmp_path

    def test_get_features(self, populated_store: Path) -> None:
        """Test reading all features."""
        reader = OfflineStoreReader(populated_store)
        table = reader.get_features("text_features", version=1)

        # 10 days * 3 entities = 30 rows
        assert table.num_rows == 30
        assert "entity_id" in table.column_names
        assert "text_length" in table.column_names

    def test_get_features_with_date_filter(self, populated_store: Path) -> None:
        """Test reading features with date filter."""
        reader = OfflineStoreReader(populated_store)

        start_date = datetime(2024, 1, 3, tzinfo=UTC)
        end_date = datetime(2024, 1, 5, tzinfo=UTC)

        table = reader.get_features(
            "text_features",
            version=1,
            start_date=start_date,
            end_date=end_date,
        )

        # 3 days * 3 entities = 9 rows
        assert table.num_rows == 9

    def test_get_features_with_entity_filter(self, populated_store: Path) -> None:
        """Test reading features for specific entities."""
        reader = OfflineStoreReader(populated_store)

        table = reader.get_features(
            "text_features",
            version=1,
            entity_ids=["entity-1", "entity-2"],
        )

        # 10 days * 2 entities = 20 rows
        assert table.num_rows == 20

    def test_get_features_not_found(self, tmp_path: Path) -> None:
        """Test error when feature group not found."""
        reader = OfflineStoreReader(tmp_path)

        with pytest.raises(FileNotFoundError):
            reader.get_features("nonexistent", version=1)

    def test_point_in_time_join(self, tmp_path: Path) -> None:
        """Test point-in-time correct join."""
        writer = OfflineStoreWriter(tmp_path)

        # Create features at different times for same entity
        for day, value in [(1, 100), (5, 150), (10, 200)]:
            partition_date = datetime(2024, 1, day, tzinfo=UTC)
            features = [
                {
                    "entity_id": "entity-1",
                    "event_timestamp": partition_date,
                    "text_length": value,
                    "word_count": 20,
                    "unicode_ratio": 0.1,
                    "special_char_ratio": 0.05,
                    "repeated_char_ratio": 0.02,
                    "created_at": partition_date,
                },
            ]
            writer.write_features(
                feature_group="text_features",
                features=features,
                schema=TEXT_FEATURES_SCHEMA,
                partition_date=partition_date,
            )

        reader = OfflineStoreReader(tmp_path)

        # Query at different points in time
        entity_timestamps = [
            {"entity_id": "entity-1", "event_timestamp": datetime(2024, 1, 3, tzinfo=UTC)},
            {"entity_id": "entity-1", "event_timestamp": datetime(2024, 1, 7, tzinfo=UTC)},
            {"entity_id": "entity-1", "event_timestamp": datetime(2024, 1, 15, tzinfo=UTC)},
        ]

        result = reader.point_in_time_join(entity_timestamps, "text_features")

        # Should get 3 rows
        assert result.num_rows == 3

        # Convert to python for easier assertion
        text_lengths = result.column("text_length").to_pylist()

        # At Jan 3, should get Jan 1 value (100)
        # At Jan 7, should get Jan 5 value (150)
        # At Jan 15, should get Jan 10 value (200)
        assert text_lengths == [100, 150, 200]

    def test_get_latest_features(self, tmp_path: Path) -> None:
        """Test getting latest features for entities."""
        writer = OfflineStoreWriter(tmp_path)

        # Create multiple versions for same entity
        for day, value in [(1, 100), (5, 150), (10, 200)]:
            partition_date = datetime(2024, 1, day, tzinfo=UTC)
            features = [
                {
                    "entity_id": "entity-1",
                    "event_timestamp": partition_date,
                    "text_length": value,
                    "word_count": 20,
                    "unicode_ratio": 0.1,
                    "special_char_ratio": 0.05,
                    "repeated_char_ratio": 0.02,
                    "created_at": partition_date,
                },
            ]
            writer.write_features(
                feature_group="text_features",
                features=features,
                schema=TEXT_FEATURES_SCHEMA,
                partition_date=partition_date,
            )

        reader = OfflineStoreReader(tmp_path)
        result = reader.get_latest_features(
            "text_features",
            entity_ids=["entity-1"],
        )

        assert result.num_rows == 1
        assert result.column("text_length").to_pylist()[0] == 200

    def test_get_feature_statistics(self, populated_store: Path) -> None:
        """Test getting feature statistics."""
        reader = OfflineStoreReader(populated_store)
        stats = reader.get_feature_statistics("text_features", version=1)

        assert stats["row_count"] == 30
        assert stats["entity_count"] == 3
        assert stats["file_count"] == 10
        assert len(stats["date_range"]) == 10

    def test_context_manager(self, populated_store: Path) -> None:
        """Test using reader as context manager."""
        with OfflineStoreReader(populated_store) as reader:
            table = reader.get_features("text_features", version=1)
            assert table.num_rows == 30
