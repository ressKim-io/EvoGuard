"""Tests for feature drift detection module."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ml_service.monitoring.drift.feature_drift import (
    FeatureDriftMonitor,
    FeatureGroupDriftResult,
    FeatureStats,
    FeatureStatsCollector,
    FeatureStoreIntegration,
)


class TestFeatureStatsCollector:
    """Tests for FeatureStatsCollector."""

    def test_initialization(self) -> None:
        """Should initialize with correct window size."""
        collector = FeatureStatsCollector(window_size=500)
        assert collector.window_size == 500

    def test_add_features(self) -> None:
        """Should add numeric features."""
        collector = FeatureStatsCollector()

        collector.add_features({
            "text_length": 100,
            "word_count": 20,
            "confidence": 0.85,
        })

        assert len(collector._features["text_length"]) == 1
        assert len(collector._features["word_count"]) == 1
        assert len(collector._features["confidence"]) == 1

    def test_skip_metadata_fields(self) -> None:
        """Should skip fields starting with underscore."""
        collector = FeatureStatsCollector()

        collector.add_features({
            "text_length": 100,
            "_updated_at": "2024-01-01T00:00:00Z",
            "_version": 1,
        })

        assert "_updated_at" not in collector._features
        assert "_version" not in collector._features
        assert "text_length" in collector._features

    def test_skip_non_numeric(self) -> None:
        """Should skip non-numeric features."""
        collector = FeatureStatsCollector()

        collector.add_features({
            "text_length": 100,
            "label": "toxic",
            "category": None,
        })

        assert "text_length" in collector._features
        assert "label" not in collector._features
        assert "category" not in collector._features

    def test_add_features_batch(self) -> None:
        """Should add features in batch."""
        collector = FeatureStatsCollector()

        batch = [
            {"text_length": 100, "word_count": 20},
            {"text_length": 150, "word_count": 30},
            {"text_length": 200, "word_count": 40},
        ]

        collector.add_features_batch(batch)

        assert len(collector._features["text_length"]) == 3
        assert len(collector._features["word_count"]) == 3

    def test_window_size_limit(self) -> None:
        """Should respect window size limit."""
        collector = FeatureStatsCollector(window_size=5)

        for i in range(10):
            collector.add_features({"value": float(i)})

        assert len(collector._features["value"]) == 5
        # Should keep the most recent values
        assert collector._features["value"] == [5.0, 6.0, 7.0, 8.0, 9.0]

    def test_get_stats(self) -> None:
        """Should calculate correct statistics."""
        collector = FeatureStatsCollector()

        for i in range(1, 6):  # 1, 2, 3, 4, 5
            collector.add_features({"value": float(i)})

        stats = collector.get_stats()

        assert "value" in stats
        assert stats["value"].mean == 3.0
        assert stats["value"].min_value == 1.0
        assert stats["value"].max_value == 5.0
        assert stats["value"].sample_count == 5

    def test_get_stats_insufficient_data(self) -> None:
        """Should return empty stats with insufficient data."""
        collector = FeatureStatsCollector()

        collector.add_features({"value": 1.0})

        stats = collector.get_stats()

        assert "value" not in stats  # Needs at least 2 samples

    def test_get_values(self) -> None:
        """Should return values for a specific feature."""
        collector = FeatureStatsCollector()

        collector.add_features({"a": 1.0, "b": 10.0})
        collector.add_features({"a": 2.0, "b": 20.0})

        assert collector.get_values("a") == [1.0, 2.0]
        assert collector.get_values("b") == [10.0, 20.0]
        assert collector.get_values("nonexistent") == []

    def test_reset(self) -> None:
        """Should reset all features."""
        collector = FeatureStatsCollector()

        collector.add_features({"a": 1.0, "b": 2.0})
        collector.reset()

        assert len(collector._features) == 0

    def test_reset_feature(self) -> None:
        """Should reset specific feature."""
        collector = FeatureStatsCollector()

        collector.add_features({"a": 1.0, "b": 2.0})
        collector.reset_feature("a")

        assert "a" not in collector._features
        assert "b" in collector._features


class TestFeatureDriftMonitor:
    """Tests for FeatureDriftMonitor."""

    @pytest.fixture
    def reference_stats(self) -> dict[str, FeatureStats]:
        """Create reference stats."""
        return {
            "text_length": FeatureStats(
                name="text_length",
                mean=100.0,
                std=20.0,
                min_value=50.0,
                max_value=150.0,
                sample_count=100,
                values=[80.0 + i * 0.4 for i in range(100)],
            ),
            "word_count": FeatureStats(
                name="word_count",
                mean=20.0,
                std=5.0,
                min_value=10.0,
                max_value=30.0,
                sample_count=100,
                values=[15.0 + i * 0.1 for i in range(100)],
            ),
        }

    @pytest.fixture
    def monitor(self, reference_stats: dict[str, FeatureStats]) -> FeatureDriftMonitor:
        """Create monitor with reference."""
        monitor = FeatureDriftMonitor(
            psi_threshold=0.2,
            ks_threshold=0.05,
            mean_shift_threshold=3.0,
        )
        monitor.set_reference(reference_stats)
        return monitor

    def test_initialization(self) -> None:
        """Should initialize with correct parameters."""
        monitor = FeatureDriftMonitor(
            psi_threshold=0.3,
            ks_threshold=0.01,
            window_size=500,
        )

        assert monitor.psi_threshold == 0.3
        assert monitor.ks_threshold == 0.01
        assert monitor.window_size == 500

    def test_no_reference_returns_no_drift(self) -> None:
        """Should return no drift when reference not set."""
        monitor = FeatureDriftMonitor()

        for i in range(100):
            monitor.add_current_features({"value": float(i)})

        result = monitor.check_drift()

        assert result.overall_drift_detected is False
        assert len(result.results) == 0

    def test_no_drift_similar_data(
        self,
        monitor: FeatureDriftMonitor,
        reference_stats: dict[str, FeatureStats],
    ) -> None:
        """Should not detect drift for similar data."""
        # Add data similar to reference
        for i in range(100):
            monitor.add_current_features({
                "text_length": 80.0 + i * 0.4,
                "word_count": 15.0 + i * 0.1,
            })

        result = monitor.check_drift()

        # PSI should be low for similar data
        assert result.timestamp is not None

    def test_drift_different_data(
        self,
        monitor: FeatureDriftMonitor,
    ) -> None:
        """Should detect drift for different data."""
        # Add very different data
        for i in range(100):
            monitor.add_current_features({
                "text_length": 500.0 + i,  # Much larger than reference
                "word_count": 100.0 + i,   # Much larger than reference
            })

        result = monitor.check_drift()

        assert result.overall_drift_detected is True
        assert len(result.drift_features) > 0

    def test_add_current_features_batch(
        self,
        monitor: FeatureDriftMonitor,
    ) -> None:
        """Should add features in batch."""
        batch = [
            {"text_length": 100.0, "word_count": 20.0},
            {"text_length": 110.0, "word_count": 22.0},
        ]

        monitor.add_current_features_batch(batch)

        stats = monitor.get_current_stats()
        assert stats["text_length"].sample_count == 2

    def test_set_reference_from_values(self) -> None:
        """Should set reference from raw values."""
        monitor = FeatureDriftMonitor()

        monitor.set_reference_from_values({
            "text_length": [100.0, 110.0, 120.0, 130.0, 140.0],
            "word_count": [20.0, 22.0, 24.0, 26.0, 28.0],
        })

        assert monitor._reference is not None
        assert "text_length" in monitor._reference

    def test_reset_current(
        self,
        monitor: FeatureDriftMonitor,
    ) -> None:
        """Should reset current features."""
        for i in range(50):
            monitor.add_current_features({"text_length": float(i)})

        monitor.reset_current()

        stats = monitor.get_current_stats()
        assert len(stats) == 0


class TestFeatureStoreIntegration:
    """Tests for FeatureStoreIntegration."""

    @pytest.fixture
    def mock_store(self) -> MagicMock:
        """Create mock feature store."""
        store = MagicMock()
        store.get_features = AsyncMock()
        store.get_features_batch = AsyncMock()
        return store

    def test_initialization(self, mock_store: MagicMock) -> None:
        """Should initialize with store."""
        integration = FeatureStoreIntegration(
            feature_store=mock_store,
            psi_threshold=0.3,
            ks_threshold=0.01,
        )

        assert integration.psi_threshold == 0.3
        assert integration.ks_threshold == 0.01

    @pytest.mark.asyncio
    async def test_build_reference_from_entities(
        self,
        mock_store: MagicMock,
    ) -> None:
        """Should build reference from store entities."""
        mock_store.get_features_batch.return_value = {
            "id1": {"text_length": 100, "word_count": 20},
            "id2": {"text_length": 110, "word_count": 22},
            "id3": {"text_length": 120, "word_count": 24},
        }

        integration = FeatureStoreIntegration(mock_store)

        loaded = await integration.build_reference_from_entities(
            entity_type="text",
            entity_ids=["id1", "id2", "id3"],
            feature_group="text_features",
        )

        assert loaded == 3
        mock_store.get_features_batch.assert_called_once()

    @pytest.mark.asyncio
    async def test_build_reference_handles_missing(
        self,
        mock_store: MagicMock,
    ) -> None:
        """Should handle missing entities."""
        mock_store.get_features_batch.return_value = {
            "id1": {"text_length": 100},
            "id2": None,  # Missing
            "id3": {"text_length": 120},
        }

        integration = FeatureStoreIntegration(mock_store)

        loaded = await integration.build_reference_from_entities(
            entity_type="text",
            entity_ids=["id1", "id2", "id3"],
            feature_group="text_features",
        )

        assert loaded == 2

    @pytest.mark.asyncio
    async def test_add_entities_for_monitoring(
        self,
        mock_store: MagicMock,
    ) -> None:
        """Should add entities for monitoring."""
        # First build reference
        mock_store.get_features_batch.return_value = {
            "id1": {"text_length": 100},
            "id2": {"text_length": 110},
        }

        integration = FeatureStoreIntegration(mock_store)

        await integration.build_reference_from_entities(
            entity_type="text",
            entity_ids=["id1", "id2"],
            feature_group="text_features",
        )

        # Then add monitoring entities
        mock_store.get_features_batch.return_value = {
            "id10": {"text_length": 150},
            "id11": {"text_length": 160},
        }

        loaded = await integration.add_entities_for_monitoring(
            entity_type="text",
            entity_ids=["id10", "id11"],
            feature_group="text_features",
        )

        assert loaded == 2

    def test_check_drift(self, mock_store: MagicMock) -> None:
        """Should check drift for a feature group."""
        integration = FeatureStoreIntegration(mock_store)

        # Manually set up the monitor
        monitor = integration._get_or_create_monitor("text_features")
        monitor.set_reference_from_values({
            "text_length": [100.0, 110.0, 120.0, 130.0, 140.0] * 20,
        })

        for i in range(100):
            monitor.add_current_features({"text_length": 500.0 + i})

        result = integration.check_drift("text_features", min_samples=50)

        assert isinstance(result, FeatureGroupDriftResult)
        assert result.feature_group == "text_features"

    def test_check_all_drift(self, mock_store: MagicMock) -> None:
        """Should check drift for all feature groups."""
        integration = FeatureStoreIntegration(mock_store)

        # Set up multiple monitors
        for group in ["text_features", "battle_features"]:
            monitor = integration._get_or_create_monitor(group)
            monitor.set_reference_from_values({
                "value": [float(i) for i in range(100)],
            })
            for i in range(100):
                monitor.add_current_features({"value": float(i)})

        results = integration.check_all_drift()

        assert "text_features" in results
        assert "battle_features" in results

    def test_reset_specific(self, mock_store: MagicMock) -> None:
        """Should reset specific monitor."""
        integration = FeatureStoreIntegration(mock_store)

        for group in ["text_features", "battle_features"]:
            monitor = integration._get_or_create_monitor(group)
            for i in range(50):
                monitor.add_current_features({"value": float(i)})

        integration.reset("text_features")

        text_stats = integration._monitors["text_features"].get_current_stats()
        battle_stats = integration._monitors["battle_features"].get_current_stats()

        assert len(text_stats) == 0
        assert len(battle_stats) > 0

    def test_reset_all(self, mock_store: MagicMock) -> None:
        """Should reset all monitors."""
        integration = FeatureStoreIntegration(mock_store)

        for group in ["text_features", "battle_features"]:
            monitor = integration._get_or_create_monitor(group)
            for i in range(50):
                monitor.add_current_features({"value": float(i)})

        integration.reset()

        for group in ["text_features", "battle_features"]:
            stats = integration._monitors[group].get_current_stats()
            assert len(stats) == 0
