"""Feature Drift Detection - Monitor feature distribution changes.

This module provides integration with the Feature Store for monitoring
drift in computed features over time.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from ml_service.monitoring.drift.data_drift import calculate_psi, ks_test

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FeatureStoreProtocol(Protocol):
    """Protocol for Feature Store operations needed by drift monitor."""

    async def get_features(
        self,
        entity_type: str,
        entity_id: str,
        feature_group: str,
        version: int = 1,
    ) -> dict[str, Any] | None: ...

    async def get_features_batch(
        self,
        entity_type: str,
        entity_ids: list[str],
        feature_group: str,
        version: int = 1,
    ) -> dict[str, dict[str, Any] | None]: ...


@dataclass
class FeatureStats:
    """Statistics for a single feature."""

    name: str
    mean: float
    std: float
    min_value: float
    max_value: float
    sample_count: int
    values: list[float]


@dataclass
class FeatureDriftResult:
    """Result of feature drift detection."""

    feature_name: str
    psi: float
    psi_alert: str  # "none", "minor", "significant"
    psi_drift: bool
    ks_statistic: float
    ks_p_value: float
    ks_drift: bool
    drift_detected: bool
    reference_mean: float
    reference_std: float
    current_mean: float
    current_std: float
    mean_shift_zscore: float
    timestamp: datetime


@dataclass
class FeatureGroupDriftResult:
    """Result of drift detection for a feature group."""

    feature_group: str
    results: dict[str, FeatureDriftResult]
    overall_drift_detected: bool
    drift_features: list[str]
    timestamp: datetime


class FeatureStatsCollector:
    """Collect feature statistics for drift analysis.

    Maintains a sliding window of feature values for each feature
    to enable statistical comparison.

    Example:
        >>> collector = FeatureStatsCollector(window_size=1000)
        >>> collector.add_features({"text_length": 100, "word_count": 20})
        >>> collector.add_features({"text_length": 150, "word_count": 30})
        >>> stats = collector.get_stats()
        >>> for name, stat in stats.items():
        ...     print(f"{name}: mean={stat.mean:.2f}, std={stat.std:.2f}")
    """

    def __init__(self, window_size: int = 1000) -> None:
        """Initialize the collector.

        Args:
            window_size: Maximum number of samples to keep per feature.
        """
        self.window_size = window_size
        self._features: dict[str, list[float]] = {}

    def add_features(self, features: dict[str, Any]) -> None:
        """Add feature values.

        Args:
            features: Dictionary of feature name to value.
        """
        for name, value in features.items():
            if name.startswith("_"):
                continue  # Skip metadata fields

            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                continue  # Skip non-numeric features

            if name not in self._features:
                self._features[name] = []

            self._features[name].append(numeric_value)

            # Maintain window size
            if len(self._features[name]) > self.window_size:
                self._features[name].pop(0)

    def add_features_batch(self, features_batch: list[dict[str, Any]]) -> None:
        """Add multiple feature sets.

        Args:
            features_batch: List of feature dictionaries.
        """
        for features in features_batch:
            self.add_features(features)

    def get_stats(self) -> dict[str, FeatureStats]:
        """Get statistics for all features.

        Returns:
            Dictionary of feature name to statistics.
        """
        stats = {}
        for name, values in self._features.items():
            if len(values) < 2:
                continue

            mean = sum(values) / len(values)
            variance = sum((x - mean) ** 2 for x in values) / len(values)
            std = variance**0.5

            stats[name] = FeatureStats(
                name=name,
                mean=mean,
                std=std,
                min_value=min(values),
                max_value=max(values),
                sample_count=len(values),
                values=values.copy(),
            )

        return stats

    def get_values(self, feature_name: str) -> list[float]:
        """Get collected values for a specific feature.

        Args:
            feature_name: Name of the feature.

        Returns:
            List of feature values.
        """
        return self._features.get(feature_name, [])

    def reset(self) -> None:
        """Reset all collected features."""
        self._features.clear()

    def reset_feature(self, feature_name: str) -> None:
        """Reset a specific feature.

        Args:
            feature_name: Name of the feature to reset.
        """
        if feature_name in self._features:
            del self._features[feature_name]


class FeatureDriftMonitor:
    """Monitor drift in features using statistical methods.

    Compares current feature distributions against reference distributions
    to detect significant shifts.

    Example:
        >>> monitor = FeatureDriftMonitor(psi_threshold=0.2)
        >>> monitor.set_reference(reference_collector.get_stats())
        >>> # Add current features
        >>> for features in incoming_features:
        ...     monitor.add_current_features(features)
        >>> # Check for drift
        >>> result = monitor.check_drift()
        >>> if result.overall_drift_detected:
        ...     print(f"Drift in: {result.drift_features}")
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        mean_shift_threshold: float = 3.0,
        window_size: int = 1000,
    ) -> None:
        """Initialize the monitor.

        Args:
            psi_threshold: PSI threshold for drift detection.
            ks_threshold: KS test p-value threshold.
            mean_shift_threshold: Mean shift Z-score threshold.
            window_size: Size of the current features window.
        """
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.mean_shift_threshold = mean_shift_threshold
        self.window_size = window_size

        self._reference: dict[str, FeatureStats] | None = None
        self._current_collector = FeatureStatsCollector(window_size=window_size)

    def set_reference(self, stats: dict[str, FeatureStats]) -> None:
        """Set reference feature statistics.

        Args:
            stats: Dictionary of feature name to statistics.
        """
        self._reference = stats

    def set_reference_from_values(
        self,
        feature_values: dict[str, list[float]],
    ) -> None:
        """Set reference from raw feature values.

        Args:
            feature_values: Dictionary of feature name to values.
        """
        collector = FeatureStatsCollector(window_size=len(next(iter(feature_values.values()), [])))
        for name, values in feature_values.items():
            for value in values:
                collector.add_features({name: value})
        self._reference = collector.get_stats()

    def add_current_features(self, features: dict[str, Any]) -> None:
        """Add current features for monitoring.

        Args:
            features: Dictionary of feature name to value.
        """
        self._current_collector.add_features(features)

    def add_current_features_batch(
        self,
        features_batch: list[dict[str, Any]],
    ) -> None:
        """Add multiple current feature sets.

        Args:
            features_batch: List of feature dictionaries.
        """
        self._current_collector.add_features_batch(features_batch)

    def check_drift(
        self,
        feature_group: str = "unknown",
        min_samples: int = 100,
    ) -> FeatureGroupDriftResult:
        """Check for drift in all monitored features.

        Args:
            feature_group: Name of the feature group.
            min_samples: Minimum samples required for drift check.

        Returns:
            FeatureGroupDriftResult with drift detection results.
        """
        if self._reference is None:
            return FeatureGroupDriftResult(
                feature_group=feature_group,
                results={},
                overall_drift_detected=False,
                drift_features=[],
                timestamp=datetime.now(UTC),
            )

        current_stats = self._current_collector.get_stats()
        results: dict[str, FeatureDriftResult] = {}
        drift_features: list[str] = []

        for feature_name, ref_stats in self._reference.items():
            if feature_name not in current_stats:
                continue

            cur_stats = current_stats[feature_name]

            if cur_stats.sample_count < min_samples:
                continue

            # Calculate PSI
            psi_result = calculate_psi(ref_stats.values, cur_stats.values)

            # Calculate KS test
            ks_result = ks_test(
                ref_stats.values,
                cur_stats.values,
                self.ks_threshold,
            )

            # Calculate mean shift Z-score
            if ref_stats.std > 0:
                mean_shift_zscore = abs(cur_stats.mean - ref_stats.mean) / ref_stats.std
            else:
                mean_shift_zscore = (
                    0.0 if abs(cur_stats.mean - ref_stats.mean) < 0.001 else float("inf")
                )

            # Determine drift
            psi_drift = psi_result.psi >= self.psi_threshold
            ks_drift = ks_result.drifted
            mean_drift = mean_shift_zscore > self.mean_shift_threshold
            drift_detected = psi_drift or ks_drift or mean_drift

            results[feature_name] = FeatureDriftResult(
                feature_name=feature_name,
                psi=psi_result.psi,
                psi_alert=psi_result.alert_level,
                psi_drift=psi_drift,
                ks_statistic=ks_result.statistic,
                ks_p_value=ks_result.p_value,
                ks_drift=ks_drift,
                drift_detected=drift_detected,
                reference_mean=ref_stats.mean,
                reference_std=ref_stats.std,
                current_mean=cur_stats.mean,
                current_std=cur_stats.std,
                mean_shift_zscore=mean_shift_zscore,
                timestamp=datetime.now(UTC),
            )

            if drift_detected:
                drift_features.append(feature_name)

        return FeatureGroupDriftResult(
            feature_group=feature_group,
            results=results,
            overall_drift_detected=len(drift_features) > 0,
            drift_features=drift_features,
            timestamp=datetime.now(UTC),
        )

    def get_current_stats(self) -> dict[str, FeatureStats]:
        """Get current feature statistics.

        Returns:
            Dictionary of feature name to statistics.
        """
        return self._current_collector.get_stats()

    def reset_current(self) -> None:
        """Reset current features window."""
        self._current_collector.reset()


class FeatureStoreIntegration:
    """Integration with Feature Store for drift monitoring.

    Fetches features from the online store and monitors them for drift.

    Example:
        >>> store = OnlineStore(config)
        >>> await store.connect()
        >>> integration = FeatureStoreIntegration(store)
        >>> # Set reference from historical data
        >>> await integration.build_reference_from_entities(
        ...     entity_type="text",
        ...     entity_ids=["id1", "id2", "id3"],
        ...     feature_group="text_features",
        ... )
        >>> # Monitor current entities
        >>> await integration.add_entities_for_monitoring(
        ...     entity_type="text",
        ...     entity_ids=["id10", "id11", "id12"],
        ...     feature_group="text_features",
        ... )
        >>> result = integration.check_drift("text_features")
    """

    def __init__(
        self,
        feature_store: FeatureStoreProtocol,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
    ) -> None:
        """Initialize the integration.

        Args:
            feature_store: Feature store instance.
            psi_threshold: PSI threshold for drift detection.
            ks_threshold: KS test p-value threshold.
        """
        self._store = feature_store
        self._monitors: dict[str, FeatureDriftMonitor] = {}
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold

    def _get_or_create_monitor(self, feature_group: str) -> FeatureDriftMonitor:
        """Get or create a monitor for a feature group.

        Args:
            feature_group: Name of the feature group.

        Returns:
            FeatureDriftMonitor instance.
        """
        if feature_group not in self._monitors:
            self._monitors[feature_group] = FeatureDriftMonitor(
                psi_threshold=self.psi_threshold,
                ks_threshold=self.ks_threshold,
            )
        return self._monitors[feature_group]

    async def build_reference_from_entities(
        self,
        entity_type: str,
        entity_ids: list[str],
        feature_group: str,
        version: int = 1,
    ) -> int:
        """Build reference statistics from feature store entities.

        Args:
            entity_type: Type of entity.
            entity_ids: List of entity IDs to use as reference.
            feature_group: Name of the feature group.
            version: Feature group version.

        Returns:
            Number of entities successfully loaded.
        """
        features_batch = await self._store.get_features_batch(
            entity_type=entity_type,
            entity_ids=entity_ids,
            feature_group=feature_group,
            version=version,
        )

        # Collect all feature values
        collector = FeatureStatsCollector(window_size=len(entity_ids))
        loaded = 0

        for features in features_batch.values():
            if features:
                collector.add_features(features)
                loaded += 1

        # Set reference
        monitor = self._get_or_create_monitor(feature_group)
        monitor.set_reference(collector.get_stats())

        return loaded

    async def add_entities_for_monitoring(
        self,
        entity_type: str,
        entity_ids: list[str],
        feature_group: str,
        version: int = 1,
    ) -> int:
        """Add entities for drift monitoring.

        Args:
            entity_type: Type of entity.
            entity_ids: List of entity IDs to monitor.
            feature_group: Name of the feature group.
            version: Feature group version.

        Returns:
            Number of entities successfully loaded.
        """
        features_batch = await self._store.get_features_batch(
            entity_type=entity_type,
            entity_ids=entity_ids,
            feature_group=feature_group,
            version=version,
        )

        monitor = self._get_or_create_monitor(feature_group)
        loaded = 0

        for features in features_batch.values():
            if features:
                monitor.add_current_features(features)
                loaded += 1

        return loaded

    def check_drift(
        self,
        feature_group: str,
        min_samples: int = 100,
    ) -> FeatureGroupDriftResult:
        """Check for drift in a feature group.

        Args:
            feature_group: Name of the feature group.
            min_samples: Minimum samples required.

        Returns:
            FeatureGroupDriftResult with drift detection results.
        """
        monitor = self._get_or_create_monitor(feature_group)
        return monitor.check_drift(feature_group, min_samples)

    def check_all_drift(
        self,
        min_samples: int = 100,
    ) -> dict[str, FeatureGroupDriftResult]:
        """Check drift in all monitored feature groups.

        Args:
            min_samples: Minimum samples required.

        Returns:
            Dictionary of feature group name to drift result.
        """
        return {
            group: monitor.check_drift(group, min_samples)
            for group, monitor in self._monitors.items()
        }

    def reset(self, feature_group: str | None = None) -> None:
        """Reset monitor(s).

        Args:
            feature_group: Specific group to reset. If None, resets all.
        """
        if feature_group:
            if feature_group in self._monitors:
                self._monitors[feature_group].reset_current()
        else:
            for monitor in self._monitors.values():
                monitor.reset_current()
