"""Confidence Monitoring - Track and analyze prediction confidence distributions."""

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ConfidenceStats:
    """Statistics about confidence distribution."""

    mean: float
    std: float
    min_value: float
    max_value: float
    sample_count: int
    timestamp: datetime


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""

    sufficient_data: bool
    current_mean: float | None = None
    current_std: float | None = None
    z_score: float | None = None
    anomaly_detected: bool = False
    severity: str | None = None


class ConfidenceAnomalyDetector:
    """Detect anomalies in prediction confidence distributions.

    Monitors the distribution of prediction confidence scores and
    detects when the distribution shifts significantly from the baseline.

    Example:
        >>> detector = ConfidenceAnomalyDetector(
        ...     reference_mean=0.85,
        ...     reference_std=0.1,
        ... )
        >>> for confidence in predictions:
        ...     detector.add_confidence(confidence)
        >>> result = detector.check_anomaly()
        >>> if result.anomaly_detected:
        ...     print("Confidence distribution anomaly detected!")
    """

    def __init__(
        self,
        reference_mean: float,
        reference_std: float,
        window_size: int = 1000,
        z_threshold: float = 2.0,
    ) -> None:
        """Initialize the anomaly detector.

        Args:
            reference_mean: Reference/baseline mean confidence.
            reference_std: Reference/baseline standard deviation.
            window_size: Size of the sliding window for current statistics.
            z_threshold: Z-score threshold for anomaly detection.
        """
        self.ref_mean = reference_mean
        self.ref_std = reference_std
        self.window_size = window_size
        self.z_threshold = z_threshold
        self._confidences: deque[float] = deque(maxlen=window_size)

    def add_confidence(self, confidence: float) -> None:
        """Add a confidence value to the monitoring window.

        Args:
            confidence: Confidence score from a prediction.
        """
        self._confidences.append(confidence)

    def add_batch(self, confidences: list[float]) -> None:
        """Add multiple confidence values at once.

        Args:
            confidences: List of confidence scores.
        """
        for conf in confidences:
            self._confidences.append(conf)

    def check_anomaly(self, min_samples: int = 100) -> AnomalyResult:
        """Check for anomalies in the confidence distribution.

        Args:
            min_samples: Minimum number of samples required for analysis.

        Returns:
            AnomalyResult with detection results.
        """
        if len(self._confidences) < min_samples:
            return AnomalyResult(sufficient_data=False)

        confidences = list(self._confidences)
        current_mean = sum(confidences) / len(confidences)
        variance = sum((x - current_mean) ** 2 for x in confidences) / len(confidences)
        current_std = math.sqrt(variance)

        # Calculate Z-score for the mean shift
        if self.ref_std > 0:
            z_score = abs(current_mean - self.ref_mean) / self.ref_std
        else:
            z_score = 0.0 if abs(current_mean - self.ref_mean) < 0.001 else float("inf")

        anomaly_detected = z_score > self.z_threshold
        severity = self._classify_severity(z_score) if anomaly_detected else None

        return AnomalyResult(
            sufficient_data=True,
            current_mean=current_mean,
            current_std=current_std,
            z_score=z_score,
            anomaly_detected=anomaly_detected,
            severity=severity,
        )

    def _classify_severity(self, z_score: float) -> str:
        """Classify anomaly severity based on Z-score.

        Args:
            z_score: The calculated Z-score.

        Returns:
            Severity level string.
        """
        if z_score > 4.0:
            return "critical"
        elif z_score > 3.0:
            return "high"
        elif z_score > 2.0:
            return "medium"
        return "low"

    def get_current_stats(self) -> ConfidenceStats | None:
        """Get current confidence statistics.

        Returns:
            ConfidenceStats or None if insufficient data.
        """
        if len(self._confidences) < 10:
            return None

        confidences = list(self._confidences)
        mean = sum(confidences) / len(confidences)
        variance = sum((x - mean) ** 2 for x in confidences) / len(confidences)

        return ConfidenceStats(
            mean=mean,
            std=math.sqrt(variance),
            min_value=min(confidences),
            max_value=max(confidences),
            sample_count=len(confidences),
            timestamp=datetime.now(UTC),
        )

    def reset(self) -> None:
        """Reset the monitoring window."""
        self._confidences.clear()

    def update_reference(self, mean: float, std: float) -> None:
        """Update the reference distribution.

        Args:
            mean: New reference mean.
            std: New reference standard deviation.
        """
        self.ref_mean = mean
        self.ref_std = std


class ConfidenceDistributionMonitor:
    """Monitor confidence distribution over time with multiple thresholds.

    Tracks confidence distribution across multiple buckets and
    monitors for changes in the distribution shape.

    Example:
        >>> monitor = ConfidenceDistributionMonitor()
        >>> for confidence in predictions:
        ...     monitor.add_confidence(confidence)
        >>> distribution = monitor.get_distribution()
        >>> if distribution["low_rate"] > 0.3:
        ...     print("High rate of low confidence predictions!")
    """

    def __init__(
        self,
        low_threshold: float = 0.7,
        medium_threshold: float = 0.85,
        high_threshold: float = 0.95,
        window_size: int = 1000,
    ) -> None:
        """Initialize the distribution monitor.

        Args:
            low_threshold: Threshold for low confidence.
            medium_threshold: Threshold for medium confidence.
            high_threshold: Threshold for high confidence.
            window_size: Size of the sliding window.
        """
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold
        self.high_threshold = high_threshold
        self.window_size = window_size

        self._confidences: deque[float] = deque(maxlen=window_size)
        self._total_count = 0

    def add_confidence(self, confidence: float) -> None:
        """Add a confidence value.

        Args:
            confidence: Confidence score.
        """
        self._confidences.append(confidence)
        self._total_count += 1

    def add_batch(self, confidences: list[float]) -> None:
        """Add multiple confidence values.

        Args:
            confidences: List of confidence scores.
        """
        for conf in confidences:
            self._confidences.append(conf)
            self._total_count += 1

    def get_distribution(self) -> dict[str, Any]:
        """Get the current confidence distribution.

        Returns:
            Dictionary with distribution statistics.
        """
        if not self._confidences:
            return {
                "total_count": 0,
                "window_count": 0,
                "very_low_count": 0,
                "low_count": 0,
                "medium_count": 0,
                "high_count": 0,
                "very_high_count": 0,
                "very_low_rate": 0.0,
                "low_rate": 0.0,
                "medium_rate": 0.0,
                "high_rate": 0.0,
                "very_high_rate": 0.0,
            }

        confidences = list(self._confidences)
        window_count = len(confidences)

        very_low = sum(1 for c in confidences if c < 0.5)
        low = sum(1 for c in confidences if 0.5 <= c < self.low_threshold)
        medium = sum(
            1 for c in confidences if self.low_threshold <= c < self.medium_threshold
        )
        high = sum(
            1 for c in confidences if self.medium_threshold <= c < self.high_threshold
        )
        very_high = sum(1 for c in confidences if c >= self.high_threshold)

        return {
            "total_count": self._total_count,
            "window_count": window_count,
            "very_low_count": very_low,
            "low_count": low,
            "medium_count": medium,
            "high_count": high,
            "very_high_count": very_high,
            "very_low_rate": very_low / window_count,
            "low_rate": low / window_count,
            "medium_rate": medium / window_count,
            "high_rate": high / window_count,
            "very_high_rate": very_high / window_count,
        }

    def get_percentiles(self) -> dict[str, float] | None:
        """Get confidence percentiles.

        Returns:
            Dictionary with percentile values or None if insufficient data.
        """
        if len(self._confidences) < 10:
            return None

        sorted_conf = sorted(self._confidences)
        n = len(sorted_conf)

        def percentile(p: float) -> float:
            k = (n - 1) * p
            f = int(k)
            c = math.ceil(k)
            if f == c:
                return sorted_conf[f]
            return sorted_conf[f] * (c - k) + sorted_conf[c] * (k - f)

        return {
            "p5": percentile(0.05),
            "p25": percentile(0.25),
            "p50": percentile(0.50),
            "p75": percentile(0.75),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
        }

    def check_threshold_alert(
        self,
        low_rate_threshold: float = 0.3,
        very_low_rate_threshold: float = 0.1,
    ) -> dict[str, Any]:
        """Check if confidence rates exceed alert thresholds.

        Args:
            low_rate_threshold: Alert threshold for low confidence rate.
            very_low_rate_threshold: Alert threshold for very low confidence rate.

        Returns:
            Dictionary with alert information.
        """
        distribution = self.get_distribution()

        alerts = []
        if distribution["low_rate"] > low_rate_threshold:
            alerts.append(
                {
                    "type": "low_confidence_rate",
                    "value": distribution["low_rate"],
                    "threshold": low_rate_threshold,
                }
            )
        if distribution["very_low_rate"] > very_low_rate_threshold:
            alerts.append(
                {
                    "type": "very_low_confidence_rate",
                    "value": distribution["very_low_rate"],
                    "threshold": very_low_rate_threshold,
                }
            )

        return {
            "has_alerts": len(alerts) > 0,
            "alerts": alerts,
            "distribution": distribution,
        }

    def reset(self) -> None:
        """Reset the monitor."""
        self._confidences.clear()
        self._total_count = 0
