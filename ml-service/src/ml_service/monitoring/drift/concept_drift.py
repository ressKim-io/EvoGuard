"""Concept Drift Detection - Performance-based drift detection.

This module provides implementations for detecting concept drift using:
- Performance-based monitoring (F1, Precision, Recall)
- ADWIN (Adaptive Windowing) algorithm for real-time detection
"""

import logging
import math
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""

    f1: float
    precision: float
    recall: float
    accuracy: float
    sample_count: int
    timestamp: datetime


@dataclass
class ConceptDriftResult:
    """Result of concept drift detection."""

    sufficient_data: bool
    drift_detected: bool = False
    severity: str = "none"  # "none", "low", "medium", "high", "critical"
    current_metrics: PerformanceMetrics | None = None
    baseline_metrics: PerformanceMetrics | None = None
    f1_drop: float = 0.0
    precision_drop: float = 0.0
    recall_drop: float = 0.0
    timestamp: datetime | None = None


@dataclass
class ADWINResult:
    """Result of ADWIN drift detection."""

    drift_detected: bool
    window_size: int
    mean: float
    variance: float
    timestamp: datetime


class ConceptDriftMonitor:
    """Monitor concept drift based on model performance degradation.

    Tracks model performance metrics (F1, Precision, Recall) over time
    and detects when performance drops below baseline thresholds.

    Example:
        >>> baseline = PerformanceMetrics(
        ...     f1=0.85, precision=0.87, recall=0.83,
        ...     accuracy=0.86, sample_count=1000, timestamp=datetime.now(UTC)
        ... )
        >>> monitor = ConceptDriftMonitor(baseline_metrics=baseline)
        >>> # Add predictions with actual labels
        >>> for pred, actual in predictions_with_labels:
        ...     monitor.add_feedback(pred, actual)
        >>> result = monitor.check_drift()
        >>> if result.drift_detected:
        ...     print(f"Concept drift detected! F1 dropped by {result.f1_drop:.2%}")
    """

    def __init__(
        self,
        baseline_metrics: PerformanceMetrics | None = None,
        window_size: int = 1000,
        f1_threshold: float = 0.05,
        precision_threshold: float = 0.05,
        recall_threshold: float = 0.05,
    ) -> None:
        """Initialize the concept drift monitor.

        Args:
            baseline_metrics: Baseline performance metrics to compare against.
            window_size: Size of the sliding window for predictions.
            f1_threshold: F1 drop threshold for drift detection.
            precision_threshold: Precision drop threshold.
            recall_threshold: Recall drop threshold.
        """
        self.baseline = baseline_metrics
        self.window_size = window_size
        self.f1_threshold = f1_threshold
        self.precision_threshold = precision_threshold
        self.recall_threshold = recall_threshold

        self._predictions: deque[int] = deque(maxlen=window_size)
        self._labels: deque[int] = deque(maxlen=window_size)

    def set_baseline(self, metrics: PerformanceMetrics) -> None:
        """Set the baseline metrics.

        Args:
            metrics: Baseline performance metrics.
        """
        self.baseline = metrics

    def set_baseline_from_data(
        self,
        predictions: list[int],
        labels: list[int],
    ) -> None:
        """Set baseline from prediction data.

        Args:
            predictions: List of predictions.
            labels: List of actual labels.
        """
        metrics = self._calculate_metrics(predictions, labels)
        self.baseline = metrics

    def add_feedback(self, prediction: int, actual: int) -> None:
        """Add a prediction with its actual label.

        Args:
            prediction: Model prediction (0 or 1).
            actual: Actual label (0 or 1).
        """
        self._predictions.append(prediction)
        self._labels.append(actual)

    def add_feedback_batch(
        self,
        predictions: list[int],
        labels: list[int],
    ) -> None:
        """Add multiple predictions with their labels.

        Args:
            predictions: List of predictions.
            labels: List of actual labels.
        """
        for pred, label in zip(predictions, labels, strict=False):
            self._predictions.append(pred)
            self._labels.append(label)

    def check_drift(self, min_samples: int | None = None) -> ConceptDriftResult:
        """Check for concept drift.

        Args:
            min_samples: Minimum number of samples required. Defaults to window_size // 2.

        Returns:
            ConceptDriftResult with detection results.
        """
        min_samples = min_samples or self.window_size // 2

        if len(self._predictions) < min_samples:
            return ConceptDriftResult(
                sufficient_data=False,
                timestamp=datetime.now(UTC),
            )

        if self.baseline is None:
            return ConceptDriftResult(
                sufficient_data=False,
                timestamp=datetime.now(UTC),
            )

        current = self._calculate_metrics(
            list(self._predictions),
            list(self._labels),
        )

        f1_drop = self.baseline.f1 - current.f1
        precision_drop = self.baseline.precision - current.precision
        recall_drop = self.baseline.recall - current.recall

        # Check for drift
        drift_detected = (
            f1_drop > self.f1_threshold
            or precision_drop > self.precision_threshold
            or recall_drop > self.recall_threshold
        )

        severity = self._classify_severity(f1_drop)

        return ConceptDriftResult(
            sufficient_data=True,
            drift_detected=drift_detected,
            severity=severity,
            current_metrics=current,
            baseline_metrics=self.baseline,
            f1_drop=f1_drop,
            precision_drop=precision_drop,
            recall_drop=recall_drop,
            timestamp=datetime.now(UTC),
        )

    def _calculate_metrics(
        self,
        predictions: list[int],
        labels: list[int],
    ) -> PerformanceMetrics:
        """Calculate performance metrics.

        Args:
            predictions: List of predictions.
            labels: List of actual labels.

        Returns:
            PerformanceMetrics instance.
        """
        if not predictions or not labels:
            return PerformanceMetrics(
                f1=0.0,
                precision=0.0,
                recall=0.0,
                accuracy=0.0,
                sample_count=0,
                timestamp=datetime.now(UTC),
            )

        # Calculate confusion matrix components
        tp = sum(1 for pred, lbl in zip(predictions, labels, strict=False) if pred == 1 and lbl == 1)
        fp = sum(1 for pred, lbl in zip(predictions, labels, strict=False) if pred == 1 and lbl == 0)
        fn = sum(1 for pred, lbl in zip(predictions, labels, strict=False) if pred == 0 and lbl == 1)
        tn = sum(1 for pred, lbl in zip(predictions, labels, strict=False) if pred == 0 and lbl == 0)

        total = len(predictions)

        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / total if total > 0 else 0.0

        return PerformanceMetrics(
            f1=f1,
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            sample_count=total,
            timestamp=datetime.now(UTC),
        )

    def _classify_severity(self, f1_drop: float) -> str:
        """Classify drift severity based on F1 drop.

        Args:
            f1_drop: The F1 score drop.

        Returns:
            Severity level string.
        """
        if f1_drop > 0.15:
            return "critical"
        elif f1_drop > 0.10:
            return "high"
        elif f1_drop > 0.05:
            return "medium"
        elif f1_drop > 0.0:
            return "low"
        return "none"

    def get_current_metrics(self) -> PerformanceMetrics | None:
        """Get current performance metrics.

        Returns:
            PerformanceMetrics or None if insufficient data.
        """
        if len(self._predictions) < 10:
            return None

        return self._calculate_metrics(
            list(self._predictions),
            list(self._labels),
        )

    def reset(self) -> None:
        """Reset the monitoring window."""
        self._predictions.clear()
        self._labels.clear()


class ADWINDriftDetector:
    """ADWIN (Adaptive Windowing) algorithm for real-time drift detection.

    ADWIN maintains a window of recent items and automatically detects
    when the mean of the window has changed significantly.

    Reference: Albert Bifet and Ricard Gavaldà.
    "Learning from Time-Changing Data with Adaptive Windowing"

    Example:
        >>> detector = ADWINDriftDetector(delta=0.002)
        >>> for value in streaming_values:
        ...     drift = detector.update(value)
        ...     if drift:
        ...         print(f"Drift detected at window size {detector.width}")
    """

    def __init__(self, delta: float = 0.002) -> None:
        """Initialize the ADWIN detector.

        Args:
            delta: Confidence parameter. Smaller values mean more confidence
                   required to detect drift (fewer false positives).
        """
        self.delta = delta
        self._window: deque[float] = deque()
        self._total = 0.0
        self._variance = 0.0
        self._width = 0

    @property
    def width(self) -> int:
        """Get current window width."""
        return self._width

    @property
    def mean(self) -> float:
        """Get current window mean."""
        return self._total / self._width if self._width > 0 else 0.0

    @property
    def variance(self) -> float:
        """Get current window variance."""
        if self._width < 2:
            return 0.0
        mean = self.mean
        return sum((x - mean) ** 2 for x in self._window) / self._width

    def update(self, value: float) -> bool:
        """Update with a new value and check for drift.

        Args:
            value: New value to add to the window.

        Returns:
            True if drift was detected, False otherwise.
        """
        self._window.append(value)
        self._width += 1
        self._total += value

        if self._width > 1:
            return self._check_drift()
        return False

    def update_batch(self, values: list[float]) -> list[bool]:
        """Update with multiple values.

        Args:
            values: List of values to add.

        Returns:
            List of drift detection results for each value.
        """
        return [self.update(value) for value in values]

    def _check_drift(self) -> bool:
        """Check for drift by comparing window splits.

        Returns:
            True if drift detected, False otherwise.
        """
        if self._width < 10:
            return False

        window_list = list(self._window)

        # Check different split points
        for split in range(5, self._width - 5):
            w0 = window_list[:split]
            w1 = window_list[split:]

            n0 = len(w0)
            n1 = len(w1)

            mean0 = sum(w0) / n0
            mean1 = sum(w1) / n1

            # Calculate the cut threshold using Hoeffding bound
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            epsilon = math.sqrt((1.0 / (2 * m)) * math.log(4.0 / self.delta))

            if abs(mean0 - mean1) > epsilon:
                # Drift detected, shrink the window
                self._window = deque(w1)
                self._width = n1
                self._total = sum(w1)
                return True

        return False

    def check_drift_result(self) -> ADWINResult:
        """Get detailed drift check result.

        Returns:
            ADWINResult with current state.
        """
        return ADWINResult(
            drift_detected=False,  # Call update() to check for drift
            window_size=self._width,
            mean=self.mean,
            variance=self.variance,
            timestamp=datetime.now(UTC),
        )

    def reset(self) -> None:
        """Reset the detector."""
        self._window.clear()
        self._total = 0.0
        self._variance = 0.0
        self._width = 0


class MultiMetricDriftDetector:
    """Monitor multiple metrics for concept drift simultaneously.

    Uses separate ADWIN detectors for different performance metrics.

    Example:
        >>> detector = MultiMetricDriftDetector(
        ...     metrics=["accuracy", "f1", "latency"],
        ...     delta=0.002,
        ... )
        >>> for metrics in streaming_metrics:
        ...     results = detector.update(metrics)
        ...     if any(results.values()):
        ...         print(f"Drift detected in: {[k for k, v in results.items() if v]}")
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        delta: float = 0.002,
    ) -> None:
        """Initialize the multi-metric detector.

        Args:
            metrics: List of metric names to monitor.
            delta: ADWIN delta parameter for all detectors.
        """
        self.metric_names = metrics or ["accuracy", "f1", "precision", "recall"]
        self.delta = delta
        self._detectors: dict[str, ADWINDriftDetector] = {
            name: ADWINDriftDetector(delta=delta) for name in self.metric_names
        }

    def update(self, values: dict[str, float]) -> dict[str, bool]:
        """Update all detectors with new metric values.

        Args:
            values: Dictionary of metric name to value.

        Returns:
            Dictionary of metric name to drift detection result.
        """
        results = {}
        for name, detector in self._detectors.items():
            if name in values:
                results[name] = detector.update(values[name])
            else:
                results[name] = False
        return results

    def get_states(self) -> dict[str, dict[str, Any]]:
        """Get the state of all detectors.

        Returns:
            Dictionary of metric name to detector state.
        """
        return {
            name: {
                "width": detector.width,
                "mean": detector.mean,
                "variance": detector.variance,
            }
            for name, detector in self._detectors.items()
        }

    def reset(self, metric: str | None = None) -> None:
        """Reset detector(s).

        Args:
            metric: Specific metric to reset. If None, resets all.
        """
        if metric:
            if metric in self._detectors:
                self._detectors[metric].reset()
        else:
            for detector in self._detectors.values():
                detector.reset()
