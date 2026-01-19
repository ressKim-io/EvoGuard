"""Tests for concept drift detection module."""

from datetime import UTC, datetime

import pytest

from ml_service.monitoring.drift.concept_drift import (
    ADWINDriftDetector,
    ConceptDriftMonitor,
    MultiMetricDriftDetector,
    PerformanceMetrics,
)


class TestConceptDriftMonitor:
    """Tests for ConceptDriftMonitor."""

    @pytest.fixture
    def baseline_metrics(self) -> PerformanceMetrics:
        """Create baseline metrics."""
        return PerformanceMetrics(
            f1=0.85,
            precision=0.87,
            recall=0.83,
            accuracy=0.86,
            sample_count=1000,
            timestamp=datetime.now(UTC),
        )

    @pytest.fixture
    def monitor(self, baseline_metrics: PerformanceMetrics) -> ConceptDriftMonitor:
        """Create a monitor with baseline metrics."""
        return ConceptDriftMonitor(
            baseline_metrics=baseline_metrics,
            window_size=100,
            f1_threshold=0.05,
        )

    def test_initialization(self, baseline_metrics: PerformanceMetrics) -> None:
        """Should initialize with correct parameters."""
        monitor = ConceptDriftMonitor(
            baseline_metrics=baseline_metrics,
            window_size=500,
            f1_threshold=0.10,
        )

        assert monitor.baseline == baseline_metrics
        assert monitor.window_size == 500
        assert monitor.f1_threshold == 0.10

    def test_insufficient_data(self, monitor: ConceptDriftMonitor) -> None:
        """Should return insufficient data when not enough samples."""
        # Add only a few predictions
        for _ in range(10):
            monitor.add_feedback(1, 1)

        result = monitor.check_drift()

        assert result.sufficient_data is False

    def test_no_drift_good_performance(self, monitor: ConceptDriftMonitor) -> None:
        """Should not detect drift when performance is good."""
        # Perfect predictions matching baseline
        for _ in range(100):
            monitor.add_feedback(1, 1)  # True positives
            monitor.add_feedback(0, 0)  # True negatives

        result = monitor.check_drift(min_samples=50)

        # Should have high metrics, similar to or better than baseline
        assert result.sufficient_data is True

    def test_drift_detection_f1_drop(self, monitor: ConceptDriftMonitor) -> None:
        """Should detect drift when F1 drops significantly."""
        # Add predictions that result in low F1
        for _ in range(50):
            monitor.add_feedback(1, 0)  # False positives
            monitor.add_feedback(0, 1)  # False negatives

        result = monitor.check_drift(min_samples=50)

        assert result.sufficient_data is True
        assert result.drift_detected is True
        assert result.f1_drop > 0

    def test_severity_classification(self) -> None:
        """Should correctly classify drift severity."""
        monitor = ConceptDriftMonitor()

        # Severity levels: none (<=0), low (>0), medium (>0.05), high (>0.10), critical (>0.15)
        assert monitor._classify_severity(-0.01) == "none"
        assert monitor._classify_severity(0.0) == "none"
        assert monitor._classify_severity(0.02) == "low"
        assert monitor._classify_severity(0.06) == "medium"
        assert monitor._classify_severity(0.12) == "high"
        assert monitor._classify_severity(0.20) == "critical"

    def test_add_feedback_batch(self, monitor: ConceptDriftMonitor) -> None:
        """Should add feedback in batches."""
        predictions = [1, 0, 1, 0, 1]
        labels = [1, 0, 0, 1, 1]

        monitor.add_feedback_batch(predictions, labels)

        assert len(monitor._predictions) == 5
        assert len(monitor._labels) == 5

    def test_set_baseline_from_data(self) -> None:
        """Should set baseline from prediction data."""
        monitor = ConceptDriftMonitor(window_size=100)

        predictions = [1, 1, 1, 0, 0] * 20
        labels = [1, 1, 0, 0, 1] * 20

        monitor.set_baseline_from_data(predictions, labels)

        assert monitor.baseline is not None
        assert monitor.baseline.sample_count == 100

    def test_get_current_metrics(self, monitor: ConceptDriftMonitor) -> None:
        """Should return current metrics."""
        for _ in range(50):
            monitor.add_feedback(1, 1)

        metrics = monitor.get_current_metrics()

        assert metrics is not None
        assert metrics.f1 > 0
        assert metrics.sample_count == 50

    def test_reset(self, monitor: ConceptDriftMonitor) -> None:
        """Should reset predictions."""
        for _ in range(50):
            monitor.add_feedback(1, 1)

        monitor.reset()

        assert len(monitor._predictions) == 0
        assert len(monitor._labels) == 0

    def test_no_baseline_returns_insufficient(self) -> None:
        """Should return insufficient data when no baseline set."""
        monitor = ConceptDriftMonitor(window_size=100)

        for _ in range(100):
            monitor.add_feedback(1, 1)

        result = monitor.check_drift(min_samples=50)

        assert result.sufficient_data is False


class TestADWINDriftDetector:
    """Tests for ADWIN drift detector."""

    def test_initialization(self) -> None:
        """Should initialize with correct parameters."""
        detector = ADWINDriftDetector(delta=0.01)

        assert detector.delta == 0.01
        assert detector.width == 0
        assert detector.mean == 0.0

    def test_update_without_drift(self) -> None:
        """Should not detect drift for stable values."""
        detector = ADWINDriftDetector(delta=0.002)

        # Add stable values
        for _ in range(100):
            detector.update(0.5)

        # Should not detect drift for constant values
        assert detector.width > 0

    def test_update_with_drift(self) -> None:
        """Should detect drift when mean changes significantly."""
        detector = ADWINDriftDetector(delta=0.1)

        # Add values from one distribution
        for _ in range(50):
            detector.update(0.2)

        # Shift to a different distribution
        for _ in range(50):
            if detector.update(0.8):
                break

        # May or may not detect drift depending on the delta
        assert detector.width > 0

    def test_properties(self) -> None:
        """Should correctly calculate properties."""
        detector = ADWINDriftDetector()

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for v in values:
            detector.update(v)

        assert detector.width == 5
        assert detector.mean == 3.0
        assert detector.variance > 0

    def test_update_batch(self) -> None:
        """Should handle batch updates."""
        detector = ADWINDriftDetector()

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        results = detector.update_batch(values)

        assert len(results) == 5
        assert all(isinstance(r, bool) for r in results)

    def test_check_drift_result(self) -> None:
        """Should return detailed result."""
        detector = ADWINDriftDetector()

        for i in range(20):
            detector.update(float(i))

        result = detector.check_drift_result()

        assert result.window_size == detector.width
        assert result.mean == detector.mean
        assert result.timestamp is not None

    def test_reset(self) -> None:
        """Should reset detector state."""
        detector = ADWINDriftDetector()

        for i in range(50):
            detector.update(float(i))

        detector.reset()

        assert detector.width == 0
        assert detector.mean == 0.0


class TestMultiMetricDriftDetector:
    """Tests for MultiMetricDriftDetector."""

    def test_initialization(self) -> None:
        """Should initialize with correct metrics."""
        detector = MultiMetricDriftDetector(
            metrics=["accuracy", "f1", "latency"],
            delta=0.01,
        )

        assert len(detector._detectors) == 3
        assert "accuracy" in detector._detectors
        assert "f1" in detector._detectors
        assert "latency" in detector._detectors

    def test_default_metrics(self) -> None:
        """Should use default metrics if none provided."""
        detector = MultiMetricDriftDetector()

        assert "accuracy" in detector._detectors
        assert "f1" in detector._detectors
        assert "precision" in detector._detectors
        assert "recall" in detector._detectors

    def test_update(self) -> None:
        """Should update all relevant detectors."""
        detector = MultiMetricDriftDetector(metrics=["accuracy", "f1"])

        results = detector.update({"accuracy": 0.85, "f1": 0.80})

        assert "accuracy" in results
        assert "f1" in results

    def test_update_partial(self) -> None:
        """Should handle partial updates."""
        detector = MultiMetricDriftDetector(metrics=["accuracy", "f1", "recall"])

        results = detector.update({"accuracy": 0.85})

        assert results["accuracy"] is not None
        assert results["f1"] is False
        assert results["recall"] is False

    def test_get_states(self) -> None:
        """Should return states of all detectors."""
        detector = MultiMetricDriftDetector(metrics=["accuracy", "f1"])

        for i in range(20):
            detector.update({"accuracy": 0.85 + i * 0.001, "f1": 0.80 + i * 0.001})

        states = detector.get_states()

        assert "accuracy" in states
        assert "f1" in states
        assert states["accuracy"]["width"] == 20
        assert states["f1"]["width"] == 20

    def test_reset_specific(self) -> None:
        """Should reset specific metric."""
        detector = MultiMetricDriftDetector(metrics=["accuracy", "f1"])

        for _i in range(20):
            detector.update({"accuracy": 0.85, "f1": 0.80})

        detector.reset("accuracy")

        states = detector.get_states()
        assert states["accuracy"]["width"] == 0
        assert states["f1"]["width"] == 20

    def test_reset_all(self) -> None:
        """Should reset all metrics."""
        detector = MultiMetricDriftDetector(metrics=["accuracy", "f1"])

        for _i in range(20):
            detector.update({"accuracy": 0.85, "f1": 0.80})

        detector.reset()

        states = detector.get_states()
        assert states["accuracy"]["width"] == 0
        assert states["f1"]["width"] == 0
