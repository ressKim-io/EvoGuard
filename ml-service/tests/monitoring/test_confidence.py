"""Tests for confidence monitoring."""


from ml_service.monitoring.prediction.confidence import (
    AnomalyResult,
    ConfidenceAnomalyDetector,
    ConfidenceDistributionMonitor,
    ConfidenceStats,
)


class TestConfidenceAnomalyDetector:
    """Test ConfidenceAnomalyDetector class."""

    def test_init(self) -> None:
        """Test initialization."""
        detector = ConfidenceAnomalyDetector(
            reference_mean=0.85,
            reference_std=0.1,
        )
        assert detector.ref_mean == 0.85
        assert detector.ref_std == 0.1
        assert detector.z_threshold == 2.0

    def test_add_confidence(self) -> None:
        """Test adding confidence values."""
        detector = ConfidenceAnomalyDetector(0.85, 0.1)
        detector.add_confidence(0.9)
        detector.add_confidence(0.8)
        # Should not raise

    def test_add_batch(self) -> None:
        """Test adding a batch of confidence values."""
        detector = ConfidenceAnomalyDetector(0.85, 0.1)
        detector.add_batch([0.9, 0.85, 0.8, 0.75, 0.95])
        # Should not raise

    def test_check_anomaly_insufficient_data(self) -> None:
        """Test anomaly check with insufficient data."""
        detector = ConfidenceAnomalyDetector(0.85, 0.1)
        detector.add_batch([0.9] * 50)  # Less than min_samples=100

        result = detector.check_anomaly()
        assert result.sufficient_data is False

    def test_check_anomaly_no_anomaly(self) -> None:
        """Test anomaly check when distribution is normal."""
        detector = ConfidenceAnomalyDetector(0.85, 0.1)
        # Add values close to reference mean
        detector.add_batch([0.84, 0.86, 0.83, 0.87, 0.85] * 25)

        result = detector.check_anomaly()
        assert result.sufficient_data is True
        assert result.anomaly_detected is False
        assert result.current_mean is not None
        assert 0.8 < result.current_mean < 0.9

    def test_check_anomaly_detects_shift(self) -> None:
        """Test anomaly detection when mean shifts significantly."""
        detector = ConfidenceAnomalyDetector(
            reference_mean=0.85,
            reference_std=0.05,
            z_threshold=2.0,
        )
        # Add values with much lower mean
        detector.add_batch([0.5, 0.55, 0.6, 0.45, 0.5] * 25)

        result = detector.check_anomaly()
        assert result.sufficient_data is True
        assert result.anomaly_detected is True
        assert result.z_score is not None
        assert result.z_score > 2.0
        assert result.severity is not None

    def test_severity_classification(self) -> None:
        """Test severity classification based on z-score."""
        detector = ConfidenceAnomalyDetector(0.85, 0.01)

        # Test different severities
        assert detector._classify_severity(4.5) == "critical"
        assert detector._classify_severity(3.5) == "high"
        assert detector._classify_severity(2.5) == "medium"
        assert detector._classify_severity(1.5) == "low"

    def test_get_current_stats_insufficient_data(self) -> None:
        """Test getting stats with insufficient data."""
        detector = ConfidenceAnomalyDetector(0.85, 0.1)
        detector.add_batch([0.9, 0.85])  # Only 2 samples

        stats = detector.get_current_stats()
        assert stats is None

    def test_get_current_stats(self) -> None:
        """Test getting current statistics."""
        detector = ConfidenceAnomalyDetector(0.85, 0.1)
        detector.add_batch([0.8, 0.85, 0.9, 0.75, 0.95] * 5)

        stats = detector.get_current_stats()
        assert stats is not None
        assert isinstance(stats, ConfidenceStats)
        assert 0.75 <= stats.mean <= 0.95
        assert stats.min_value == 0.75
        assert stats.max_value == 0.95
        assert stats.sample_count == 25

    def test_reset(self) -> None:
        """Test resetting the detector."""
        detector = ConfidenceAnomalyDetector(0.85, 0.1)
        detector.add_batch([0.9] * 50)

        detector.reset()
        result = detector.check_anomaly(min_samples=10)
        assert result.sufficient_data is False

    def test_update_reference(self) -> None:
        """Test updating reference distribution."""
        detector = ConfidenceAnomalyDetector(0.85, 0.1)
        detector.update_reference(0.9, 0.05)

        assert detector.ref_mean == 0.9
        assert detector.ref_std == 0.05

    def test_zero_std_handling(self) -> None:
        """Test handling when reference std is zero."""
        detector = ConfidenceAnomalyDetector(0.85, 0.0)
        detector.add_batch([0.85] * 150)

        # Should not raise
        result = detector.check_anomaly()
        assert result.sufficient_data is True


class TestConfidenceDistributionMonitor:
    """Test ConfidenceDistributionMonitor class."""

    def test_init(self) -> None:
        """Test initialization."""
        monitor = ConfidenceDistributionMonitor()
        assert monitor.low_threshold == 0.7
        assert monitor.medium_threshold == 0.85
        assert monitor.high_threshold == 0.95

    def test_init_custom_thresholds(self) -> None:
        """Test initialization with custom thresholds."""
        monitor = ConfidenceDistributionMonitor(
            low_threshold=0.6,
            medium_threshold=0.8,
            high_threshold=0.9,
        )
        assert monitor.low_threshold == 0.6
        assert monitor.medium_threshold == 0.8
        assert monitor.high_threshold == 0.9

    def test_add_confidence(self) -> None:
        """Test adding confidence values."""
        monitor = ConfidenceDistributionMonitor()
        monitor.add_confidence(0.9)
        monitor.add_confidence(0.8)

    def test_add_batch(self) -> None:
        """Test adding a batch of confidence values."""
        monitor = ConfidenceDistributionMonitor()
        monitor.add_batch([0.9, 0.85, 0.8, 0.75, 0.95])

    def test_get_distribution_empty(self) -> None:
        """Test getting distribution with no data."""
        monitor = ConfidenceDistributionMonitor()
        dist = monitor.get_distribution()

        assert dist["total_count"] == 0
        assert dist["window_count"] == 0

    def test_get_distribution(self) -> None:
        """Test getting confidence distribution."""
        monitor = ConfidenceDistributionMonitor()

        # Add values in different ranges
        monitor.add_batch([0.4, 0.45])  # Very low (<0.5)
        monitor.add_batch([0.55, 0.65])  # Low (0.5-0.7)
        monitor.add_batch([0.75, 0.8])  # Medium (0.7-0.85)
        monitor.add_batch([0.88, 0.92])  # High (0.85-0.95)
        monitor.add_batch([0.97, 0.99])  # Very high (>=0.95)

        dist = monitor.get_distribution()

        assert dist["total_count"] == 10
        assert dist["window_count"] == 10
        assert dist["very_low_count"] == 2
        assert dist["low_count"] == 2
        assert dist["medium_count"] == 2
        assert dist["high_count"] == 2
        assert dist["very_high_count"] == 2

        # Check rates
        assert dist["very_low_rate"] == 0.2
        assert dist["low_rate"] == 0.2
        assert dist["medium_rate"] == 0.2
        assert dist["high_rate"] == 0.2
        assert dist["very_high_rate"] == 0.2

    def test_get_percentiles_insufficient_data(self) -> None:
        """Test getting percentiles with insufficient data."""
        monitor = ConfidenceDistributionMonitor()
        monitor.add_batch([0.9] * 5)  # Less than 10 samples

        percentiles = monitor.get_percentiles()
        assert percentiles is None

    def test_get_percentiles(self) -> None:
        """Test getting confidence percentiles."""
        monitor = ConfidenceDistributionMonitor()
        # Add 100 values from 0.5 to 1.0
        values = [0.5 + (i * 0.005) for i in range(100)]
        monitor.add_batch(values)

        percentiles = monitor.get_percentiles()
        assert percentiles is not None
        assert "p5" in percentiles
        assert "p25" in percentiles
        assert "p50" in percentiles
        assert "p75" in percentiles
        assert "p95" in percentiles
        assert "p99" in percentiles

        # Check ordering
        assert percentiles["p5"] <= percentiles["p25"]
        assert percentiles["p25"] <= percentiles["p50"]
        assert percentiles["p50"] <= percentiles["p75"]
        assert percentiles["p75"] <= percentiles["p95"]
        assert percentiles["p95"] <= percentiles["p99"]

    def test_check_threshold_alert_no_alerts(self) -> None:
        """Test threshold alert check with no alerts."""
        monitor = ConfidenceDistributionMonitor()
        # Add high confidence values
        monitor.add_batch([0.9, 0.95, 0.85, 0.92, 0.88] * 10)

        result = monitor.check_threshold_alert()
        assert result["has_alerts"] is False
        assert len(result["alerts"]) == 0

    def test_check_threshold_alert_low_confidence(self) -> None:
        """Test threshold alert for high low confidence rate."""
        monitor = ConfidenceDistributionMonitor()
        # Add mostly low confidence values
        monitor.add_batch([0.55, 0.6, 0.65] * 20)  # Low (0.5-0.7)
        monitor.add_batch([0.9] * 10)  # Some high values

        result = monitor.check_threshold_alert(low_rate_threshold=0.3)
        assert result["has_alerts"] is True
        assert any(a["type"] == "low_confidence_rate" for a in result["alerts"])

    def test_check_threshold_alert_very_low_confidence(self) -> None:
        """Test threshold alert for high very low confidence rate."""
        monitor = ConfidenceDistributionMonitor()
        # Add some very low confidence values
        monitor.add_batch([0.3, 0.4, 0.45] * 10)  # Very low (<0.5)
        monitor.add_batch([0.9] * 50)  # Mostly high values

        result = monitor.check_threshold_alert(very_low_rate_threshold=0.1)
        assert result["has_alerts"] is True
        assert any(a["type"] == "very_low_confidence_rate" for a in result["alerts"])

    def test_reset(self) -> None:
        """Test resetting the monitor."""
        monitor = ConfidenceDistributionMonitor()
        monitor.add_batch([0.9] * 50)

        monitor.reset()
        dist = monitor.get_distribution()

        assert dist["total_count"] == 0
        assert dist["window_count"] == 0

    def test_window_size_limit(self) -> None:
        """Test that window respects size limit."""
        monitor = ConfidenceDistributionMonitor(window_size=10)

        for _ in range(20):
            monitor.add_confidence(0.9)

        dist = monitor.get_distribution()
        assert dist["total_count"] == 20  # Total is tracked
        assert dist["window_count"] == 10  # Window is limited


class TestAnomalyResult:
    """Test AnomalyResult dataclass."""

    def test_insufficient_data_result(self) -> None:
        """Test result for insufficient data."""
        result = AnomalyResult(sufficient_data=False)
        assert result.sufficient_data is False
        assert result.anomaly_detected is False
        assert result.z_score is None

    def test_full_result(self) -> None:
        """Test full result with all fields."""
        result = AnomalyResult(
            sufficient_data=True,
            current_mean=0.75,
            current_std=0.15,
            z_score=3.5,
            anomaly_detected=True,
            severity="high",
        )
        assert result.sufficient_data is True
        assert result.current_mean == 0.75
        assert result.current_std == 0.15
        assert result.z_score == 3.5
        assert result.anomaly_detected is True
        assert result.severity == "high"
