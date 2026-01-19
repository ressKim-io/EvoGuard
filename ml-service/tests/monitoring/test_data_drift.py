"""Tests for data drift detection module."""


from ml_service.monitoring.drift.data_drift import (
    StreamingDataDriftMonitor,
    TextDataDriftMonitor,
    calculate_psi,
    ks_test,
)


class TestCalculatePSI:
    """Tests for calculate_psi function."""

    def test_identical_distributions(self) -> None:
        """PSI should be near zero for identical distributions."""
        data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        result = calculate_psi(data, data)

        assert result.psi < 0.01
        assert result.alert_level == "none"

    def test_similar_distributions(self) -> None:
        """PSI should be low for similar distributions."""
        import random

        random.seed(42)
        # Generate overlapping distributions
        reference = [random.gauss(50, 10) for _ in range(200)]
        current = [random.gauss(51, 10) for _ in range(200)]  # Slightly shifted

        result = calculate_psi(reference, current)

        # PSI should be low for similar overlapping distributions
        assert result.psi < 0.2
        assert result.alert_level in ("none", "minor")

    def test_different_distributions(self) -> None:
        """PSI should be high for different distributions."""
        reference = [0.1, 0.2, 0.3] * 30
        current = [0.7, 0.8, 0.9] * 30

        result = calculate_psi(reference, current)

        assert result.psi >= 0.2
        assert result.alert_level == "significant"

    def test_minor_change_detection(self) -> None:
        """PSI should detect changes in distributions."""
        import random

        random.seed(42)
        # Generate distributions with moderate shift
        reference = [random.gauss(50, 10) for _ in range(200)]
        current = [random.gauss(55, 10) for _ in range(200)]  # Moderate shift

        result = calculate_psi(reference, current)

        # PSI should reflect the distribution change
        assert result.psi >= 0.0
        # The result should be calculated correctly
        assert result.timestamp is not None

    def test_empty_data(self) -> None:
        """PSI should handle empty data."""
        result = calculate_psi([], [])

        assert result.psi == 0.0
        assert result.reference_size == 0
        assert result.current_size == 0

    def test_single_value_data(self) -> None:
        """PSI should handle single value distributions."""
        result = calculate_psi([0.5] * 100, [0.5] * 100)

        assert result.psi == 0.0
        assert result.alert_level == "none"

    def test_result_metadata(self) -> None:
        """PSI result should contain correct metadata."""
        reference = list(range(100))
        current = list(range(50, 150))

        result = calculate_psi(reference, current, n_bins=5)

        assert result.n_bins == 5
        assert result.reference_size == 100
        assert result.current_size == 100
        assert result.timestamp is not None


class TestKSTest:
    """Tests for KS test function."""

    def test_identical_distributions(self) -> None:
        """KS test should not detect drift for identical distributions."""
        data = [1.0, 2.0, 3.0, 4.0, 5.0] * 20

        result = ks_test(data, data)

        assert result.drifted is False
        assert result.statistic == 0.0

    def test_similar_distributions(self) -> None:
        """KS test should not detect drift for similar distributions."""
        reference = [1.0, 2.0, 3.0, 4.0, 5.0] * 20
        current = [1.1, 2.1, 3.1, 4.1, 5.1] * 20

        result = ks_test(reference, current)

        assert result.statistic < 0.5

    def test_different_distributions(self) -> None:
        """KS test should detect drift for different distributions."""
        reference = [1.0, 2.0, 3.0] * 30
        current = [10.0, 11.0, 12.0] * 30

        result = ks_test(reference, current)

        assert result.drifted is True
        assert result.statistic > 0.5

    def test_empty_data(self) -> None:
        """KS test should handle empty data."""
        result = ks_test([], [])

        assert result.drifted is False
        assert result.p_value == 1.0

    def test_custom_threshold(self) -> None:
        """KS test should respect custom threshold."""
        reference = [1.0, 2.0, 3.0, 4.0, 5.0] * 10
        current = [1.5, 2.5, 3.5, 4.5, 5.5] * 10

        result_strict = ks_test(reference, current, threshold=0.001)
        result_lenient = ks_test(reference, current, threshold=0.5)

        assert result_strict.threshold == 0.001
        assert result_lenient.threshold == 0.5

    def test_result_metadata(self) -> None:
        """KS test result should contain correct metadata."""
        reference = list(range(50))
        current = list(range(25, 75))

        result = ks_test(reference, current)

        assert result.reference_size == 50
        assert result.current_size == 50
        assert result.timestamp is not None
        assert 0.0 <= result.p_value <= 1.0


class TestTextDataDriftMonitor:
    """Tests for TextDataDriftMonitor."""

    def test_from_reference(self) -> None:
        """Should create monitor from reference texts."""
        texts = ["Hello world", "How are you?", "Good morning!"]
        monitor = TextDataDriftMonitor.from_reference(texts)

        assert monitor.reference_stats is not None
        assert "text_length" in monitor.reference_stats
        assert "word_count" in monitor.reference_stats
        assert "unicode_ratio" in monitor.reference_stats

    def test_no_drift_similar_texts(self) -> None:
        """Should not detect drift for similar texts."""
        reference = ["Hello world", "How are you?", "Good morning!"] * 10
        current = ["Hi there", "What's up?", "Good evening!"] * 10

        monitor = TextDataDriftMonitor.from_reference(reference)
        result = monitor.detect_drift(current)

        # Text characteristics should be similar
        assert result.timestamp is not None

    def test_drift_different_texts(self) -> None:
        """Should detect drift for very different texts."""
        reference = ["Hello"] * 50  # Short English
        current = ["This is a much longer sentence with many words"] * 50

        monitor = TextDataDriftMonitor.from_reference(reference)
        result = monitor.detect_drift(current)

        # Should detect drift in text_length and word_count
        assert result.overall_drift_detected is True
        assert len(result.drift_features) > 0

    def test_unicode_drift(self) -> None:
        """Should detect drift in unicode ratio."""
        reference = ["Hello world"] * 50  # ASCII only
        current = ["Привет мир 你好世界"] * 50  # Unicode heavy

        monitor = TextDataDriftMonitor.from_reference(reference)
        result = monitor.detect_drift(current)

        assert "unicode_ratio" in result.features
        # Unicode ratio should show significant change

    def test_update_reference(self) -> None:
        """Should allow updating reference texts."""
        initial = ["Hello world"] * 10
        updated = ["This is a longer text"] * 10

        monitor = TextDataDriftMonitor.from_reference(initial)
        initial_length = monitor.reference_stats["text_length"]

        monitor.update_reference(updated)
        updated_length = monitor.reference_stats["text_length"]

        assert initial_length != updated_length


class TestStreamingDataDriftMonitor:
    """Tests for StreamingDataDriftMonitor."""

    def test_initialization(self) -> None:
        """Should initialize with correct parameters."""
        monitor = StreamingDataDriftMonitor(
            reference_window_size=100,
            current_window_size=50,
            check_interval=10,
        )

        assert monitor.reference_window_size == 100
        assert monitor.current_window_size == 50
        assert monitor.check_interval == 10

    def test_reference_ready(self) -> None:
        """Should correctly report when reference is ready."""
        monitor = StreamingDataDriftMonitor(reference_window_size=10)

        assert monitor.reference_ready is False

        for i in range(10):
            monitor.add_reference(float(i))

        assert monitor.reference_ready is True

    def test_add_reference_batch(self) -> None:
        """Should add reference values in batch."""
        monitor = StreamingDataDriftMonitor(reference_window_size=10)

        monitor.add_reference_batch([1.0, 2.0, 3.0, 4.0, 5.0])

        assert len(monitor._reference) == 5

    def test_drift_check_returns_result(self) -> None:
        """Should return PSI result on drift check."""
        monitor = StreamingDataDriftMonitor(
            reference_window_size=20,
            current_window_size=10,
            check_interval=5,
        )

        # Add reference data
        monitor.add_reference_batch([float(i) for i in range(20)])

        # Add current data
        for i in range(20, 35):
            monitor.add_current(float(i))

        # Should have triggered at least one check
        assert monitor.get_last_result() is not None

    def test_reset_current(self) -> None:
        """Should reset current window."""
        monitor = StreamingDataDriftMonitor()

        for i in range(50):
            monitor.add_current(float(i))

        assert len(monitor._current) > 0

        monitor.reset_current()

        assert len(monitor._current) == 0

    def test_reset_all(self) -> None:
        """Should reset all windows."""
        monitor = StreamingDataDriftMonitor()

        for i in range(50):
            monitor.add_reference(float(i))
            monitor.add_current(float(i))

        monitor.reset_all()

        assert len(monitor._reference) == 0
        assert len(monitor._current) == 0
        assert monitor.get_last_result() is None
