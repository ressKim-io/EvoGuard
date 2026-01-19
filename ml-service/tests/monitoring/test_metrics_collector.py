"""Tests for Prometheus metrics collector."""


from ml_service.monitoring.metrics.collector import MetricsCollector


class TestMetricsCollector:
    """Test MetricsCollector class."""

    def test_init(self) -> None:
        """Test MetricsCollector initialization."""
        collector = MetricsCollector("test_model")
        assert collector.model_name == "test_model"

    def test_get_instance_returns_same_instance(self) -> None:
        """Test that get_instance returns the same instance for same model."""
        collector1 = MetricsCollector.get_instance("singleton_model")
        collector2 = MetricsCollector.get_instance("singleton_model")
        assert collector1 is collector2

    def test_get_instance_different_models(self) -> None:
        """Test that get_instance returns different instances for different models."""
        collector1 = MetricsCollector.get_instance("model_a")
        collector2 = MetricsCollector.get_instance("model_b")
        assert collector1 is not collector2

    def test_record_prediction_positive(self) -> None:
        """Test recording a positive prediction."""
        collector = MetricsCollector("test_pred_pos")
        # Should not raise
        collector.record_prediction(prediction=1, confidence=0.95)

    def test_record_prediction_negative(self) -> None:
        """Test recording a negative prediction."""
        collector = MetricsCollector("test_pred_neg")
        # Should not raise
        collector.record_prediction(prediction=0, confidence=0.85)

    def test_record_prediction_low_confidence(self) -> None:
        """Test recording a low confidence prediction."""
        collector = MetricsCollector("test_low_conf")
        # Should not raise
        collector.record_prediction(
            prediction=1,
            confidence=0.5,
            low_confidence_threshold=0.7,
        )

    def test_record_latency(self) -> None:
        """Test recording prediction latency."""
        collector = MetricsCollector("test_latency")
        # Should not raise
        collector.record_latency(0.025)

    def test_update_data_drift(self) -> None:
        """Test updating data drift PSI."""
        collector = MetricsCollector("test_drift")
        # Should not raise
        collector.update_data_drift("text_length", 0.15)

    def test_update_concept_drift_detected(self) -> None:
        """Test updating concept drift detection flag."""
        collector = MetricsCollector("test_concept")
        collector.update_concept_drift(detected=True)
        collector.update_concept_drift(detected=False)

    def test_update_drift_score(self) -> None:
        """Test updating drift score."""
        collector = MetricsCollector("test_drift_score")
        collector.update_drift_score("data", 0.18)
        collector.update_drift_score("concept", 0.05)

    def test_update_f1_score(self) -> None:
        """Test updating F1 score."""
        collector = MetricsCollector("test_f1")
        collector.update_f1_score(0.87, "production")
        collector.update_f1_score(0.92, "validation")

    def test_update_precision(self) -> None:
        """Test updating precision."""
        collector = MetricsCollector("test_precision")
        collector.update_precision(0.89, "production")

    def test_update_recall(self) -> None:
        """Test updating recall."""
        collector = MetricsCollector("test_recall")
        collector.update_recall(0.85, "production")

    def test_update_performance_metrics(self) -> None:
        """Test updating all performance metrics at once."""
        collector = MetricsCollector("test_perf")
        collector.update_performance_metrics(
            f1=0.87,
            precision=0.89,
            recall=0.85,
            dataset="production",
        )

    def test_recommend_retrain(self) -> None:
        """Test updating retrain recommendation."""
        collector = MetricsCollector("test_retrain")
        collector.recommend_retrain(True)
        collector.recommend_retrain(False)

    def test_update_last_retrain_timestamp(self) -> None:
        """Test updating last retrain timestamp."""
        collector = MetricsCollector("test_timestamp")
        import time

        collector.update_last_retrain_timestamp(time.time())
