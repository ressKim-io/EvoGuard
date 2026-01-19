"""Tests for PredictionLogger."""

import pytest

from ml_service.monitoring.prediction.logger import PredictionLogger, PredictionRecord


class TestPredictionLogger:
    """Test PredictionLogger class."""

    def test_init(self) -> None:
        """Test PredictionLogger initialization."""
        logger = PredictionLogger("test_model")
        assert logger.model_name == "test_model"
        assert logger.threshold == 0.7
        assert logger.buffer_size == 1000

    def test_init_custom_params(self) -> None:
        """Test PredictionLogger with custom parameters."""
        logger = PredictionLogger(
            model_name="custom_model",
            low_confidence_threshold=0.8,
            buffer_size=500,
        )
        assert logger.threshold == 0.8
        assert logger.buffer_size == 500

    def test_log_prediction_returns_record(self) -> None:
        """Test that log_prediction returns a PredictionRecord."""
        logger = PredictionLogger("test_record")
        record = logger.log_prediction(prediction=1, confidence=0.95)

        assert isinstance(record, PredictionRecord)
        assert record.model_name == "test_record"
        assert record.prediction == 1
        assert record.confidence == 0.95
        assert record.prediction_id is not None

    def test_log_prediction_with_input_data(self) -> None:
        """Test logging prediction with input data for hashing."""
        logger = PredictionLogger("test_hash")
        record = logger.log_prediction(
            prediction=0,
            confidence=0.75,
            input_data="test input text",
        )

        assert record.input_hash is not None
        assert len(record.input_hash) == 16  # SHA256 truncated to 16 chars

    def test_log_prediction_with_bytes_input(self) -> None:
        """Test logging prediction with bytes input data."""
        logger = PredictionLogger("test_bytes")
        record = logger.log_prediction(
            prediction=1,
            confidence=0.9,
            input_data=b"binary input data",
        )

        assert record.input_hash is not None

    def test_log_prediction_with_metadata(self) -> None:
        """Test logging prediction with metadata."""
        logger = PredictionLogger("test_meta")
        metadata = {"request_id": "req-123", "user_id": "user-456"}
        record = logger.log_prediction(
            prediction=1,
            confidence=0.88,
            metadata=metadata,
        )

        assert record.metadata == metadata

    def test_log_prediction_updates_counters(self) -> None:
        """Test that logging predictions updates counters."""
        logger = PredictionLogger("test_counters")

        logger.log_prediction(prediction=1, confidence=0.95)
        logger.log_prediction(prediction=0, confidence=0.85)
        logger.log_prediction(prediction=1, confidence=0.6)  # Low confidence

        stats = logger.get_statistics()
        assert stats["total_predictions"] == 3
        assert stats["positive_count"] == 2
        assert stats["negative_count"] == 1
        assert stats["low_confidence_count"] == 1

    def test_log_batch_predictions(self) -> None:
        """Test logging a batch of predictions."""
        logger = PredictionLogger("test_batch")

        predictions = [1, 0, 1, 0, 1]
        confidences = [0.9, 0.8, 0.7, 0.6, 0.95]

        records = logger.log_batch_predictions(predictions, confidences)

        assert len(records) == 5
        assert all(isinstance(r, PredictionRecord) for r in records)
        assert logger.get_statistics()["total_predictions"] == 5

    def test_log_batch_predictions_with_input_data(self) -> None:
        """Test logging batch predictions with input data."""
        logger = PredictionLogger("test_batch_input")

        predictions = [1, 0]
        confidences = [0.9, 0.8]
        input_data = ["text 1", "text 2"]

        records = logger.log_batch_predictions(predictions, confidences, input_data)

        assert all(r.input_hash is not None for r in records)

    def test_log_batch_predictions_length_mismatch_raises(self) -> None:
        """Test that mismatched lengths raise ValueError."""
        logger = PredictionLogger("test_mismatch")

        with pytest.raises(ValueError, match="same length"):
            logger.log_batch_predictions([1, 0], [0.9])

    def test_log_batch_predictions_input_length_mismatch_raises(self) -> None:
        """Test that mismatched input_data length raises ValueError."""
        logger = PredictionLogger("test_input_mismatch")

        with pytest.raises(ValueError, match="same length"):
            logger.log_batch_predictions([1, 0], [0.9, 0.8], ["only one"])

    def test_get_statistics(self) -> None:
        """Test getting prediction statistics."""
        logger = PredictionLogger("test_stats")

        for i in range(10):
            logger.log_prediction(
                prediction=i % 2,
                confidence=0.7 + (i * 0.02),
            )

        stats = logger.get_statistics()

        assert stats["model_name"] == "test_stats"
        assert stats["total_predictions"] == 10
        assert stats["positive_count"] == 5
        assert stats["negative_count"] == 5
        assert 0.0 <= stats["positive_rate"] <= 1.0
        assert stats["buffer_size"] == 10

    def test_get_recent_predictions(self) -> None:
        """Test getting recent predictions."""
        logger = PredictionLogger("test_recent", buffer_size=100)

        for _ in range(20):
            logger.log_prediction(prediction=1, confidence=0.8)

        recent = logger.get_recent_predictions(n=5)
        assert len(recent) == 5

    def test_get_recent_predictions_less_than_n(self) -> None:
        """Test getting recent predictions when buffer has less than n."""
        logger = PredictionLogger("test_recent_less")

        for _ in range(3):
            logger.log_prediction(prediction=1, confidence=0.8)

        recent = logger.get_recent_predictions(n=10)
        assert len(recent) == 3

    def test_get_low_confidence_predictions(self) -> None:
        """Test getting low confidence predictions."""
        logger = PredictionLogger("test_low", low_confidence_threshold=0.7)

        # Log some high and low confidence predictions
        logger.log_prediction(prediction=1, confidence=0.95)
        logger.log_prediction(prediction=0, confidence=0.5)  # Low
        logger.log_prediction(prediction=1, confidence=0.85)
        logger.log_prediction(prediction=0, confidence=0.6)  # Low
        logger.log_prediction(prediction=1, confidence=0.65)  # Low

        low_conf = logger.get_low_confidence_predictions()
        assert len(low_conf) == 3

    def test_reset_counters(self) -> None:
        """Test resetting counters."""
        logger = PredictionLogger("test_reset")

        for _ in range(5):
            logger.log_prediction(prediction=1, confidence=0.8)

        logger.reset_counters()
        stats = logger.get_statistics()

        assert stats["total_predictions"] == 0
        assert stats["positive_count"] == 0
        assert stats["negative_count"] == 0
        assert stats["low_confidence_count"] == 0

    def test_clear_buffer(self) -> None:
        """Test clearing the prediction buffer."""
        logger = PredictionLogger("test_clear")

        for _ in range(5):
            logger.log_prediction(prediction=1, confidence=0.8)

        logger.clear_buffer()
        assert len(logger.get_recent_predictions()) == 0

    def test_track_latency_context_manager(self) -> None:
        """Test the track_latency context manager."""
        logger = PredictionLogger("test_latency")

        import time

        with logger.track_latency():
            time.sleep(0.01)  # Sleep 10ms
            record = logger.log_prediction(prediction=1, confidence=0.9)

        # Latency should be captured
        assert record.latency_seconds is not None
        assert record.latency_seconds >= 0.01

    def test_buffer_size_limit(self) -> None:
        """Test that buffer respects size limit."""
        logger = PredictionLogger("test_buffer_limit", buffer_size=10)

        for _ in range(20):
            logger.log_prediction(prediction=1, confidence=0.8)

        assert len(logger.get_recent_predictions(n=100)) == 10
