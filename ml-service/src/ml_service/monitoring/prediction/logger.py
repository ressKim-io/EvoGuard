"""Prediction Logger - Log and track model predictions."""

import hashlib
import logging
import time
from collections import deque
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ml_service.monitoring.metrics.collector import MetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """Record of a single prediction."""

    prediction_id: str
    model_name: str
    prediction: int
    confidence: float
    input_hash: str | None
    timestamp: datetime
    latency_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class PredictionLogger:
    """Log predictions and track prediction statistics.

    This class provides utilities for logging model predictions,
    tracking prediction statistics, and integrating with Prometheus metrics.

    Example:
        >>> pred_logger = PredictionLogger("text_classifier")
        >>> with pred_logger.track_latency():
        ...     prediction, confidence = model.predict(input_data)
        >>> pred_logger.log_prediction(prediction, confidence, input_data)
    """

    def __init__(
        self,
        model_name: str,
        low_confidence_threshold: float = 0.7,
        buffer_size: int = 1000,
        sample_rate: float = 0.01,
    ) -> None:
        """Initialize the prediction logger.

        Args:
            model_name: Name of the model.
            low_confidence_threshold: Threshold below which predictions
                are considered low confidence.
            buffer_size: Size of the in-memory prediction buffer.
            sample_rate: Rate at which to sample predictions for detailed logging.
        """
        self.model_name = model_name
        self.threshold = low_confidence_threshold
        self.buffer_size = buffer_size
        self.sample_rate = sample_rate

        self.metrics_collector = MetricsCollector.get_instance(model_name)
        self._prediction_buffer: deque[PredictionRecord] = deque(maxlen=buffer_size)
        self._start_time: float | None = None

        # Counters
        self._total_predictions = 0
        self._low_confidence_count = 0
        self._positive_count = 0
        self._negative_count = 0

    @contextmanager
    def track_latency(self) -> Generator[None, None, None]:
        """Context manager to track prediction latency.

        Example:
            >>> with pred_logger.track_latency():
            ...     result = model.predict(data)
        """
        self._start_time = time.perf_counter()
        try:
            yield
        finally:
            if self._start_time is not None:
                latency = time.perf_counter() - self._start_time
                self.metrics_collector.record_latency(latency)
                self._start_time = None

    def log_prediction(
        self,
        prediction: int,
        confidence: float,
        input_data: str | bytes | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PredictionRecord:
        """Log a prediction with optional input data.

        Args:
            prediction: Prediction class (0 or 1).
            confidence: Prediction confidence score.
            input_data: Optional input data for hashing.
            metadata: Optional additional metadata.

        Returns:
            PredictionRecord with logged information.
        """
        # Calculate latency if tracking was started
        latency = None
        if self._start_time is not None:
            latency = time.perf_counter() - self._start_time
            self._start_time = None

        # Hash input data if provided
        input_hash = None
        if input_data is not None:
            if isinstance(input_data, str):
                input_data = input_data.encode("utf-8")
            input_hash = hashlib.sha256(input_data).hexdigest()[:16]

        # Create record
        record = PredictionRecord(
            prediction_id=str(uuid4()),
            model_name=self.model_name,
            prediction=prediction,
            confidence=confidence,
            input_hash=input_hash,
            timestamp=datetime.now(UTC),
            latency_seconds=latency,
            metadata=metadata or {},
        )

        # Update metrics
        self.metrics_collector.record_prediction(
            prediction=prediction,
            confidence=confidence,
            low_confidence_threshold=self.threshold,
        )

        if latency is not None:
            self.metrics_collector.record_latency(latency)

        # Update counters
        self._total_predictions += 1
        if prediction == 1:
            self._positive_count += 1
        else:
            self._negative_count += 1
        if confidence < self.threshold:
            self._low_confidence_count += 1

        # Add to buffer
        self._prediction_buffer.append(record)

        return record

    def log_batch_predictions(
        self,
        predictions: list[int],
        confidences: list[float],
        input_data_list: list[str | bytes] | None = None,
    ) -> list[PredictionRecord]:
        """Log a batch of predictions.

        Args:
            predictions: List of prediction classes.
            confidences: List of confidence scores.
            input_data_list: Optional list of input data.

        Returns:
            List of PredictionRecords.
        """
        if len(predictions) != len(confidences):
            raise ValueError("predictions and confidences must have same length")

        if input_data_list is not None and len(input_data_list) != len(predictions):
            raise ValueError("input_data_list must have same length as predictions")

        records = []
        for i, (pred, conf) in enumerate(zip(predictions, confidences, strict=True)):
            input_data = input_data_list[i] if input_data_list else None
            record = self.log_prediction(pred, conf, input_data)
            records.append(record)

        return records

    def get_statistics(self) -> dict[str, Any]:
        """Get current prediction statistics.

        Returns:
            Dictionary with prediction statistics.
        """
        low_confidence_rate = (
            self._low_confidence_count / self._total_predictions
            if self._total_predictions > 0
            else 0.0
        )
        positive_rate = (
            self._positive_count / self._total_predictions
            if self._total_predictions > 0
            else 0.0
        )

        # Calculate statistics from buffer
        buffer_confidences = [r.confidence for r in self._prediction_buffer]
        buffer_latencies = [
            r.latency_seconds
            for r in self._prediction_buffer
            if r.latency_seconds is not None
        ]

        avg_confidence = (
            sum(buffer_confidences) / len(buffer_confidences)
            if buffer_confidences
            else 0.0
        )
        avg_latency = (
            sum(buffer_latencies) / len(buffer_latencies) if buffer_latencies else 0.0
        )

        return {
            "model_name": self.model_name,
            "total_predictions": self._total_predictions,
            "positive_count": self._positive_count,
            "negative_count": self._negative_count,
            "positive_rate": positive_rate,
            "low_confidence_count": self._low_confidence_count,
            "low_confidence_rate": low_confidence_rate,
            "buffer_size": len(self._prediction_buffer),
            "avg_confidence": avg_confidence,
            "avg_latency_seconds": avg_latency,
        }

    def get_recent_predictions(self, n: int = 100) -> list[PredictionRecord]:
        """Get the most recent predictions from the buffer.

        Args:
            n: Number of recent predictions to return.

        Returns:
            List of recent PredictionRecords.
        """
        return list(self._prediction_buffer)[-n:]

    def get_low_confidence_predictions(
        self,
        n: int = 100,
    ) -> list[PredictionRecord]:
        """Get recent low confidence predictions.

        Args:
            n: Maximum number of predictions to return.

        Returns:
            List of low confidence PredictionRecords.
        """
        low_conf = [r for r in self._prediction_buffer if r.confidence < self.threshold]
        return low_conf[-n:]

    def reset_counters(self) -> None:
        """Reset all prediction counters."""
        self._total_predictions = 0
        self._low_confidence_count = 0
        self._positive_count = 0
        self._negative_count = 0

    def clear_buffer(self) -> None:
        """Clear the prediction buffer."""
        self._prediction_buffer.clear()
