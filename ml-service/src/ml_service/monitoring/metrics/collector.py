"""Prometheus Metrics Collector - ML model monitoring metrics."""

import logging
from typing import ClassVar

from prometheus_client import Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


# Prediction metrics
PREDICTION_CONFIDENCE = Histogram(
    "ml_prediction_confidence",
    "Model prediction confidence distribution",
    ["model", "prediction"],
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

PREDICTION_COUNT = Counter(
    "ml_predictions_total",
    "Total predictions",
    ["model", "prediction"],
)

LOW_CONFIDENCE_COUNT = Counter(
    "ml_low_confidence_predictions_total",
    "Predictions with confidence below threshold",
    ["model"],
)

PREDICTION_LATENCY = Histogram(
    "ml_prediction_latency_seconds",
    "Model prediction latency in seconds",
    ["model"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

# Drift metrics
DATA_DRIFT_PSI = Gauge(
    "ml_data_drift_psi",
    "Population Stability Index for data drift",
    ["model", "feature"],
)

CONCEPT_DRIFT_DETECTED = Gauge(
    "ml_concept_drift_detected",
    "Concept drift detection flag (1=detected)",
    ["model"],
)

MODEL_DRIFT_SCORE = Gauge(
    "ml_drift_score",
    "Current drift score",
    ["model", "drift_type"],
)

# Model performance metrics
MODEL_F1_SCORE = Gauge(
    "ml_model_f1_score",
    "Current model F1 score",
    ["model", "dataset"],
)

MODEL_PRECISION = Gauge(
    "ml_model_precision",
    "Current model precision",
    ["model", "dataset"],
)

MODEL_RECALL = Gauge(
    "ml_model_recall",
    "Current model recall",
    ["model", "dataset"],
)

# Retraining metrics
RETRAIN_RECOMMENDED = Gauge(
    "ml_retrain_recommended",
    "Model retrain recommendation flag",
    ["model"],
)

LAST_RETRAIN_TIMESTAMP = Gauge(
    "ml_last_retrain_timestamp",
    "Timestamp of last model retrain",
    ["model"],
)


class MetricsCollector:
    """Collect and update ML model monitoring metrics.

    This class provides a centralized interface for updating Prometheus
    metrics related to ML model monitoring.

    Example:
        >>> collector = MetricsCollector("text_classifier")
        >>> collector.record_prediction(1, 0.95)
        >>> collector.update_f1_score(0.87, "production")
    """

    _instances: ClassVar[dict[str, "MetricsCollector"]] = {}

    def __init__(self, model_name: str) -> None:
        """Initialize the metrics collector.

        Args:
            model_name: Name of the model to track metrics for.
        """
        self.model_name = model_name

    @classmethod
    def get_instance(cls, model_name: str) -> "MetricsCollector":
        """Get or create a MetricsCollector instance for a model.

        Args:
            model_name: Name of the model.

        Returns:
            MetricsCollector instance.
        """
        if model_name not in cls._instances:
            cls._instances[model_name] = cls(model_name)
        return cls._instances[model_name]

    def record_prediction(
        self,
        prediction: int,
        confidence: float,
        low_confidence_threshold: float = 0.7,
    ) -> None:
        """Record a prediction with its confidence.

        Args:
            prediction: Prediction class (0 or 1).
            confidence: Prediction confidence score.
            low_confidence_threshold: Threshold for low confidence alerts.
        """
        label = "positive" if prediction == 1 else "negative"

        PREDICTION_CONFIDENCE.labels(
            model=self.model_name,
            prediction=label,
        ).observe(confidence)

        PREDICTION_COUNT.labels(
            model=self.model_name,
            prediction=label,
        ).inc()

        if confidence < low_confidence_threshold:
            LOW_CONFIDENCE_COUNT.labels(
                model=self.model_name,
            ).inc()

    def record_latency(self, latency_seconds: float) -> None:
        """Record prediction latency.

        Args:
            latency_seconds: Prediction latency in seconds.
        """
        PREDICTION_LATENCY.labels(
            model=self.model_name,
        ).observe(latency_seconds)

    def update_data_drift(self, feature: str, psi: float) -> None:
        """Update Data Drift PSI metric.

        Args:
            feature: Feature name.
            psi: Population Stability Index value.
        """
        DATA_DRIFT_PSI.labels(
            model=self.model_name,
            feature=feature,
        ).set(psi)

    def update_concept_drift(self, detected: bool) -> None:
        """Update Concept Drift detection flag.

        Args:
            detected: Whether concept drift was detected.
        """
        CONCEPT_DRIFT_DETECTED.labels(
            model=self.model_name,
        ).set(1 if detected else 0)

    def update_drift_score(self, drift_type: str, score: float) -> None:
        """Update drift score metric.

        Args:
            drift_type: Type of drift (data, concept, feature).
            score: Drift score value.
        """
        MODEL_DRIFT_SCORE.labels(
            model=self.model_name,
            drift_type=drift_type,
        ).set(score)

    def update_f1_score(self, f1: float, dataset: str = "production") -> None:
        """Update model F1 score.

        Args:
            f1: F1 score value.
            dataset: Dataset type (production, validation).
        """
        MODEL_F1_SCORE.labels(
            model=self.model_name,
            dataset=dataset,
        ).set(f1)

    def update_precision(self, precision: float, dataset: str = "production") -> None:
        """Update model precision.

        Args:
            precision: Precision value.
            dataset: Dataset type.
        """
        MODEL_PRECISION.labels(
            model=self.model_name,
            dataset=dataset,
        ).set(precision)

    def update_recall(self, recall: float, dataset: str = "production") -> None:
        """Update model recall.

        Args:
            recall: Recall value.
            dataset: Dataset type.
        """
        MODEL_RECALL.labels(
            model=self.model_name,
            dataset=dataset,
        ).set(recall)

    def update_performance_metrics(
        self,
        f1: float,
        precision: float,
        recall: float,
        dataset: str = "production",
    ) -> None:
        """Update all performance metrics at once.

        Args:
            f1: F1 score.
            precision: Precision score.
            recall: Recall score.
            dataset: Dataset type.
        """
        self.update_f1_score(f1, dataset)
        self.update_precision(precision, dataset)
        self.update_recall(recall, dataset)

    def recommend_retrain(self, recommend: bool) -> None:
        """Update retrain recommendation flag.

        Args:
            recommend: Whether retraining is recommended.
        """
        RETRAIN_RECOMMENDED.labels(
            model=self.model_name,
        ).set(1 if recommend else 0)

    def update_last_retrain_timestamp(self, timestamp: float) -> None:
        """Update last retrain timestamp.

        Args:
            timestamp: Unix timestamp of last retrain.
        """
        LAST_RETRAIN_TIMESTAMP.labels(
            model=self.model_name,
        ).set(timestamp)
