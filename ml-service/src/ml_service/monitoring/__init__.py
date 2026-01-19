"""Model Monitoring - ML model performance and drift monitoring.

This module provides tools for monitoring ML model performance in production:
- Prometheus metrics collection
- Prediction logging and tracking
- Confidence distribution monitoring
- (Phase 2) Data and Concept Drift detection
- (Phase 3) Alert system
- (Phase 4) Automated actions
"""

from ml_service.monitoring.metrics import MetricsCollector
from ml_service.monitoring.prediction import (
    AnomalyResult,
    ConfidenceAnomalyDetector,
    ConfidenceDistributionMonitor,
    ConfidenceStats,
    PredictionLogger,
    PredictionRecord,
)

__all__ = [
    "AnomalyResult",
    "ConfidenceAnomalyDetector",
    "ConfidenceDistributionMonitor",
    "ConfidenceStats",
    "MetricsCollector",
    "PredictionLogger",
    "PredictionRecord",
]
