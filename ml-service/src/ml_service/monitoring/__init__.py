"""Model Monitoring - ML model performance and drift monitoring.

This module provides tools for monitoring ML model performance in production:
- Prometheus metrics collection
- Prediction logging and tracking
- Confidence distribution monitoring
- Data and Concept Drift detection (Phase 2)
- Alert system (Phase 3)
- Automated actions (Phase 4)
- A/B testing and report generation (Phase 5)
"""

from ml_service.monitoring.alerts import (
    Alert,
    AlertDispatcher,
    AlertHandler,
    AlertManagerHandler,
    AlertRule,
    AlertRuleRegistry,
    AlertSeverity,
    AlertStatus,
    CompositeRule,
    ConsoleHandler,
    DriftRule,
    EmailHandler,
    HandlerResult,
    SlackHandler,
    ThresholdRule,
    WebhookHandler,
    create_default_rules,
)
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
    "Alert",
    "AlertDispatcher",
    "AlertHandler",
    "AlertManagerHandler",
    "AlertRule",
    "AlertRuleRegistry",
    "AlertSeverity",
    "AlertStatus",
    "AnomalyResult",
    "CompositeRule",
    "ConfidenceAnomalyDetector",
    "ConfidenceDistributionMonitor",
    "ConfidenceStats",
    "ConsoleHandler",
    "DriftRule",
    "EmailHandler",
    "HandlerResult",
    "MetricsCollector",
    "PredictionLogger",
    "PredictionRecord",
    "SlackHandler",
    "ThresholdRule",
    "WebhookHandler",
    "create_default_rules",
]
