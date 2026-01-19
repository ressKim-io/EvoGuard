"""Alert Rules - Threshold-based and drift-based alert rules.

This module provides alert rule definitions for ML model monitoring:
- Threshold-based rules (F1, confidence, latency)
- Drift-based rules (PSI, concept drift)
- Composite rules (multiple conditions)
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""

    FIRING = "firing"
    RESOLVED = "resolved"
    PENDING = "pending"


@dataclass
class Alert:
    """Represents an alert instance."""

    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    model_name: str
    metric_name: str
    metric_value: float
    threshold: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary.

        Returns:
            Dictionary representation of the alert.
        """
        return {
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "model_name": self.model_name,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "labels": self.labels,
            "annotations": self.annotations,
        }


class AlertRule(ABC):
    """Abstract base class for alert rules."""

    def __init__(
        self,
        name: str,
        severity: AlertSeverity,
        model_name: str,
        description: str = "",
    ) -> None:
        """Initialize the alert rule.

        Args:
            name: Rule name.
            severity: Alert severity level.
            model_name: Name of the model to monitor.
            description: Rule description.
        """
        self.name = name
        self.severity = severity
        self.model_name = model_name
        self.description = description
        self._firing = False
        self._last_check: datetime | None = None

    @abstractmethod
    def evaluate(self, metrics: dict[str, float]) -> Alert | None:
        """Evaluate the rule against current metrics.

        Args:
            metrics: Dictionary of metric name to value.

        Returns:
            Alert if rule fires, None otherwise.
        """
        pass

    @property
    def is_firing(self) -> bool:
        """Check if the rule is currently firing."""
        return self._firing


class ThresholdRule(AlertRule):
    """Threshold-based alert rule.

    Fires when a metric crosses a specified threshold.

    Example:
        >>> rule = ThresholdRule(
        ...     name="F1ScoreLow",
        ...     severity=AlertSeverity.CRITICAL,
        ...     model_name="text_classifier",
        ...     metric_name="f1_score",
        ...     threshold=0.7,
        ...     comparison="lt",
        ... )
        >>> alert = rule.evaluate({"f1_score": 0.65})
        >>> if alert:
        ...     print(f"Alert: {alert.message}")
    """

    def __init__(
        self,
        name: str,
        severity: AlertSeverity,
        model_name: str,
        metric_name: str,
        threshold: float,
        comparison: str = "gt",
        description: str = "",
        for_duration_seconds: int = 0,
    ) -> None:
        """Initialize the threshold rule.

        Args:
            name: Rule name.
            severity: Alert severity level.
            model_name: Model name.
            metric_name: Name of the metric to monitor.
            threshold: Threshold value.
            comparison: Comparison operator ("gt", "lt", "gte", "lte", "eq").
            description: Rule description.
            for_duration_seconds: Duration the condition must be true before firing.
        """
        super().__init__(name, severity, model_name, description)
        self.metric_name = metric_name
        self.threshold = threshold
        self.comparison = comparison
        self.for_duration_seconds = for_duration_seconds
        self._condition_start: datetime | None = None

    def evaluate(self, metrics: dict[str, float]) -> Alert | None:
        """Evaluate the threshold rule.

        Args:
            metrics: Dictionary of metric name to value.

        Returns:
            Alert if threshold is crossed, None otherwise.
        """
        self._last_check = datetime.now(UTC)

        if self.metric_name not in metrics:
            return None

        value = metrics[self.metric_name]
        condition_met = self._check_condition(value)

        if condition_met:
            # Check for_duration if specified
            if self.for_duration_seconds > 0:
                if self._condition_start is None:
                    self._condition_start = datetime.now(UTC)
                    return None

                elapsed = (datetime.now(UTC) - self._condition_start).total_seconds()
                if elapsed < self.for_duration_seconds:
                    return None

            self._firing = True
            return self._create_alert(value)
        else:
            self._firing = False
            self._condition_start = None
            return None

    def _check_condition(self, value: float) -> bool:
        """Check if the condition is met.

        Args:
            value: Current metric value.

        Returns:
            True if condition is met, False otherwise.
        """
        if self.comparison == "gt":
            return value > self.threshold
        elif self.comparison == "lt":
            return value < self.threshold
        elif self.comparison == "gte":
            return value >= self.threshold
        elif self.comparison == "lte":
            return value <= self.threshold
        elif self.comparison == "eq":
            return abs(value - self.threshold) < 1e-9
        return False

    def _create_alert(self, value: float) -> Alert:
        """Create an alert instance.

        Args:
            value: Current metric value.

        Returns:
            Alert instance.
        """
        comparison_text = {
            "gt": "above",
            "lt": "below",
            "gte": "at or above",
            "lte": "at or below",
            "eq": "equal to",
        }

        message = (
            f"{self.metric_name} is {comparison_text.get(self.comparison, '')} "
            f"threshold: {value:.4f} (threshold: {self.threshold:.4f})"
        )

        return Alert(
            name=self.name,
            severity=self.severity,
            status=AlertStatus.FIRING,
            message=message,
            model_name=self.model_name,
            metric_name=self.metric_name,
            metric_value=value,
            threshold=self.threshold,
            labels={
                "model": self.model_name,
                "rule": self.name,
            },
            annotations={
                "description": self.description,
            },
        )


class DriftRule(AlertRule):
    """Drift-based alert rule.

    Fires when drift is detected based on PSI or other drift metrics.

    Example:
        >>> rule = DriftRule(
        ...     name="DataDriftDetected",
        ...     severity=AlertSeverity.WARNING,
        ...     model_name="text_classifier",
        ...     drift_type="data",
        ...     psi_threshold=0.2,
        ... )
        >>> alert = rule.evaluate({"data_drift_psi": 0.25})
    """

    def __init__(
        self,
        name: str,
        severity: AlertSeverity,
        model_name: str,
        drift_type: str,
        psi_threshold: float = 0.2,
        f1_drop_threshold: float = 0.05,
        description: str = "",
    ) -> None:
        """Initialize the drift rule.

        Args:
            name: Rule name.
            severity: Alert severity level.
            model_name: Model name.
            drift_type: Type of drift ("data", "concept", "feature").
            psi_threshold: PSI threshold for data drift.
            f1_drop_threshold: F1 drop threshold for concept drift.
            description: Rule description.
        """
        super().__init__(name, severity, model_name, description)
        self.drift_type = drift_type
        self.psi_threshold = psi_threshold
        self.f1_drop_threshold = f1_drop_threshold

    def evaluate(self, metrics: dict[str, float]) -> Alert | None:
        """Evaluate the drift rule.

        Args:
            metrics: Dictionary of metric name to value.

        Returns:
            Alert if drift is detected, None otherwise.
        """
        self._last_check = datetime.now(UTC)

        if self.drift_type == "data":
            return self._evaluate_data_drift(metrics)
        elif self.drift_type == "concept":
            return self._evaluate_concept_drift(metrics)
        elif self.drift_type == "feature":
            return self._evaluate_feature_drift(metrics)

        return None

    def _evaluate_data_drift(self, metrics: dict[str, float]) -> Alert | None:
        """Evaluate data drift based on PSI.

        Args:
            metrics: Metrics dictionary.

        Returns:
            Alert if data drift detected, None otherwise.
        """
        psi_key = "data_drift_psi"
        if psi_key not in metrics:
            return None

        psi = metrics[psi_key]
        if psi >= self.psi_threshold:
            self._firing = True
            return Alert(
                name=self.name,
                severity=self.severity,
                status=AlertStatus.FIRING,
                message=f"Data drift detected: PSI={psi:.4f} (threshold: {self.psi_threshold})",
                model_name=self.model_name,
                metric_name=psi_key,
                metric_value=psi,
                threshold=self.psi_threshold,
                labels={
                    "model": self.model_name,
                    "drift_type": self.drift_type,
                },
                annotations={
                    "description": self.description or "Data distribution has shifted significantly",
                },
            )

        self._firing = False
        return None

    def _evaluate_concept_drift(self, metrics: dict[str, float]) -> Alert | None:
        """Evaluate concept drift based on F1 drop.

        Args:
            metrics: Metrics dictionary.

        Returns:
            Alert if concept drift detected, None otherwise.
        """
        f1_drop_key = "f1_drop"
        if f1_drop_key not in metrics:
            return None

        f1_drop = metrics[f1_drop_key]
        if f1_drop >= self.f1_drop_threshold:
            self._firing = True

            severity_text = "critical" if f1_drop > 0.15 else "significant" if f1_drop > 0.10 else "moderate"

            return Alert(
                name=self.name,
                severity=self.severity,
                status=AlertStatus.FIRING,
                message=f"Concept drift detected: F1 drop={f1_drop:.4f} ({severity_text})",
                model_name=self.model_name,
                metric_name=f1_drop_key,
                metric_value=f1_drop,
                threshold=self.f1_drop_threshold,
                labels={
                    "model": self.model_name,
                    "drift_type": self.drift_type,
                },
                annotations={
                    "description": self.description or "Model performance has degraded",
                },
            )

        self._firing = False
        return None

    def _evaluate_feature_drift(self, metrics: dict[str, float]) -> Alert | None:
        """Evaluate feature drift.

        Args:
            metrics: Metrics dictionary.

        Returns:
            Alert if feature drift detected, None otherwise.
        """
        # Check for any feature drift indicators
        drifted_features = []
        max_psi = 0.0

        for key, value in metrics.items():
            if key.startswith("feature_drift_") and value >= self.psi_threshold:
                feature_name = key.replace("feature_drift_", "")
                drifted_features.append(feature_name)
                max_psi = max(max_psi, value)

        if drifted_features:
            self._firing = True
            return Alert(
                name=self.name,
                severity=self.severity,
                status=AlertStatus.FIRING,
                message=f"Feature drift detected in {len(drifted_features)} features: {', '.join(drifted_features[:3])}",
                model_name=self.model_name,
                metric_name="feature_drift",
                metric_value=max_psi,
                threshold=self.psi_threshold,
                labels={
                    "model": self.model_name,
                    "drift_type": self.drift_type,
                    "drifted_features": ",".join(drifted_features),
                },
                annotations={
                    "description": self.description or "Feature distributions have changed",
                },
            )

        self._firing = False
        return None


class CompositeRule(AlertRule):
    """Composite alert rule combining multiple conditions.

    Example:
        >>> rule = CompositeRule(
        ...     name="ModelHealthCritical",
        ...     severity=AlertSeverity.CRITICAL,
        ...     model_name="text_classifier",
        ...     rules=[f1_rule, drift_rule],
        ...     operator="or",
        ... )
    """

    def __init__(
        self,
        name: str,
        severity: AlertSeverity,
        model_name: str,
        rules: list[AlertRule],
        operator: str = "and",
        description: str = "",
    ) -> None:
        """Initialize the composite rule.

        Args:
            name: Rule name.
            severity: Alert severity level.
            model_name: Model name.
            rules: List of child rules.
            operator: Logical operator ("and", "or").
            description: Rule description.
        """
        super().__init__(name, severity, model_name, description)
        self.rules = rules
        self.operator = operator

    def evaluate(self, metrics: dict[str, float]) -> Alert | None:
        """Evaluate all child rules.

        Args:
            metrics: Dictionary of metric name to value.

        Returns:
            Alert if composite condition is met, None otherwise.
        """
        self._last_check = datetime.now(UTC)

        alerts = []
        for rule in self.rules:
            alert = rule.evaluate(metrics)
            if alert:
                alerts.append(alert)

        if self.operator == "and":
            # All rules must fire
            if len(alerts) == len(self.rules):
                self._firing = True
                return self._create_composite_alert(alerts)
        elif self.operator == "or":
            # Any rule can fire
            if alerts:
                self._firing = True
                return self._create_composite_alert(alerts)

        self._firing = False
        return None

    def _create_composite_alert(self, child_alerts: list[Alert]) -> Alert:
        """Create a composite alert from child alerts.

        Args:
            child_alerts: List of child alerts that fired.

        Returns:
            Composite alert instance.
        """
        messages = [a.message for a in child_alerts]
        combined_message = f"Multiple conditions met: {'; '.join(messages)}"

        # Use the highest severity from child alerts
        max_severity = max(child_alerts, key=lambda a: list(AlertSeverity).index(a.severity)).severity

        return Alert(
            name=self.name,
            severity=max_severity,
            status=AlertStatus.FIRING,
            message=combined_message,
            model_name=self.model_name,
            metric_name="composite",
            metric_value=len(child_alerts),
            threshold=len(self.rules) if self.operator == "and" else 1,
            labels={
                "model": self.model_name,
                "rule": self.name,
                "child_rules": ",".join(a.name for a in child_alerts),
            },
            annotations={
                "description": self.description,
            },
        )


class AlertRuleRegistry:
    """Registry for managing alert rules.

    Example:
        >>> registry = AlertRuleRegistry()
        >>> registry.register(f1_rule)
        >>> registry.register(drift_rule)
        >>> alerts = registry.evaluate_all(current_metrics)
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._rules: dict[str, AlertRule] = {}

    def register(self, rule: AlertRule) -> None:
        """Register an alert rule.

        Args:
            rule: Alert rule to register.
        """
        self._rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name}")

    def unregister(self, rule_name: str) -> bool:
        """Unregister an alert rule.

        Args:
            rule_name: Name of the rule to unregister.

        Returns:
            True if rule was removed, False if not found.
        """
        if rule_name in self._rules:
            del self._rules[rule_name]
            logger.info(f"Unregistered alert rule: {rule_name}")
            return True
        return False

    def get_rule(self, rule_name: str) -> AlertRule | None:
        """Get a rule by name.

        Args:
            rule_name: Name of the rule.

        Returns:
            AlertRule or None if not found.
        """
        return self._rules.get(rule_name)

    def evaluate_all(self, metrics: dict[str, float]) -> list[Alert]:
        """Evaluate all registered rules.

        Args:
            metrics: Dictionary of metric name to value.

        Returns:
            List of alerts that fired.
        """
        alerts = []
        for rule in self._rules.values():
            alert = rule.evaluate(metrics)
            if alert:
                alerts.append(alert)
                logger.warning(f"Alert fired: {alert.name} - {alert.message}")
        return alerts

    def get_firing_rules(self) -> list[str]:
        """Get names of currently firing rules.

        Returns:
            List of rule names that are currently firing.
        """
        return [name for name, rule in self._rules.items() if rule.is_firing]

    def list_rules(self) -> list[dict[str, Any]]:
        """List all registered rules.

        Returns:
            List of rule information dictionaries.
        """
        return [
            {
                "name": rule.name,
                "severity": rule.severity.value,
                "model_name": rule.model_name,
                "description": rule.description,
                "is_firing": rule.is_firing,
            }
            for rule in self._rules.values()
        ]


def create_default_rules(model_name: str) -> list[AlertRule]:
    """Create a set of default alert rules for a model.

    Args:
        model_name: Name of the model.

    Returns:
        List of default alert rules.
    """
    rules = [
        # F1 score critical threshold
        ThresholdRule(
            name="ModelF1ScoreCritical",
            severity=AlertSeverity.CRITICAL,
            model_name=model_name,
            metric_name="f1_score",
            threshold=0.7,
            comparison="lt",
            description="Model F1 score dropped below critical threshold",
            for_duration_seconds=300,  # 5 minutes
        ),
        # F1 score warning threshold
        ThresholdRule(
            name="ModelF1ScoreWarning",
            severity=AlertSeverity.WARNING,
            model_name=model_name,
            metric_name="f1_score",
            threshold=0.8,
            comparison="lt",
            description="Model F1 score dropped below warning threshold",
            for_duration_seconds=600,  # 10 minutes
        ),
        # Low confidence spike
        ThresholdRule(
            name="LowConfidenceSpike",
            severity=AlertSeverity.WARNING,
            model_name=model_name,
            metric_name="low_confidence_rate",
            threshold=0.3,
            comparison="gt",
            description="High rate of low confidence predictions",
            for_duration_seconds=300,
        ),
        # Data drift detection
        DriftRule(
            name="DataDriftDetected",
            severity=AlertSeverity.WARNING,
            model_name=model_name,
            drift_type="data",
            psi_threshold=0.2,
            description="Significant data distribution shift detected",
        ),
        # Concept drift detection
        DriftRule(
            name="ConceptDriftDetected",
            severity=AlertSeverity.CRITICAL,
            model_name=model_name,
            drift_type="concept",
            f1_drop_threshold=0.05,
            description="Model performance degradation detected",
        ),
        # Feature drift detection
        DriftRule(
            name="FeatureDriftDetected",
            severity=AlertSeverity.WARNING,
            model_name=model_name,
            drift_type="feature",
            psi_threshold=0.2,
            description="Feature distribution shift detected",
        ),
        # High latency
        ThresholdRule(
            name="HighPredictionLatency",
            severity=AlertSeverity.WARNING,
            model_name=model_name,
            metric_name="prediction_latency_p99",
            threshold=1.0,
            comparison="gt",
            description="Prediction latency exceeds threshold",
        ),
    ]

    return rules
