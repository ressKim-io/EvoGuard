"""Alert System - Alert rules and handlers (Phase 3).

This module provides a complete alerting system for ML model monitoring:
- Threshold-based and drift-based alert rules
- Multiple notification handlers (Slack, Email, Webhook, AlertManager)
- Alert dispatching and management
"""

from ml_service.monitoring.alerts.handlers import (
    AlertDispatcher,
    AlertHandler,
    AlertManagerHandler,
    ConsoleHandler,
    EmailHandler,
    HandlerResult,
    SlackHandler,
    WebhookHandler,
)
from ml_service.monitoring.alerts.rules import (
    Alert,
    AlertRule,
    AlertRuleRegistry,
    AlertSeverity,
    AlertStatus,
    CompositeRule,
    DriftRule,
    ThresholdRule,
    create_default_rules,
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
    "CompositeRule",
    "ConsoleHandler",
    "DriftRule",
    "EmailHandler",
    "HandlerResult",
    "SlackHandler",
    "ThresholdRule",
    "WebhookHandler",
    "create_default_rules",
]
