"""Alert Handlers - Send alerts to various destinations.

This module provides alert handlers for different notification channels:
- Slack notifications
- Email notifications
- Webhook (generic HTTP POST)
- AlertManager integration
- Console logging (for development)
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import aiohttp

from ml_service.monitoring.alerts.rules import Alert, AlertSeverity

logger = logging.getLogger(__name__)


@dataclass
class HandlerResult:
    """Result of alert handler execution."""

    success: bool
    handler_name: str
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    details: dict[str, Any] = field(default_factory=dict)


class AlertHandler(ABC):
    """Abstract base class for alert handlers."""

    def __init__(self, name: str, enabled: bool = True) -> None:
        """Initialize the handler.

        Args:
            name: Handler name.
            enabled: Whether the handler is enabled.
        """
        self.name = name
        self.enabled = enabled

    @abstractmethod
    async def send(self, alert: Alert) -> HandlerResult:
        """Send an alert.

        Args:
            alert: Alert to send.

        Returns:
            HandlerResult indicating success or failure.
        """
        pass

    async def send_batch(self, alerts: list[Alert]) -> list[HandlerResult]:
        """Send multiple alerts.

        Args:
            alerts: List of alerts to send.

        Returns:
            List of HandlerResults.
        """
        results = []
        for alert in alerts:
            result = await self.send(alert)
            results.append(result)
        return results


class ConsoleHandler(AlertHandler):
    """Console logging handler for development and debugging.

    Example:
        >>> handler = ConsoleHandler()
        >>> await handler.send(alert)
    """

    def __init__(self, enabled: bool = True) -> None:
        """Initialize the console handler."""
        super().__init__("console", enabled)

    async def send(self, alert: Alert) -> HandlerResult:
        """Log alert to console.

        Args:
            alert: Alert to log.

        Returns:
            HandlerResult.
        """
        if not self.enabled:
            return HandlerResult(
                success=False,
                handler_name=self.name,
                message="Handler is disabled",
            )

        severity_colors = {
            AlertSeverity.INFO: "\033[94m",  # Blue
            AlertSeverity.WARNING: "\033[93m",  # Yellow
            AlertSeverity.CRITICAL: "\033[91m",  # Red
        }
        reset = "\033[0m"

        color = severity_colors.get(alert.severity, "")
        log_message = (
            f"{color}[{alert.severity.value.upper()}]{reset} "
            f"{alert.name}: {alert.message} "
            f"(model: {alert.model_name}, value: {alert.metric_value:.4f})"
        )

        if alert.severity == AlertSeverity.CRITICAL:
            logger.error(log_message)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(log_message)
        else:
            logger.info(log_message)

        return HandlerResult(
            success=True,
            handler_name=self.name,
            message="Alert logged to console",
        )


class SlackHandler(AlertHandler):
    """Slack webhook handler for sending alerts to Slack channels.

    Example:
        >>> handler = SlackHandler(
        ...     webhook_url="https://hooks.slack.com/services/...",
        ...     channel="#ml-alerts",
        ... )
        >>> await handler.send(alert)
    """

    def __init__(
        self,
        webhook_url: str,
        channel: str | None = None,
        username: str = "ML Monitor",
        icon_emoji: str = ":robot_face:",
        enabled: bool = True,
    ) -> None:
        """Initialize the Slack handler.

        Args:
            webhook_url: Slack webhook URL.
            channel: Optional channel override.
            username: Bot username.
            icon_emoji: Bot icon emoji.
            enabled: Whether the handler is enabled.
        """
        super().__init__("slack", enabled)
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji

    async def send(self, alert: Alert) -> HandlerResult:
        """Send alert to Slack.

        Args:
            alert: Alert to send.

        Returns:
            HandlerResult indicating success or failure.
        """
        if not self.enabled:
            return HandlerResult(
                success=False,
                handler_name=self.name,
                message="Handler is disabled",
            )

        payload = self._build_payload(alert)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 200:
                        return HandlerResult(
                            success=True,
                            handler_name=self.name,
                            message="Alert sent to Slack",
                            details={"status_code": response.status},
                        )
                    else:
                        error_text = await response.text()
                        return HandlerResult(
                            success=False,
                            handler_name=self.name,
                            message=f"Slack API error: {response.status}",
                            details={"status_code": response.status, "error": error_text},
                        )
        except TimeoutError:
            return HandlerResult(
                success=False,
                handler_name=self.name,
                message="Slack webhook timeout",
            )
        except Exception as e:
            logger.exception(f"Failed to send Slack alert: {e}")
            return HandlerResult(
                success=False,
                handler_name=self.name,
                message=f"Failed to send: {e!s}",
            )

    def _build_payload(self, alert: Alert) -> dict[str, Any]:
        """Build Slack message payload.

        Args:
            alert: Alert to format.

        Returns:
            Slack message payload.
        """
        severity_colors = {
            AlertSeverity.INFO: "#36a64f",  # Green
            AlertSeverity.WARNING: "#ff9800",  # Orange
            AlertSeverity.CRITICAL: "#f44336",  # Red
        }

        severity_emoji = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.CRITICAL: ":rotating_light:",
        }

        attachment = {
            "color": severity_colors.get(alert.severity, "#808080"),
            "title": f"{severity_emoji.get(alert.severity, '')} {alert.name}",
            "text": alert.message,
            "fields": [
                {"title": "Model", "value": alert.model_name, "short": True},
                {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                {"title": "Metric", "value": alert.metric_name, "short": True},
                {"title": "Value", "value": f"{alert.metric_value:.4f}", "short": True},
                {"title": "Threshold", "value": f"{alert.threshold:.4f}", "short": True},
                {"title": "Status", "value": alert.status.value.upper(), "short": True},
            ],
            "footer": "ML Monitoring System",
            "ts": int(alert.timestamp.timestamp()),
        }

        payload: dict[str, Any] = {
            "username": self.username,
            "icon_emoji": self.icon_emoji,
            "attachments": [attachment],
        }

        if self.channel:
            payload["channel"] = self.channel

        return payload


class EmailHandler(AlertHandler):
    """Email handler using SMTP.

    Example:
        >>> handler = EmailHandler(
        ...     smtp_host="smtp.gmail.com",
        ...     smtp_port=587,
        ...     username="alerts@example.com",
        ...     password="app_password",
        ...     from_email="alerts@example.com",
        ...     to_emails=["team@example.com"],
        ... )
        >>> await handler.send(alert)
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_email: str,
        to_emails: list[str],
        use_tls: bool = True,
        enabled: bool = True,
    ) -> None:
        """Initialize the email handler.

        Args:
            smtp_host: SMTP server host.
            smtp_port: SMTP server port.
            username: SMTP username.
            password: SMTP password.
            from_email: Sender email address.
            to_emails: List of recipient email addresses.
            use_tls: Whether to use TLS.
            enabled: Whether the handler is enabled.
        """
        super().__init__("email", enabled)
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_email = from_email
        self.to_emails = to_emails
        self.use_tls = use_tls

    async def send(self, alert: Alert) -> HandlerResult:
        """Send alert via email.

        Args:
            alert: Alert to send.

        Returns:
            HandlerResult indicating success or failure.
        """
        if not self.enabled:
            return HandlerResult(
                success=False,
                handler_name=self.name,
                message="Handler is disabled",
            )

        try:
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            import aiosmtplib

            msg = MIMEMultipart("alternative")
            msg["Subject"] = self._build_subject(alert)
            msg["From"] = self.from_email
            msg["To"] = ", ".join(self.to_emails)

            # Plain text version
            text_content = self._build_text_content(alert)
            msg.attach(MIMEText(text_content, "plain"))

            # HTML version
            html_content = self._build_html_content(alert)
            msg.attach(MIMEText(html_content, "html"))

            await aiosmtplib.send(
                msg,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.username,
                password=self.password,
                start_tls=self.use_tls,
            )

            return HandlerResult(
                success=True,
                handler_name=self.name,
                message=f"Alert email sent to {len(self.to_emails)} recipients",
            )

        except ImportError:
            return HandlerResult(
                success=False,
                handler_name=self.name,
                message="aiosmtplib not installed. Install with: pip install aiosmtplib",
            )
        except Exception as e:
            logger.exception(f"Failed to send email alert: {e}")
            return HandlerResult(
                success=False,
                handler_name=self.name,
                message=f"Failed to send email: {e!s}",
            )

    def _build_subject(self, alert: Alert) -> str:
        """Build email subject.

        Args:
            alert: Alert.

        Returns:
            Email subject string.
        """
        severity_prefix = {
            AlertSeverity.INFO: "[INFO]",
            AlertSeverity.WARNING: "[WARNING]",
            AlertSeverity.CRITICAL: "[CRITICAL]",
        }
        prefix = severity_prefix.get(alert.severity, "[ALERT]")
        return f"{prefix} {alert.name} - {alert.model_name}"

    def _build_text_content(self, alert: Alert) -> str:
        """Build plain text email content.

        Args:
            alert: Alert.

        Returns:
            Plain text content.
        """
        return f"""
ML Model Alert

Alert: {alert.name}
Severity: {alert.severity.value.upper()}
Status: {alert.status.value.upper()}

Model: {alert.model_name}
Metric: {alert.metric_name}
Value: {alert.metric_value:.4f}
Threshold: {alert.threshold:.4f}

Message: {alert.message}

Timestamp: {alert.timestamp.isoformat()}

---
ML Monitoring System
"""

    def _build_html_content(self, alert: Alert) -> str:
        """Build HTML email content.

        Args:
            alert: Alert.

        Returns:
            HTML content.
        """
        severity_colors = {
            AlertSeverity.INFO: "#28a745",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.CRITICAL: "#dc3545",
        }
        color = severity_colors.get(alert.severity, "#6c757d")

        return f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; }}
        .alert-box {{ border: 2px solid {color}; border-radius: 8px; padding: 20px; max-width: 600px; }}
        .header {{ background-color: {color}; color: white; padding: 10px; border-radius: 4px; }}
        .metric {{ background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 4px; }}
        .footer {{ color: #6c757d; font-size: 12px; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="alert-box">
        <div class="header">
            <h2>{alert.name}</h2>
            <span>{alert.severity.value.upper()}</span>
        </div>
        <p><strong>Message:</strong> {alert.message}</p>
        <div class="metric">
            <p><strong>Model:</strong> {alert.model_name}</p>
            <p><strong>Metric:</strong> {alert.metric_name}</p>
            <p><strong>Current Value:</strong> {alert.metric_value:.4f}</p>
            <p><strong>Threshold:</strong> {alert.threshold:.4f}</p>
        </div>
        <p class="footer">Timestamp: {alert.timestamp.isoformat()}<br>ML Monitoring System</p>
    </div>
</body>
</html>
"""


class WebhookHandler(AlertHandler):
    """Generic webhook handler for HTTP POST notifications.

    Example:
        >>> handler = WebhookHandler(
        ...     url="https://api.example.com/alerts",
        ...     headers={"Authorization": "Bearer token"},
        ... )
        >>> await handler.send(alert)
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 10,
        enabled: bool = True,
    ) -> None:
        """Initialize the webhook handler.

        Args:
            url: Webhook URL.
            headers: Optional HTTP headers.
            timeout: Request timeout in seconds.
            enabled: Whether the handler is enabled.
        """
        super().__init__("webhook", enabled)
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout

    async def send(self, alert: Alert) -> HandlerResult:
        """Send alert via webhook.

        Args:
            alert: Alert to send.

        Returns:
            HandlerResult indicating success or failure.
        """
        if not self.enabled:
            return HandlerResult(
                success=False,
                handler_name=self.name,
                message="Handler is disabled",
            )

        payload = alert.to_dict()

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        **self.headers,
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    response_text = await response.text()
                    success = 200 <= response.status < 300

                    return HandlerResult(
                        success=success,
                        handler_name=self.name,
                        message=f"Webhook {'succeeded' if success else 'failed'}",
                        details={
                            "status_code": response.status,
                            "response": response_text[:200],
                        },
                    )
        except TimeoutError:
            return HandlerResult(
                success=False,
                handler_name=self.name,
                message="Webhook timeout",
            )
        except Exception as e:
            logger.exception(f"Failed to send webhook: {e}")
            return HandlerResult(
                success=False,
                handler_name=self.name,
                message=f"Failed to send: {e!s}",
            )


class AlertManagerHandler(AlertHandler):
    """Prometheus AlertManager integration.

    Example:
        >>> handler = AlertManagerHandler(
        ...     alertmanager_url="http://alertmanager:9093",
        ... )
        >>> await handler.send(alert)
    """

    def __init__(
        self,
        alertmanager_url: str,
        timeout: int = 10,
        enabled: bool = True,
    ) -> None:
        """Initialize the AlertManager handler.

        Args:
            alertmanager_url: AlertManager base URL.
            timeout: Request timeout in seconds.
            enabled: Whether the handler is enabled.
        """
        super().__init__("alertmanager", enabled)
        self.alertmanager_url = alertmanager_url.rstrip("/")
        self.timeout = timeout

    async def send(self, alert: Alert) -> HandlerResult:
        """Send alert to AlertManager.

        Args:
            alert: Alert to send.

        Returns:
            HandlerResult indicating success or failure.
        """
        if not self.enabled:
            return HandlerResult(
                success=False,
                handler_name=self.name,
                message="Handler is disabled",
            )

        # Convert to AlertManager format
        am_alert = self._to_alertmanager_format(alert)

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.alertmanager_url}/api/v2/alerts",
                    json=[am_alert],
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    success = response.status == 200

                    return HandlerResult(
                        success=success,
                        handler_name=self.name,
                        message=f"AlertManager {'accepted' if success else 'rejected'} alert",
                        details={"status_code": response.status},
                    )
        except Exception as e:
            logger.exception(f"Failed to send to AlertManager: {e}")
            return HandlerResult(
                success=False,
                handler_name=self.name,
                message=f"Failed to send: {e!s}",
            )

    def _to_alertmanager_format(self, alert: Alert) -> dict[str, Any]:
        """Convert alert to AlertManager format.

        Args:
            alert: Alert to convert.

        Returns:
            AlertManager-compatible alert dictionary.
        """
        return {
            "labels": {
                "alertname": alert.name,
                "severity": alert.severity.value,
                "model": alert.model_name,
                "metric": alert.metric_name,
                **alert.labels,
            },
            "annotations": {
                "summary": alert.message,
                "description": alert.annotations.get("description", ""),
                "value": str(alert.metric_value),
                "threshold": str(alert.threshold),
            },
            "startsAt": alert.timestamp.isoformat(),
            "generatorURL": f"http://ml-service/alerts/{alert.name}",
        }

    async def send_batch(self, alerts: list[Alert]) -> list[HandlerResult]:
        """Send multiple alerts to AlertManager in one request.

        Args:
            alerts: List of alerts to send.

        Returns:
            List of HandlerResults.
        """
        if not self.enabled or not alerts:
            return [
                HandlerResult(
                    success=False,
                    handler_name=self.name,
                    message="Handler disabled or no alerts",
                )
            ]

        am_alerts = [self._to_alertmanager_format(a) for a in alerts]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.alertmanager_url}/api/v2/alerts",
                    json=am_alerts,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as response:
                    success = response.status == 200
                    return [
                        HandlerResult(
                            success=success,
                            handler_name=self.name,
                            message=f"Batch of {len(alerts)} alerts {'sent' if success else 'failed'}",
                            details={"status_code": response.status, "count": len(alerts)},
                        )
                    ]
        except Exception as e:
            logger.exception(f"Failed to send batch to AlertManager: {e}")
            return [
                HandlerResult(
                    success=False,
                    handler_name=self.name,
                    message=f"Batch send failed: {e!s}",
                )
            ]


class AlertDispatcher:
    """Dispatch alerts to multiple handlers.

    Example:
        >>> dispatcher = AlertDispatcher()
        >>> dispatcher.add_handler(SlackHandler(webhook_url="..."))
        >>> dispatcher.add_handler(ConsoleHandler())
        >>> results = await dispatcher.dispatch(alert)
    """

    def __init__(self) -> None:
        """Initialize the dispatcher."""
        self._handlers: list[AlertHandler] = []

    def add_handler(self, handler: AlertHandler) -> None:
        """Add a handler to the dispatcher.

        Args:
            handler: Handler to add.
        """
        self._handlers.append(handler)
        logger.info(f"Added alert handler: {handler.name}")

    def remove_handler(self, handler_name: str) -> bool:
        """Remove a handler by name.

        Args:
            handler_name: Name of the handler to remove.

        Returns:
            True if handler was removed, False if not found.
        """
        for i, handler in enumerate(self._handlers):
            if handler.name == handler_name:
                self._handlers.pop(i)
                logger.info(f"Removed alert handler: {handler_name}")
                return True
        return False

    async def dispatch(self, alert: Alert) -> list[HandlerResult]:
        """Dispatch alert to all handlers.

        Args:
            alert: Alert to dispatch.

        Returns:
            List of HandlerResults from all handlers.
        """
        if not self._handlers:
            logger.warning("No handlers configured for alert dispatch")
            return []

        # Run all handlers concurrently
        tasks = [handler.send(alert) for handler in self._handlers if handler.enabled]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to HandlerResults
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(
                    HandlerResult(
                        success=False,
                        handler_name=self._handlers[i].name,
                        message=f"Handler error: {result!s}",
                    )
                )
            else:
                processed_results.append(result)

        return processed_results

    async def dispatch_batch(self, alerts: list[Alert]) -> dict[str, list[HandlerResult]]:
        """Dispatch multiple alerts to all handlers.

        Args:
            alerts: List of alerts to dispatch.

        Returns:
            Dictionary mapping alert names to their results.
        """
        results: dict[str, list[HandlerResult]] = {}
        for alert in alerts:
            results[alert.name] = await self.dispatch(alert)
        return results

    def list_handlers(self) -> list[dict[str, Any]]:
        """List all registered handlers.

        Returns:
            List of handler information dictionaries.
        """
        return [
            {
                "name": handler.name,
                "enabled": handler.enabled,
                "type": handler.__class__.__name__,
            }
            for handler in self._handlers
        ]
