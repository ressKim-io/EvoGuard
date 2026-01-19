"""Tests for alert handlers."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
from ml_service.monitoring.alerts.rules import Alert, AlertSeverity, AlertStatus


@pytest.fixture
def sample_alert() -> Alert:
    """Create a sample alert for testing."""
    return Alert(
        name="TestAlert",
        severity=AlertSeverity.WARNING,
        status=AlertStatus.FIRING,
        message="Test alert message",
        model_name="test_model",
        metric_name="f1_score",
        metric_value=0.65,
        threshold=0.7,
        labels={"env": "test"},
        annotations={"description": "Test description"},
    )


@pytest.fixture
def critical_alert() -> Alert:
    """Create a critical alert for testing."""
    return Alert(
        name="CriticalAlert",
        severity=AlertSeverity.CRITICAL,
        status=AlertStatus.FIRING,
        message="Critical alert message",
        model_name="test_model",
        metric_name="f1_score",
        metric_value=0.5,
        threshold=0.7,
    )


class TestConsoleHandler:
    """Tests for ConsoleHandler."""

    @pytest.mark.asyncio
    async def test_console_handler_sends_alert(self, sample_alert: Alert) -> None:
        """Test console handler logs alert successfully."""
        handler = ConsoleHandler()
        result = await handler.send(sample_alert)

        assert result.success
        assert result.handler_name == "console"

    @pytest.mark.asyncio
    async def test_console_handler_disabled(self, sample_alert: Alert) -> None:
        """Test disabled console handler does not send."""
        handler = ConsoleHandler(enabled=False)
        result = await handler.send(sample_alert)

        assert not result.success
        assert "disabled" in result.message.lower()

    @pytest.mark.asyncio
    async def test_console_handler_critical_severity(self, critical_alert: Alert) -> None:
        """Test console handler handles critical alerts."""
        handler = ConsoleHandler()
        result = await handler.send(critical_alert)

        assert result.success


class TestSlackHandler:
    """Tests for SlackHandler."""

    @pytest.mark.asyncio
    async def test_slack_handler_disabled(self, sample_alert: Alert) -> None:
        """Test disabled Slack handler does not send."""
        handler = SlackHandler(
            webhook_url="https://hooks.slack.com/test",
            enabled=False,
        )
        result = await handler.send(sample_alert)

        assert not result.success
        assert "disabled" in result.message.lower()

    @pytest.mark.asyncio
    async def test_slack_handler_builds_payload(self, sample_alert: Alert) -> None:
        """Test Slack handler builds correct payload."""
        handler = SlackHandler(
            webhook_url="https://hooks.slack.com/test",
            channel="#test",
            username="TestBot",
            icon_emoji=":robot:",
        )

        payload = handler._build_payload(sample_alert)

        assert payload["username"] == "TestBot"
        assert payload["icon_emoji"] == ":robot:"
        assert payload["channel"] == "#test"
        assert len(payload["attachments"]) == 1
        assert payload["attachments"][0]["color"] == "#ff9800"  # Warning color

    @pytest.mark.asyncio
    async def test_slack_handler_critical_color(self, critical_alert: Alert) -> None:
        """Test Slack handler uses correct color for critical alerts."""
        handler = SlackHandler(webhook_url="https://hooks.slack.com/test")

        payload = handler._build_payload(critical_alert)

        assert payload["attachments"][0]["color"] == "#f44336"  # Critical color

    @pytest.mark.asyncio
    async def test_slack_handler_success(self, sample_alert: Alert) -> None:
        """Test Slack handler sends successfully."""
        handler = SlackHandler(webhook_url="https://hooks.slack.com/test")

        with patch("ml_service.monitoring.alerts.handlers.aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value="ok")

            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            result = await handler.send(sample_alert)

            assert result.success

    @pytest.mark.asyncio
    async def test_slack_handler_timeout(self, sample_alert: Alert) -> None:
        """Test Slack handler handles timeout."""
        handler = SlackHandler(webhook_url="https://hooks.slack.com/test")

        with patch("ml_service.monitoring.alerts.handlers.aiohttp.ClientSession") as mock_session_class:
            mock_session = MagicMock()
            mock_session.post = MagicMock(side_effect=TimeoutError())
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            result = await handler.send(sample_alert)

            assert not result.success
            assert "timeout" in result.message.lower()


class TestWebhookHandler:
    """Tests for WebhookHandler."""

    @pytest.mark.asyncio
    async def test_webhook_handler_disabled(self, sample_alert: Alert) -> None:
        """Test disabled webhook handler does not send."""
        handler = WebhookHandler(
            url="https://api.example.com/alerts",
            enabled=False,
        )
        result = await handler.send(sample_alert)

        assert not result.success
        assert "disabled" in result.message.lower()

    @pytest.mark.asyncio
    async def test_webhook_handler_success(self, sample_alert: Alert) -> None:
        """Test webhook handler sends successfully."""
        handler = WebhookHandler(
            url="https://api.example.com/alerts",
            headers={"Authorization": "Bearer token"},
        )

        with patch("ml_service.monitoring.alerts.handlers.aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='{"status": "ok"}')

            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            result = await handler.send(sample_alert)

            assert result.success

    @pytest.mark.asyncio
    async def test_webhook_handler_failure(self, sample_alert: Alert) -> None:
        """Test webhook handler handles failure response."""
        handler = WebhookHandler(url="https://api.example.com/alerts")

        with patch("ml_service.monitoring.alerts.handlers.aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 500
            mock_response.text = AsyncMock(return_value="Internal Server Error")

            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            result = await handler.send(sample_alert)

            assert not result.success


class TestEmailHandler:
    """Tests for EmailHandler."""

    def test_email_handler_builds_subject(self, sample_alert: Alert) -> None:
        """Test email handler builds correct subject."""
        handler = EmailHandler(
            smtp_host="smtp.example.com",
            smtp_port=587,
            username="user",
            password="pass",
            from_email="alerts@example.com",
            to_emails=["team@example.com"],
        )

        subject = handler._build_subject(sample_alert)

        assert "[WARNING]" in subject
        assert "TestAlert" in subject
        assert "test_model" in subject

    def test_email_handler_critical_subject(self, critical_alert: Alert) -> None:
        """Test email handler builds correct subject for critical alerts."""
        handler = EmailHandler(
            smtp_host="smtp.example.com",
            smtp_port=587,
            username="user",
            password="pass",
            from_email="alerts@example.com",
            to_emails=["team@example.com"],
        )

        subject = handler._build_subject(critical_alert)

        assert "[CRITICAL]" in subject

    def test_email_handler_builds_text_content(self, sample_alert: Alert) -> None:
        """Test email handler builds text content."""
        handler = EmailHandler(
            smtp_host="smtp.example.com",
            smtp_port=587,
            username="user",
            password="pass",
            from_email="alerts@example.com",
            to_emails=["team@example.com"],
        )

        content = handler._build_text_content(sample_alert)

        assert "TestAlert" in content
        assert "WARNING" in content
        assert "test_model" in content
        assert "f1_score" in content

    def test_email_handler_builds_html_content(self, sample_alert: Alert) -> None:
        """Test email handler builds HTML content."""
        handler = EmailHandler(
            smtp_host="smtp.example.com",
            smtp_port=587,
            username="user",
            password="pass",
            from_email="alerts@example.com",
            to_emails=["team@example.com"],
        )

        content = handler._build_html_content(sample_alert)

        assert "<html>" in content
        assert "TestAlert" in content
        assert "#ffc107" in content  # Warning color

    @pytest.mark.asyncio
    async def test_email_handler_disabled(self, sample_alert: Alert) -> None:
        """Test disabled email handler does not send."""
        handler = EmailHandler(
            smtp_host="smtp.example.com",
            smtp_port=587,
            username="user",
            password="pass",
            from_email="alerts@example.com",
            to_emails=["team@example.com"],
            enabled=False,
        )

        result = await handler.send(sample_alert)

        assert not result.success


class TestAlertManagerHandler:
    """Tests for AlertManagerHandler."""

    @pytest.mark.asyncio
    async def test_alertmanager_handler_disabled(self, sample_alert: Alert) -> None:
        """Test disabled AlertManager handler does not send."""
        handler = AlertManagerHandler(
            alertmanager_url="http://alertmanager:9093",
            enabled=False,
        )
        result = await handler.send(sample_alert)

        assert not result.success

    def test_alertmanager_format_conversion(self, sample_alert: Alert) -> None:
        """Test AlertManager format conversion."""
        handler = AlertManagerHandler(alertmanager_url="http://alertmanager:9093")

        am_alert = handler._to_alertmanager_format(sample_alert)

        assert am_alert["labels"]["alertname"] == "TestAlert"
        assert am_alert["labels"]["severity"] == "warning"
        assert am_alert["labels"]["model"] == "test_model"
        assert am_alert["annotations"]["summary"] == "Test alert message"

    @pytest.mark.asyncio
    async def test_alertmanager_handler_success(self, sample_alert: Alert) -> None:
        """Test AlertManager handler sends successfully."""
        handler = AlertManagerHandler(alertmanager_url="http://alertmanager:9093")

        with patch("ml_service.monitoring.alerts.handlers.aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200

            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            result = await handler.send(sample_alert)

            assert result.success

    @pytest.mark.asyncio
    async def test_alertmanager_batch_send(self, sample_alert: Alert, critical_alert: Alert) -> None:
        """Test AlertManager batch send."""
        handler = AlertManagerHandler(alertmanager_url="http://alertmanager:9093")

        with patch("ml_service.monitoring.alerts.handlers.aiohttp.ClientSession") as mock_session_class:
            mock_response = AsyncMock()
            mock_response.status = 200

            mock_post = MagicMock()
            mock_post.__aenter__ = AsyncMock(return_value=mock_response)
            mock_post.__aexit__ = AsyncMock(return_value=None)

            mock_session = MagicMock()
            mock_session.post = MagicMock(return_value=mock_post)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)

            mock_session_class.return_value = mock_session

            results = await handler.send_batch([sample_alert, critical_alert])

            assert len(results) == 1
            assert results[0].success


class TestAlertDispatcher:
    """Tests for AlertDispatcher."""

    @pytest.mark.asyncio
    async def test_dispatcher_add_handler(self) -> None:
        """Test adding handlers to dispatcher."""
        dispatcher = AlertDispatcher()
        handler = ConsoleHandler()

        dispatcher.add_handler(handler)

        handlers_list = dispatcher.list_handlers()
        assert len(handlers_list) == 1
        assert handlers_list[0]["name"] == "console"

    @pytest.mark.asyncio
    async def test_dispatcher_remove_handler(self) -> None:
        """Test removing handlers from dispatcher."""
        dispatcher = AlertDispatcher()
        handler = ConsoleHandler()

        dispatcher.add_handler(handler)
        assert dispatcher.remove_handler("console")

        handlers_list = dispatcher.list_handlers()
        assert len(handlers_list) == 0

    @pytest.mark.asyncio
    async def test_dispatcher_remove_nonexistent_handler(self) -> None:
        """Test removing nonexistent handler returns False."""
        dispatcher = AlertDispatcher()
        assert not dispatcher.remove_handler("nonexistent")

    @pytest.mark.asyncio
    async def test_dispatcher_dispatch_to_all_handlers(self, sample_alert: Alert) -> None:
        """Test dispatcher sends to all handlers."""
        dispatcher = AlertDispatcher()
        dispatcher.add_handler(ConsoleHandler())
        dispatcher.add_handler(ConsoleHandler())

        results = await dispatcher.dispatch(sample_alert)

        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_dispatcher_skips_disabled_handlers(self, sample_alert: Alert) -> None:
        """Test dispatcher skips disabled handlers."""
        dispatcher = AlertDispatcher()
        dispatcher.add_handler(ConsoleHandler(enabled=True))
        dispatcher.add_handler(ConsoleHandler(enabled=False))

        results = await dispatcher.dispatch(sample_alert)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_dispatcher_no_handlers(self, sample_alert: Alert) -> None:
        """Test dispatcher with no handlers returns empty list."""
        dispatcher = AlertDispatcher()

        results = await dispatcher.dispatch(sample_alert)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_dispatcher_handles_handler_exception(self, sample_alert: Alert) -> None:
        """Test dispatcher handles exceptions from handlers."""
        dispatcher = AlertDispatcher()

        # Create a mock handler that raises an exception
        mock_handler = MagicMock(spec=AlertHandler)
        mock_handler.name = "failing_handler"
        mock_handler.enabled = True
        mock_handler.send = AsyncMock(side_effect=Exception("Handler error"))

        dispatcher.add_handler(mock_handler)

        results = await dispatcher.dispatch(sample_alert)

        assert len(results) == 1
        assert not results[0].success
        assert "error" in results[0].message.lower()

    @pytest.mark.asyncio
    async def test_dispatcher_batch_dispatch(self, sample_alert: Alert, critical_alert: Alert) -> None:
        """Test dispatcher batch dispatch."""
        dispatcher = AlertDispatcher()
        dispatcher.add_handler(ConsoleHandler())

        results = await dispatcher.dispatch_batch([sample_alert, critical_alert])

        assert "TestAlert" in results
        assert "CriticalAlert" in results
        assert all(r.success for r in results["TestAlert"])
        assert all(r.success for r in results["CriticalAlert"])

    @pytest.mark.asyncio
    async def test_dispatcher_list_handlers(self) -> None:
        """Test listing handlers with their status."""
        dispatcher = AlertDispatcher()
        dispatcher.add_handler(ConsoleHandler(enabled=True))
        dispatcher.add_handler(SlackHandler(webhook_url="https://test", enabled=False))

        handlers_list = dispatcher.list_handlers()

        assert len(handlers_list) == 2
        console = next(h for h in handlers_list if h["name"] == "console")
        slack = next(h for h in handlers_list if h["name"] == "slack")

        assert console["enabled"]
        assert not slack["enabled"]
        assert console["type"] == "ConsoleHandler"
        assert slack["type"] == "SlackHandler"


class TestHandlerResult:
    """Tests for HandlerResult."""

    def test_handler_result_creation(self) -> None:
        """Test HandlerResult creation."""
        result = HandlerResult(
            success=True,
            handler_name="test_handler",
            message="Success message",
            details={"key": "value"},
        )

        assert result.success
        assert result.handler_name == "test_handler"
        assert result.message == "Success message"
        assert result.details == {"key": "value"}
        assert result.timestamp is not None

    def test_handler_result_default_details(self) -> None:
        """Test HandlerResult with default details."""
        result = HandlerResult(
            success=True,
            handler_name="test_handler",
            message="Success message",
        )

        assert result.details == {}
