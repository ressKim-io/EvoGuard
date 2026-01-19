"""Tests for AutoRetrainTrigger and ScheduledRetrainTrigger."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from ml_service.monitoring.actions.retrain_trigger import (
    AutoRetrainTrigger,
    MockRetrainExecutor,
    RetrainDecision,
    RetrainJob,
    RetrainOrchestrator,
    RetrainPriority,
    RetrainReason,
    ScheduledRetrainTrigger,
)


class TestRetrainPriority:
    """Tests for RetrainPriority enum."""

    def test_priority_values(self) -> None:
        """Test priority enum values."""
        assert RetrainPriority.LOW.value == "low"
        assert RetrainPriority.NORMAL.value == "normal"
        assert RetrainPriority.HIGH.value == "high"
        assert RetrainPriority.URGENT.value == "urgent"


class TestRetrainReason:
    """Tests for RetrainReason enum."""

    def test_reason_values(self) -> None:
        """Test reason enum values."""
        assert RetrainReason.DATA_DRIFT.value == "data_drift"
        assert RetrainReason.CONCEPT_DRIFT.value == "concept_drift"
        assert RetrainReason.FEATURE_DRIFT.value == "feature_drift"
        assert RetrainReason.PERFORMANCE_DEGRADATION.value == "performance_degradation"
        assert RetrainReason.SCHEDULED.value == "scheduled"
        assert RetrainReason.MANUAL.value == "manual"


class TestRetrainDecision:
    """Tests for RetrainDecision dataclass."""

    def test_decision_creation(self) -> None:
        """Test creating a retrain decision."""
        decision = RetrainDecision(
            should_retrain=True,
            reasons=["data_drift_psi=0.300"],
            priority=RetrainPriority.HIGH,
            metrics={"data_drift_psi": 0.3},
        )
        assert decision.should_retrain is True
        assert len(decision.reasons) == 1
        assert decision.priority == RetrainPriority.HIGH
        assert decision.metrics == {"data_drift_psi": 0.3}

    def test_decision_no_retrain(self) -> None:
        """Test decision with no retrain."""
        decision = RetrainDecision(
            should_retrain=False,
            reasons=[],
            priority=RetrainPriority.NORMAL,
            metrics={},
        )
        assert decision.should_retrain is False
        assert len(decision.reasons) == 0

    def test_decision_to_dict(self) -> None:
        """Test converting decision to dictionary."""
        decision = RetrainDecision(
            should_retrain=True,
            reasons=["test"],
            priority=RetrainPriority.HIGH,
            metrics={"test": 1.0},
        )
        result = decision.to_dict()
        assert result["should_retrain"] is True
        assert result["reasons"] == ["test"]
        assert result["priority"] == "high"


class TestAutoRetrainTrigger:
    """Tests for AutoRetrainTrigger."""

    def test_init_default_thresholds(self) -> None:
        """Test initialization with default thresholds."""
        trigger = AutoRetrainTrigger(model_name="test_model")
        assert trigger.model_name == "test_model"
        assert trigger.data_drift_threshold == 0.25
        assert trigger.concept_drift_threshold == 0.10
        assert trigger.feature_drift_threshold == 0.25
        assert trigger.f1_critical_threshold == 0.65

    def test_init_custom_thresholds(self) -> None:
        """Test initialization with custom thresholds."""
        trigger = AutoRetrainTrigger(
            model_name="test_model",
            data_drift_threshold=0.3,
            concept_drift_threshold=0.15,
            f1_critical_threshold=0.7,
        )
        assert trigger.data_drift_threshold == 0.3
        assert trigger.concept_drift_threshold == 0.15
        assert trigger.f1_critical_threshold == 0.7

    def test_evaluate_no_drift(self) -> None:
        """Test evaluation with no drift detected."""
        trigger = AutoRetrainTrigger(model_name="test_model")
        metrics = {
            "data_drift_psi": 0.05,
            "f1_drop": 0.02,
            "feature_drift_psi": 0.1,
            "current_f1": 0.9,
        }
        decision = trigger.evaluate(metrics)
        assert decision.should_retrain is False
        assert len(decision.reasons) == 0

    def test_evaluate_data_drift_detected(self) -> None:
        """Test evaluation with data drift detected."""
        trigger = AutoRetrainTrigger(model_name="test_model")
        metrics = {
            "data_drift_psi": 0.35,  # Above threshold 0.25
            "f1_drop": 0.02,
            "feature_drift_psi": 0.1,
            "current_f1": 0.85,
        }
        decision = trigger.evaluate(metrics)
        assert decision.should_retrain is True
        assert any("data_drift_psi" in r for r in decision.reasons)

    def test_evaluate_concept_drift_detected(self) -> None:
        """Test evaluation with concept drift (F1 drop) detected."""
        trigger = AutoRetrainTrigger(model_name="test_model")
        metrics = {
            "data_drift_psi": 0.05,
            "f1_drop": 0.15,  # Above threshold 0.10
            "feature_drift_psi": 0.1,
            "current_f1": 0.85,
        }
        decision = trigger.evaluate(metrics)
        assert decision.should_retrain is True
        assert any("f1_drop" in r for r in decision.reasons)

    def test_evaluate_feature_drift_detected(self) -> None:
        """Test evaluation with feature drift detected."""
        trigger = AutoRetrainTrigger(model_name="test_model")
        metrics = {
            "data_drift_psi": 0.05,
            "f1_drop": 0.02,
            "feature_drift_psi": 0.35,  # Above threshold 0.25
            "current_f1": 0.85,
        }
        decision = trigger.evaluate(metrics)
        assert decision.should_retrain is True
        assert any("feature_drift_psi" in r for r in decision.reasons)

    def test_evaluate_critical_f1(self) -> None:
        """Test evaluation with critical F1 score."""
        trigger = AutoRetrainTrigger(model_name="test_model")
        metrics = {
            "data_drift_psi": 0.05,
            "f1_drop": 0.02,
            "feature_drift_psi": 0.1,
            "current_f1": 0.6,  # Below critical threshold 0.65
        }
        decision = trigger.evaluate(metrics)
        assert decision.should_retrain is True
        assert any("critical_f1" in r for r in decision.reasons)
        assert decision.priority == RetrainPriority.URGENT

    def test_evaluate_respects_cooldown(self) -> None:
        """Test that cooldown period is respected."""
        trigger = AutoRetrainTrigger(
            model_name="test_model",
            min_interval_hours=24,
        )
        metrics = {"data_drift_psi": 0.35, "current_f1": 0.85}

        # Record a retrain
        trigger.record_retrain()

        # Next evaluation should be blocked by cooldown
        decision = trigger.evaluate(metrics)
        assert decision.should_retrain is False
        assert "cooldown_active" in decision.reasons

    def test_evaluate_respects_daily_limit(self) -> None:
        """Test that daily retrain limit is respected."""
        trigger = AutoRetrainTrigger(
            model_name="test_model",
            min_interval_hours=0,  # No cooldown
            max_retrains_per_day=2,
        )
        metrics = {"data_drift_psi": 0.35, "current_f1": 0.85}

        # Record two retrains
        trigger.record_retrain()
        trigger.record_retrain()

        # Third should be blocked
        decision = trigger.evaluate(metrics)
        assert decision.should_retrain is False
        assert "daily_limit_exceeded" in decision.reasons

    def test_evaluate_disabled(self) -> None:
        """Test evaluation when trigger is disabled."""
        trigger = AutoRetrainTrigger(model_name="test_model")
        trigger.disable()

        metrics = {"data_drift_psi": 0.35, "current_f1": 0.85}
        decision = trigger.evaluate(metrics)
        assert decision.should_retrain is False
        assert "trigger_disabled" in decision.reasons

    def test_enable_disable(self) -> None:
        """Test enable/disable functionality."""
        trigger = AutoRetrainTrigger(model_name="test_model")
        assert trigger.enabled is True

        trigger.disable()
        assert trigger.enabled is False

        trigger.enable()
        assert trigger.enabled is True

    def test_record_retrain(self) -> None:
        """Test recording retrain events."""
        trigger = AutoRetrainTrigger(model_name="test_model")
        assert trigger._last_retrain is None

        trigger.record_retrain()
        assert trigger._last_retrain is not None
        assert len(trigger._retrain_history) == 1

    def test_reset(self) -> None:
        """Test reset clears state."""
        trigger = AutoRetrainTrigger(model_name="test_model")
        trigger.record_retrain()
        trigger.record_retrain()

        trigger.reset()
        assert trigger._last_retrain is None
        assert len(trigger._retrain_history) == 0

    def test_get_status(self) -> None:
        """Test get_status returns correct info."""
        trigger = AutoRetrainTrigger(model_name="test_model")
        status = trigger.get_status()

        assert status["model_name"] == "test_model"
        assert status["enabled"] is True
        assert "thresholds" in status
        assert "cooldown" in status
        assert "daily_limit" in status


class TestScheduledRetrainTrigger:
    """Tests for ScheduledRetrainTrigger."""

    def test_init(self) -> None:
        """Test initialization."""
        trigger = ScheduledRetrainTrigger(
            model_name="test_model",
            interval_hours=24,
        )
        assert trigger.model_name == "test_model"
        assert trigger.interval == timedelta(hours=24)

    def test_is_due_first_time(self) -> None:
        """Test is_due returns True on first call."""
        trigger = ScheduledRetrainTrigger(
            model_name="test_model",
            interval_hours=24,
        )
        assert trigger.is_due() is True

    def test_is_due_after_execution(self) -> None:
        """Test is_due returns False right after execution."""
        trigger = ScheduledRetrainTrigger(
            model_name="test_model",
            interval_hours=24,
        )
        trigger.record_retrain()
        assert trigger.is_due() is False

    def test_is_due_after_interval(self) -> None:
        """Test is_due returns True after interval passed."""
        trigger = ScheduledRetrainTrigger(
            model_name="test_model",
            interval_hours=1,
        )
        # Set last retrain to 2 hours ago
        trigger._last_retrain = datetime.now(UTC) - timedelta(hours=2)
        assert trigger.is_due() is True

    def test_evaluate_when_due(self) -> None:
        """Test evaluate when retrain is due."""
        trigger = ScheduledRetrainTrigger(
            model_name="test_model",
            interval_hours=24,
        )
        decision = trigger.evaluate()
        assert decision.should_retrain is True
        assert "scheduled_interval_reached" in decision.reasons
        assert decision.priority == RetrainPriority.LOW

    def test_evaluate_when_not_due(self) -> None:
        """Test evaluate when retrain is not due."""
        trigger = ScheduledRetrainTrigger(
            model_name="test_model",
            interval_hours=24,
        )
        trigger.record_retrain()
        decision = trigger.evaluate()
        assert decision.should_retrain is False
        assert "not_due_yet" in decision.reasons

    def test_evaluate_when_disabled(self) -> None:
        """Test evaluate when trigger is disabled."""
        trigger = ScheduledRetrainTrigger(
            model_name="test_model",
            interval_hours=24,
            enabled=False,
        )
        decision = trigger.evaluate()
        assert decision.should_retrain is False
        assert "trigger_disabled" in decision.reasons

    def test_time_until_next(self) -> None:
        """Test time_until_next calculation."""
        trigger = ScheduledRetrainTrigger(
            model_name="test_model",
            interval_hours=24,
        )
        # No retrain yet
        assert trigger.time_until_next() is None

        # After retrain
        trigger.record_retrain()
        remaining = trigger.time_until_next()
        assert remaining is not None
        assert remaining <= timedelta(hours=24)


class TestRetrainOrchestrator:
    """Tests for RetrainOrchestrator."""

    @pytest.fixture
    def orchestrator(self) -> RetrainOrchestrator:
        """Create a retrain orchestrator."""
        return RetrainOrchestrator(model_name="test_model")

    @pytest.fixture
    def mock_executor(self) -> MockRetrainExecutor:
        """Create a mock retrain executor."""
        return MockRetrainExecutor()

    def test_add_drift_trigger(self, orchestrator: RetrainOrchestrator) -> None:
        """Test adding drift trigger."""
        trigger = AutoRetrainTrigger(model_name="test_model")
        orchestrator.add_drift_trigger(trigger)
        assert len(orchestrator._drift_triggers) == 1

    def test_add_scheduled_trigger(self, orchestrator: RetrainOrchestrator) -> None:
        """Test adding scheduled trigger."""
        trigger = ScheduledRetrainTrigger(model_name="test_model")
        orchestrator.add_scheduled_trigger(trigger)
        assert len(orchestrator._scheduled_triggers) == 1

    def test_evaluate_no_triggers(self, orchestrator: RetrainOrchestrator) -> None:
        """Test evaluate with no triggers."""
        decision = orchestrator.evaluate({"data_drift_psi": 0.35})
        assert decision.should_retrain is False

    def test_evaluate_with_drift_trigger(
        self, orchestrator: RetrainOrchestrator
    ) -> None:
        """Test evaluate with drift trigger."""
        trigger = AutoRetrainTrigger(model_name="test_model")
        orchestrator.add_drift_trigger(trigger)

        metrics = {"data_drift_psi": 0.35}
        decision = orchestrator.evaluate(metrics)
        assert decision.should_retrain is True

    @pytest.mark.asyncio
    async def test_evaluate_and_execute_no_retrain(
        self, orchestrator: RetrainOrchestrator, mock_executor: MockRetrainExecutor
    ) -> None:
        """Test evaluate_and_execute when no retrain needed."""
        orchestrator._executor = mock_executor
        trigger = AutoRetrainTrigger(model_name="test_model")
        orchestrator.add_drift_trigger(trigger)

        metrics = {"data_drift_psi": 0.05}  # Below threshold
        decision, job = await orchestrator.evaluate_and_execute(metrics)
        assert decision.should_retrain is False
        assert job is None

    @pytest.mark.asyncio
    async def test_evaluate_and_execute_with_retrain(
        self, orchestrator: RetrainOrchestrator, mock_executor: MockRetrainExecutor
    ) -> None:
        """Test evaluate_and_execute when retrain is triggered."""
        orchestrator._executor = mock_executor
        trigger = AutoRetrainTrigger(model_name="test_model")
        orchestrator.add_drift_trigger(trigger)

        metrics = {"data_drift_psi": 0.35}  # Above threshold
        decision, job = await orchestrator.evaluate_and_execute(metrics)
        assert decision.should_retrain is True
        assert job is not None
        assert job.status == "completed"

    @pytest.mark.asyncio
    async def test_evaluate_and_execute_no_executor(
        self, orchestrator: RetrainOrchestrator
    ) -> None:
        """Test evaluate_and_execute without executor."""
        trigger = AutoRetrainTrigger(model_name="test_model")
        orchestrator.add_drift_trigger(trigger)

        metrics = {"data_drift_psi": 0.35}
        decision, job = await orchestrator.evaluate_and_execute(metrics)
        assert decision.should_retrain is True
        assert job is None  # No executor configured

    def test_get_status(self, orchestrator: RetrainOrchestrator) -> None:
        """Test get_status returns correct info."""
        status = orchestrator.get_status()
        assert status["model_name"] == "test_model"
        assert status["drift_triggers"] == 0
        assert status["scheduled_triggers"] == 0
        assert status["has_executor"] is False


class TestMockRetrainExecutor:
    """Tests for MockRetrainExecutor."""

    @pytest.fixture
    def executor(self) -> MockRetrainExecutor:
        """Create a mock executor."""
        return MockRetrainExecutor()

    @pytest.mark.asyncio
    async def test_execute_retrain(self, executor: MockRetrainExecutor) -> None:
        """Test execute_retrain returns a job."""
        job = await executor.execute_retrain(
            model_name="test_model",
            priority=RetrainPriority.HIGH,
            reasons=["data_drift_psi=0.35"],
        )
        assert job.model_name == "test_model"
        assert job.priority == RetrainPriority.HIGH
        assert job.status == "completed"
        assert job.job_id is not None

    @pytest.mark.asyncio
    async def test_execute_retrain_increments_counter(
        self, executor: MockRetrainExecutor
    ) -> None:
        """Test execute_retrain increments job counter."""
        await executor.execute_retrain("model1", RetrainPriority.LOW, ["test"])
        await executor.execute_retrain("model2", RetrainPriority.LOW, ["test"])

        assert executor._job_counter == 2
        assert len(executor._jobs) == 2


class TestRetrainJob:
    """Tests for RetrainJob dataclass."""

    def test_job_creation(self) -> None:
        """Test creating a retrain job."""
        job = RetrainJob(
            job_id="test-123",
            model_name="test_model",
            priority=RetrainPriority.HIGH,
            reasons=["data_drift_psi=0.35"],
            status="running",
        )
        assert job.job_id == "test-123"
        assert job.model_name == "test_model"
        assert job.priority == RetrainPriority.HIGH
        assert job.status == "running"
        assert job.completed_at is None

    def test_job_with_metrics(self) -> None:
        """Test creating a job with metrics."""
        job = RetrainJob(
            job_id="test-123",
            model_name="test_model",
            priority=RetrainPriority.HIGH,
            reasons=["test"],
            status="completed",
            metrics_before={"f1_score": 0.7},
            metrics_after={"f1_score": 0.85},
        )
        assert job.metrics_before == {"f1_score": 0.7}
        assert job.metrics_after == {"f1_score": 0.85}
