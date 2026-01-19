"""Auto Retrain Trigger - Automated model retraining based on drift detection.

This module provides automatic retraining triggers based on:
- Data drift (PSI threshold)
- Concept drift (F1 drop)
- Scheduled intervals
- Manual triggers
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class RetrainPriority(Enum):
    """Retrain job priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class RetrainReason(Enum):
    """Reasons for triggering retraining."""

    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    FEATURE_DRIFT = "feature_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class RetrainDecision:
    """Result of retrain trigger evaluation."""

    should_retrain: bool
    reasons: list[str] = field(default_factory=list)
    priority: RetrainPriority = RetrainPriority.NORMAL
    metrics: dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    cooldown_remaining: timedelta | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "should_retrain": self.should_retrain,
            "reasons": self.reasons,
            "priority": self.priority.value,
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "cooldown_remaining": str(self.cooldown_remaining) if self.cooldown_remaining else None,
        }


@dataclass
class RetrainJob:
    """Represents a retraining job."""

    job_id: str
    model_name: str
    priority: RetrainPriority
    reasons: list[str]
    status: str = "pending"  # pending, running, completed, failed
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    started_at: datetime | None = None
    completed_at: datetime | None = None
    error_message: str | None = None
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)


class RetrainExecutorProtocol(Protocol):
    """Protocol for retrain job execution."""

    async def execute_retrain(
        self,
        model_name: str,
        priority: RetrainPriority,
        reasons: list[str],
    ) -> RetrainJob:
        """Execute a retraining job.

        Args:
            model_name: Name of the model to retrain.
            priority: Job priority.
            reasons: Reasons for retraining.

        Returns:
            RetrainJob with execution status.
        """
        ...


class AutoRetrainTrigger:
    """Automatic retraining trigger based on drift metrics.

    Monitors drift metrics and triggers retraining when thresholds are exceeded.

    Example:
        >>> trigger = AutoRetrainTrigger(
        ...     model_name="text_classifier",
        ...     data_drift_threshold=0.25,
        ...     concept_drift_threshold=0.10,
        ... )
        >>> decision = trigger.evaluate({
        ...     "data_drift_psi": 0.30,
        ...     "f1_drop": 0.08,
        ... })
        >>> if decision.should_retrain:
        ...     print(f"Retrain needed: {decision.reasons}")
    """

    def __init__(
        self,
        model_name: str,
        data_drift_threshold: float = 0.25,
        concept_drift_threshold: float = 0.10,
        feature_drift_threshold: float = 0.25,
        f1_critical_threshold: float = 0.65,
        min_interval_hours: int = 24,
        max_retrains_per_day: int = 3,
    ) -> None:
        """Initialize the retrain trigger.

        Args:
            model_name: Name of the model to monitor.
            data_drift_threshold: PSI threshold for data drift.
            concept_drift_threshold: F1 drop threshold for concept drift.
            feature_drift_threshold: PSI threshold for feature drift.
            f1_critical_threshold: Critical F1 score threshold.
            min_interval_hours: Minimum hours between retrains.
            max_retrains_per_day: Maximum retrains allowed per day.
        """
        self.model_name = model_name
        self.data_drift_threshold = data_drift_threshold
        self.concept_drift_threshold = concept_drift_threshold
        self.feature_drift_threshold = feature_drift_threshold
        self.f1_critical_threshold = f1_critical_threshold
        self.min_interval = timedelta(hours=min_interval_hours)
        self.max_retrains_per_day = max_retrains_per_day

        self._last_retrain: datetime | None = None
        self._retrain_history: list[datetime] = []
        self._enabled = True

    @property
    def enabled(self) -> bool:
        """Check if trigger is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable the trigger."""
        self._enabled = True
        logger.info(f"Retrain trigger enabled for model: {self.model_name}")

    def disable(self) -> None:
        """Disable the trigger."""
        self._enabled = False
        logger.info(f"Retrain trigger disabled for model: {self.model_name}")

    def evaluate(self, metrics: dict[str, float]) -> RetrainDecision:
        """Evaluate metrics and decide if retraining is needed.

        Args:
            metrics: Dictionary of metric name to value.
                Expected keys: data_drift_psi, f1_drop, feature_drift_psi,
                              current_f1, etc.

        Returns:
            RetrainDecision with evaluation result.
        """
        if not self._enabled:
            return RetrainDecision(
                should_retrain=False,
                reasons=["trigger_disabled"],
                metrics=metrics,
            )

        # Check cooldown
        cooldown = self._check_cooldown()
        if cooldown:
            return RetrainDecision(
                should_retrain=False,
                reasons=["cooldown_active"],
                cooldown_remaining=cooldown,
                metrics=metrics,
            )

        # Check daily limit
        if self._exceeded_daily_limit():
            return RetrainDecision(
                should_retrain=False,
                reasons=["daily_limit_exceeded"],
                metrics=metrics,
            )

        # Evaluate drift conditions
        reasons = []
        priority = RetrainPriority.NORMAL

        # Data drift check
        data_drift_psi = metrics.get("data_drift_psi", 0.0)
        if data_drift_psi > self.data_drift_threshold:
            reasons.append(f"data_drift_psi={data_drift_psi:.3f}")
            if data_drift_psi > self.data_drift_threshold * 2:
                priority = max(priority, RetrainPriority.HIGH, key=lambda p: list(RetrainPriority).index(p))

        # Concept drift check (F1 drop)
        f1_drop = metrics.get("f1_drop", 0.0)
        if f1_drop > self.concept_drift_threshold:
            reasons.append(f"f1_drop={f1_drop:.3f}")
            if f1_drop > 0.15:
                priority = RetrainPriority.URGENT
            elif f1_drop > 0.10:
                priority = max(priority, RetrainPriority.HIGH, key=lambda p: list(RetrainPriority).index(p))

        # Feature drift check
        feature_drift_psi = metrics.get("feature_drift_psi", 0.0)
        if feature_drift_psi > self.feature_drift_threshold:
            reasons.append(f"feature_drift_psi={feature_drift_psi:.3f}")

        # Critical F1 check
        current_f1 = metrics.get("current_f1")
        if current_f1 is not None and current_f1 < self.f1_critical_threshold:
            reasons.append(f"critical_f1={current_f1:.3f}")
            priority = RetrainPriority.URGENT

        should_retrain = len(reasons) > 0

        return RetrainDecision(
            should_retrain=should_retrain,
            reasons=reasons,
            priority=priority,
            metrics=metrics,
        )

    def _check_cooldown(self) -> timedelta | None:
        """Check if cooldown period is active.

        Returns:
            Remaining cooldown time or None if not in cooldown.
        """
        if self._last_retrain is None:
            return None

        elapsed = datetime.now(UTC) - self._last_retrain
        if elapsed < self.min_interval:
            return self.min_interval - elapsed
        return None

    def _exceeded_daily_limit(self) -> bool:
        """Check if daily retrain limit is exceeded.

        Returns:
            True if limit exceeded, False otherwise.
        """
        now = datetime.now(UTC)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # Count retrains today
        today_retrains = sum(
            1 for t in self._retrain_history
            if t >= day_start
        )

        return today_retrains >= self.max_retrains_per_day

    def record_retrain(self, timestamp: datetime | None = None) -> None:
        """Record that a retrain was triggered.

        Args:
            timestamp: Optional timestamp. Defaults to now.
        """
        timestamp = timestamp or datetime.now(UTC)
        self._last_retrain = timestamp
        self._retrain_history.append(timestamp)

        # Clean old history (keep last 7 days)
        cutoff = datetime.now(UTC) - timedelta(days=7)
        self._retrain_history = [t for t in self._retrain_history if t > cutoff]

        logger.info(f"Recorded retrain for model: {self.model_name} at {timestamp}")

    def get_status(self) -> dict[str, Any]:
        """Get current trigger status.

        Returns:
            Status dictionary.
        """
        now = datetime.now(UTC)
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        return {
            "model_name": self.model_name,
            "enabled": self._enabled,
            "thresholds": {
                "data_drift_psi": self.data_drift_threshold,
                "concept_drift_f1_drop": self.concept_drift_threshold,
                "feature_drift_psi": self.feature_drift_threshold,
                "f1_critical": self.f1_critical_threshold,
            },
            "cooldown": {
                "min_interval_hours": self.min_interval.total_seconds() / 3600,
                "last_retrain": self._last_retrain.isoformat() if self._last_retrain else None,
                "remaining": str(self._check_cooldown()) if self._check_cooldown() else None,
            },
            "daily_limit": {
                "max_per_day": self.max_retrains_per_day,
                "today_count": sum(1 for t in self._retrain_history if t >= day_start),
            },
        }

    def reset(self) -> None:
        """Reset trigger state."""
        self._last_retrain = None
        self._retrain_history.clear()
        logger.info(f"Reset retrain trigger for model: {self.model_name}")


class ScheduledRetrainTrigger:
    """Scheduled retraining based on time intervals.

    Example:
        >>> trigger = ScheduledRetrainTrigger(
        ...     model_name="text_classifier",
        ...     interval_hours=168,  # Weekly
        ... )
        >>> if trigger.is_due():
        ...     trigger.record_retrain()
    """

    def __init__(
        self,
        model_name: str,
        interval_hours: int = 168,  # Default: weekly
        enabled: bool = True,
    ) -> None:
        """Initialize the scheduled trigger.

        Args:
            model_name: Name of the model.
            interval_hours: Hours between scheduled retrains.
            enabled: Whether the trigger is enabled.
        """
        self.model_name = model_name
        self.interval = timedelta(hours=interval_hours)
        self._enabled = enabled
        self._last_retrain: datetime | None = None

    @property
    def enabled(self) -> bool:
        """Check if trigger is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable the trigger."""
        self._enabled = True

    def disable(self) -> None:
        """Disable the trigger."""
        self._enabled = False

    def is_due(self) -> bool:
        """Check if scheduled retrain is due.

        Returns:
            True if retrain is due, False otherwise.
        """
        if not self._enabled:
            return False

        if self._last_retrain is None:
            return True

        return datetime.now(UTC) - self._last_retrain >= self.interval

    def time_until_next(self) -> timedelta | None:
        """Get time until next scheduled retrain.

        Returns:
            Time remaining or None if due now.
        """
        if self._last_retrain is None:
            return None

        elapsed = datetime.now(UTC) - self._last_retrain
        if elapsed >= self.interval:
            return None

        return self.interval - elapsed

    def record_retrain(self, timestamp: datetime | None = None) -> None:
        """Record that a retrain occurred.

        Args:
            timestamp: Optional timestamp. Defaults to now.
        """
        self._last_retrain = timestamp or datetime.now(UTC)

    def evaluate(self) -> RetrainDecision:
        """Evaluate if scheduled retrain is due.

        Returns:
            RetrainDecision with evaluation result.
        """
        if not self._enabled:
            return RetrainDecision(
                should_retrain=False,
                reasons=["trigger_disabled"],
            )

        if self.is_due():
            return RetrainDecision(
                should_retrain=True,
                reasons=["scheduled_interval_reached"],
                priority=RetrainPriority.LOW,
            )

        return RetrainDecision(
            should_retrain=False,
            reasons=["not_due_yet"],
        )


class RetrainOrchestrator:
    """Orchestrate retraining across multiple triggers.

    Combines drift-based and scheduled triggers to manage retraining.

    Example:
        >>> orchestrator = RetrainOrchestrator("text_classifier")
        >>> orchestrator.add_drift_trigger(drift_trigger)
        >>> orchestrator.add_scheduled_trigger(scheduled_trigger)
        >>> decision = await orchestrator.evaluate_and_execute(metrics)
    """

    def __init__(
        self,
        model_name: str,
        executor: RetrainExecutorProtocol | None = None,
    ) -> None:
        """Initialize the orchestrator.

        Args:
            model_name: Name of the model.
            executor: Optional executor for running retrain jobs.
        """
        self.model_name = model_name
        self._executor = executor
        self._drift_triggers: list[AutoRetrainTrigger] = []
        self._scheduled_triggers: list[ScheduledRetrainTrigger] = []
        self._job_history: list[RetrainJob] = []

    def add_drift_trigger(self, trigger: AutoRetrainTrigger) -> None:
        """Add a drift-based trigger.

        Args:
            trigger: Drift trigger to add.
        """
        self._drift_triggers.append(trigger)

    def add_scheduled_trigger(self, trigger: ScheduledRetrainTrigger) -> None:
        """Add a scheduled trigger.

        Args:
            trigger: Scheduled trigger to add.
        """
        self._scheduled_triggers.append(trigger)

    def evaluate(self, metrics: dict[str, float]) -> RetrainDecision:
        """Evaluate all triggers and return combined decision.

        Args:
            metrics: Current metrics.

        Returns:
            Combined RetrainDecision.
        """
        all_reasons: list[str] = []
        highest_priority = RetrainPriority.LOW
        should_retrain = False

        # Evaluate drift triggers
        for trigger in self._drift_triggers:
            decision = trigger.evaluate(metrics)
            if decision.should_retrain:
                should_retrain = True
                all_reasons.extend(decision.reasons)
                if list(RetrainPriority).index(decision.priority) > list(RetrainPriority).index(highest_priority):
                    highest_priority = decision.priority

        # Evaluate scheduled triggers
        for trigger in self._scheduled_triggers:
            decision = trigger.evaluate()
            if decision.should_retrain:
                should_retrain = True
                all_reasons.extend(decision.reasons)

        return RetrainDecision(
            should_retrain=should_retrain,
            reasons=all_reasons,
            priority=highest_priority,
            metrics=metrics,
        )

    async def evaluate_and_execute(
        self,
        metrics: dict[str, float],
    ) -> tuple[RetrainDecision, RetrainJob | None]:
        """Evaluate triggers and execute retrain if needed.

        Args:
            metrics: Current metrics.

        Returns:
            Tuple of (decision, job or None).
        """
        decision = self.evaluate(metrics)

        if not decision.should_retrain:
            return decision, None

        if self._executor is None:
            logger.warning("No executor configured, cannot execute retrain")
            return decision, None

        # Execute retrain
        job = await self._executor.execute_retrain(
            model_name=self.model_name,
            priority=decision.priority,
            reasons=decision.reasons,
        )

        self._job_history.append(job)

        # Record retrain in all triggers
        for trigger in self._drift_triggers:
            trigger.record_retrain()
        for trigger in self._scheduled_triggers:
            trigger.record_retrain()

        return decision, job

    def get_job_history(self, limit: int = 10) -> list[RetrainJob]:
        """Get recent job history.

        Args:
            limit: Maximum number of jobs to return.

        Returns:
            List of recent RetrainJobs.
        """
        return self._job_history[-limit:]

    def get_status(self) -> dict[str, Any]:
        """Get orchestrator status.

        Returns:
            Status dictionary.
        """
        return {
            "model_name": self.model_name,
            "drift_triggers": len(self._drift_triggers),
            "scheduled_triggers": len(self._scheduled_triggers),
            "has_executor": self._executor is not None,
            "total_jobs": len(self._job_history),
            "recent_jobs": [
                {
                    "job_id": job.job_id,
                    "status": job.status,
                    "created_at": job.created_at.isoformat(),
                }
                for job in self._job_history[-5:]
            ],
        }


class MockRetrainExecutor:
    """Mock executor for testing."""

    def __init__(self) -> None:
        """Initialize mock executor."""
        self._job_counter = 0
        self._jobs: list[RetrainJob] = []

    async def execute_retrain(
        self,
        model_name: str,
        priority: RetrainPriority,
        reasons: list[str],
    ) -> RetrainJob:
        """Execute a mock retrain job.

        Args:
            model_name: Model name.
            priority: Job priority.
            reasons: Retrain reasons.

        Returns:
            Mock RetrainJob.
        """
        self._job_counter += 1
        job_id = f"retrain-{model_name}-{self._job_counter}"

        job = RetrainJob(
            job_id=job_id,
            model_name=model_name,
            priority=priority,
            reasons=reasons,
            status="running",
            started_at=datetime.now(UTC),
        )

        # Simulate async execution
        await asyncio.sleep(0.01)

        job.status = "completed"
        job.completed_at = datetime.now(UTC)

        self._jobs.append(job)
        logger.info(f"Mock retrain completed: {job_id}")

        return job
