"""Champion Rollback - Model version rollback and management.

This module provides automatic rollback capabilities:
- Champion/Challenger model management
- Automatic rollback on performance degradation
- Version history tracking
- Health checks for model versions
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model deployment status."""

    ACTIVE = "active"
    CHALLENGER = "challenger"
    RETIRED = "retired"
    ROLLED_BACK = "rolled_back"


class RollbackReason(Enum):
    """Reasons for model rollback."""

    PERFORMANCE_DEGRADATION = "performance_degradation"
    ERROR_RATE_HIGH = "error_rate_high"
    LATENCY_HIGH = "latency_high"
    MANUAL = "manual"
    HEALTH_CHECK_FAILED = "health_check_failed"
    DRIFT_DETECTED = "drift_detected"


@dataclass
class ModelVersion:
    """Represents a model version."""

    version: str
    model_name: str
    status: ModelStatus = ModelStatus.CHALLENGER
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    promoted_at: datetime | None = None
    retired_at: datetime | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "version": self.version,
            "model_name": self.model_name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "promoted_at": self.promoted_at.isoformat() if self.promoted_at else None,
            "retired_at": self.retired_at.isoformat() if self.retired_at else None,
            "metrics": self.metrics,
            "metadata": self.metadata,
        }


@dataclass
class RollbackResult:
    """Result of a rollback operation."""

    success: bool
    from_version: str
    to_version: str
    reason: RollbackReason
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metrics_before: dict[str, float] = field(default_factory=dict)
    metrics_after: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation.
        """
        return {
            "success": self.success,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "reason": self.reason.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metrics_before": self.metrics_before,
            "metrics_after": self.metrics_after,
        }


@dataclass
class HealthCheckResult:
    """Result of model health check."""

    healthy: bool
    version: str
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class ModelRegistryProtocol(Protocol):
    """Protocol for model registry operations."""

    async def get_champion(self, model_name: str) -> ModelVersion | None:
        """Get current champion version.

        Args:
            model_name: Name of the model.

        Returns:
            Champion ModelVersion or None.
        """
        ...

    async def get_version(self, model_name: str, version: str) -> ModelVersion | None:
        """Get specific model version.

        Args:
            model_name: Name of the model.
            version: Version string.

        Returns:
            ModelVersion or None.
        """
        ...

    async def list_versions(
        self,
        model_name: str,
        status: ModelStatus | None = None,
    ) -> list[ModelVersion]:
        """List model versions.

        Args:
            model_name: Name of the model.
            status: Optional status filter.

        Returns:
            List of ModelVersions.
        """
        ...

    async def set_champion(self, model_name: str, version: str) -> bool:
        """Set a version as champion.

        Args:
            model_name: Name of the model.
            version: Version to promote.

        Returns:
            True if successful.
        """
        ...

    async def retire_version(self, model_name: str, version: str) -> bool:
        """Retire a model version.

        Args:
            model_name: Name of the model.
            version: Version to retire.

        Returns:
            True if successful.
        """
        ...


class ChampionRollback:
    """Manage champion model rollback.

    Provides automatic and manual rollback capabilities for model versions.

    Example:
        >>> rollback = ChampionRollback(
        ...     model_name="text_classifier",
        ...     registry=model_registry,
        ... )
        >>> result = await rollback.rollback_to_previous(
        ...     reason=RollbackReason.PERFORMANCE_DEGRADATION,
        ... )
        >>> if result.success:
        ...     print(f"Rolled back to version {result.to_version}")
    """

    def __init__(
        self,
        model_name: str,
        registry: ModelRegistryProtocol,
        min_versions_to_keep: int = 3,
        rollback_cooldown_hours: int = 1,
    ) -> None:
        """Initialize the rollback manager.

        Args:
            model_name: Name of the model.
            registry: Model registry instance.
            min_versions_to_keep: Minimum versions to keep for rollback.
            rollback_cooldown_hours: Hours between rollbacks.
        """
        self.model_name = model_name
        self._registry = registry
        self.min_versions_to_keep = min_versions_to_keep
        self.rollback_cooldown = timedelta(hours=rollback_cooldown_hours)

        self._rollback_history: list[RollbackResult] = []
        self._last_rollback: datetime | None = None

    async def get_current_champion(self) -> ModelVersion | None:
        """Get the current champion version.

        Returns:
            Current champion ModelVersion or None.
        """
        return await self._registry.get_champion(self.model_name)

    async def get_previous_champion(self) -> ModelVersion | None:
        """Get the previous champion version (for rollback).

        Returns:
            Previous champion ModelVersion or None.
        """
        versions = await self._registry.list_versions(self.model_name)

        # Sort by promoted_at descending
        promoted_versions = [
            v for v in versions
            if v.promoted_at is not None and v.status != ModelStatus.ROLLED_BACK
        ]
        promoted_versions.sort(key=lambda v: v.promoted_at or v.created_at, reverse=True)

        # Skip current champion, return next
        if len(promoted_versions) >= 2:
            return promoted_versions[1]

        return None

    async def rollback_to_previous(
        self,
        reason: RollbackReason,
        metrics_before: dict[str, float] | None = None,
    ) -> RollbackResult:
        """Rollback to the previous champion version.

        Args:
            reason: Reason for rollback.
            metrics_before: Current metrics before rollback.

        Returns:
            RollbackResult with operation outcome.
        """
        # Check cooldown
        if not self._can_rollback():
            return RollbackResult(
                success=False,
                from_version="",
                to_version="",
                reason=reason,
                message="Rollback cooldown active",
            )

        current = await self.get_current_champion()
        if current is None:
            return RollbackResult(
                success=False,
                from_version="",
                to_version="",
                reason=reason,
                message="No current champion found",
            )

        previous = await self.get_previous_champion()
        if previous is None:
            return RollbackResult(
                success=False,
                from_version=current.version,
                to_version="",
                reason=reason,
                message="No previous version available for rollback",
            )

        return await self.rollback_to_version(
            target_version=previous.version,
            reason=reason,
            metrics_before=metrics_before,
        )

    async def rollback_to_version(
        self,
        target_version: str,
        reason: RollbackReason,
        metrics_before: dict[str, float] | None = None,
    ) -> RollbackResult:
        """Rollback to a specific version.

        Args:
            target_version: Version to rollback to.
            reason: Reason for rollback.
            metrics_before: Current metrics before rollback.

        Returns:
            RollbackResult with operation outcome.
        """
        current = await self.get_current_champion()
        current_version = current.version if current else ""

        # Validate target version exists
        target = await self._registry.get_version(self.model_name, target_version)
        if target is None:
            return RollbackResult(
                success=False,
                from_version=current_version,
                to_version=target_version,
                reason=reason,
                message=f"Target version {target_version} not found",
            )

        # Perform rollback
        try:
            # Set new champion
            success = await self._registry.set_champion(self.model_name, target_version)
            if not success:
                return RollbackResult(
                    success=False,
                    from_version=current_version,
                    to_version=target_version,
                    reason=reason,
                    message="Failed to set new champion",
                )

            # Mark old version as rolled back
            if current:
                await self._registry.retire_version(self.model_name, current_version)

            result = RollbackResult(
                success=True,
                from_version=current_version,
                to_version=target_version,
                reason=reason,
                message=f"Successfully rolled back from {current_version} to {target_version}",
                metrics_before=metrics_before or {},
            )

            self._record_rollback(result)
            logger.info(
                f"Rolled back model {self.model_name} from {current_version} "
                f"to {target_version}: {reason.value}"
            )

            return result

        except Exception as e:
            logger.exception(f"Rollback failed: {e}")
            return RollbackResult(
                success=False,
                from_version=current_version,
                to_version=target_version,
                reason=reason,
                message=f"Rollback failed: {e!s}",
            )

    def _can_rollback(self) -> bool:
        """Check if rollback is allowed (cooldown check).

        Returns:
            True if rollback is allowed.
        """
        if self._last_rollback is None:
            return True

        return datetime.now(UTC) - self._last_rollback >= self.rollback_cooldown

    def _record_rollback(self, result: RollbackResult) -> None:
        """Record a rollback in history.

        Args:
            result: RollbackResult to record.
        """
        self._last_rollback = result.timestamp
        self._rollback_history.append(result)

        # Keep last 100 rollbacks
        if len(self._rollback_history) > 100:
            self._rollback_history = self._rollback_history[-100:]

    def get_rollback_history(self, limit: int = 10) -> list[RollbackResult]:
        """Get recent rollback history.

        Args:
            limit: Maximum number of results.

        Returns:
            List of RollbackResults.
        """
        return self._rollback_history[-limit:]

    def get_status(self) -> dict[str, Any]:
        """Get rollback manager status.

        Returns:
            Status dictionary.
        """
        return {
            "model_name": self.model_name,
            "can_rollback": self._can_rollback(),
            "last_rollback": self._last_rollback.isoformat() if self._last_rollback else None,
            "cooldown_hours": self.rollback_cooldown.total_seconds() / 3600,
            "total_rollbacks": len(self._rollback_history),
        }


class AutoRollbackMonitor:
    """Monitor model health and trigger automatic rollbacks.

    Example:
        >>> monitor = AutoRollbackMonitor(
        ...     rollback_manager=rollback,
        ...     f1_threshold=0.6,
        ...     error_rate_threshold=0.1,
        ... )
        >>> result = await monitor.check_and_rollback(current_metrics)
    """

    def __init__(
        self,
        rollback_manager: ChampionRollback,
        f1_threshold: float = 0.6,
        error_rate_threshold: float = 0.1,
        latency_threshold_ms: float = 5000,
        consecutive_failures: int = 3,
        enabled: bool = True,
    ) -> None:
        """Initialize the auto rollback monitor.

        Args:
            rollback_manager: ChampionRollback instance.
            f1_threshold: F1 score threshold for rollback.
            error_rate_threshold: Error rate threshold.
            latency_threshold_ms: Latency threshold in milliseconds.
            consecutive_failures: Required consecutive failures before rollback.
            enabled: Whether auto rollback is enabled.
        """
        self._rollback = rollback_manager
        self.f1_threshold = f1_threshold
        self.error_rate_threshold = error_rate_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.consecutive_failures = consecutive_failures
        self._enabled = enabled

        self._failure_count = 0
        self._last_check: datetime | None = None

    @property
    def enabled(self) -> bool:
        """Check if auto rollback is enabled."""
        return self._enabled

    def enable(self) -> None:
        """Enable auto rollback."""
        self._enabled = True
        logger.info("Auto rollback enabled")

    def disable(self) -> None:
        """Disable auto rollback."""
        self._enabled = False
        logger.info("Auto rollback disabled")

    async def check_health(self, metrics: dict[str, float]) -> HealthCheckResult:
        """Check model health based on metrics.

        Args:
            metrics: Current model metrics.

        Returns:
            HealthCheckResult with check outcome.
        """
        current = await self._rollback.get_current_champion()
        version = current.version if current else "unknown"

        checks = {}
        errors = []

        # F1 score check
        f1 = metrics.get("f1_score")
        if f1 is not None:
            checks["f1_score"] = f1 >= self.f1_threshold
            if not checks["f1_score"]:
                errors.append(f"F1 score {f1:.3f} below threshold {self.f1_threshold}")

        # Error rate check
        error_rate = metrics.get("error_rate")
        if error_rate is not None:
            checks["error_rate"] = error_rate <= self.error_rate_threshold
            if not checks["error_rate"]:
                errors.append(f"Error rate {error_rate:.3f} above threshold {self.error_rate_threshold}")

        # Latency check
        latency = metrics.get("latency_p99_ms")
        if latency is not None:
            checks["latency"] = latency <= self.latency_threshold_ms
            if not checks["latency"]:
                errors.append(f"Latency {latency:.1f}ms above threshold {self.latency_threshold_ms}ms")

        healthy = all(checks.values()) if checks else True

        return HealthCheckResult(
            healthy=healthy,
            version=version,
            checks=checks,
            metrics=metrics,
            errors=errors,
        )

    async def check_and_rollback(
        self,
        metrics: dict[str, float],
    ) -> tuple[HealthCheckResult, RollbackResult | None]:
        """Check health and trigger rollback if needed.

        Args:
            metrics: Current model metrics.

        Returns:
            Tuple of (health check result, rollback result or None).
        """
        self._last_check = datetime.now(UTC)
        health = await self.check_health(metrics)

        if not self._enabled:
            return health, None

        if health.healthy:
            self._failure_count = 0
            return health, None

        # Increment failure count
        self._failure_count += 1
        logger.warning(
            f"Health check failed ({self._failure_count}/{self.consecutive_failures}): "
            f"{health.errors}"
        )

        if self._failure_count < self.consecutive_failures:
            return health, None

        # Determine rollback reason
        if "f1_score" in health.checks and not health.checks["f1_score"]:
            reason = RollbackReason.PERFORMANCE_DEGRADATION
        elif "error_rate" in health.checks and not health.checks["error_rate"]:
            reason = RollbackReason.ERROR_RATE_HIGH
        elif "latency" in health.checks and not health.checks["latency"]:
            reason = RollbackReason.LATENCY_HIGH
        else:
            reason = RollbackReason.HEALTH_CHECK_FAILED

        # Trigger rollback
        result = await self._rollback.rollback_to_previous(
            reason=reason,
            metrics_before=metrics,
        )

        if result.success:
            self._failure_count = 0

        return health, result

    def get_status(self) -> dict[str, Any]:
        """Get monitor status.

        Returns:
            Status dictionary.
        """
        return {
            "enabled": self._enabled,
            "thresholds": {
                "f1_score": self.f1_threshold,
                "error_rate": self.error_rate_threshold,
                "latency_ms": self.latency_threshold_ms,
            },
            "consecutive_failures_required": self.consecutive_failures,
            "current_failure_count": self._failure_count,
            "last_check": self._last_check.isoformat() if self._last_check else None,
        }

    def reset(self) -> None:
        """Reset failure counter."""
        self._failure_count = 0


class InMemoryModelRegistry:
    """In-memory model registry for testing."""

    def __init__(self) -> None:
        """Initialize the registry."""
        self._versions: dict[str, dict[str, ModelVersion]] = {}
        self._champions: dict[str, str] = {}

    def add_version(self, version: ModelVersion) -> None:
        """Add a model version.

        Args:
            version: ModelVersion to add.
        """
        if version.model_name not in self._versions:
            self._versions[version.model_name] = {}
        self._versions[version.model_name][version.version] = version

    async def get_champion(self, model_name: str) -> ModelVersion | None:
        """Get current champion version."""
        champion_version = self._champions.get(model_name)
        if champion_version is None:
            return None

        versions = self._versions.get(model_name, {})
        return versions.get(champion_version)

    async def get_version(self, model_name: str, version: str) -> ModelVersion | None:
        """Get specific model version."""
        versions = self._versions.get(model_name, {})
        return versions.get(version)

    async def list_versions(
        self,
        model_name: str,
        status: ModelStatus | None = None,
    ) -> list[ModelVersion]:
        """List model versions."""
        versions = self._versions.get(model_name, {})
        result = list(versions.values())

        if status:
            result = [v for v in result if v.status == status]

        return result

    async def set_champion(self, model_name: str, version: str) -> bool:
        """Set a version as champion."""
        versions = self._versions.get(model_name, {})
        if version not in versions:
            return False

        # Demote current champion
        current_champion = self._champions.get(model_name)
        if current_champion and current_champion in versions:
            versions[current_champion].status = ModelStatus.RETIRED

        # Promote new champion
        versions[version].status = ModelStatus.ACTIVE
        versions[version].promoted_at = datetime.now(UTC)
        self._champions[model_name] = version

        return True

    async def retire_version(self, model_name: str, version: str) -> bool:
        """Retire a model version."""
        versions = self._versions.get(model_name, {})
        if version not in versions:
            return False

        versions[version].status = ModelStatus.ROLLED_BACK
        versions[version].retired_at = datetime.now(UTC)
        return True
