"""Tests for ChampionRollback and AutoRollbackMonitor."""

from __future__ import annotations

import pytest

from ml_service.monitoring.actions.rollback import (
    AutoRollbackMonitor,
    ChampionRollback,
    HealthCheckResult,
    InMemoryModelRegistry,
    ModelStatus,
    ModelVersion,
    RollbackReason,
    RollbackResult,
)


class TestModelStatus:
    """Tests for ModelStatus enum."""

    def test_status_values(self) -> None:
        """Test status enum values."""
        assert ModelStatus.ACTIVE.value == "active"
        assert ModelStatus.CHALLENGER.value == "challenger"
        assert ModelStatus.RETIRED.value == "retired"
        assert ModelStatus.ROLLED_BACK.value == "rolled_back"


class TestRollbackReason:
    """Tests for RollbackReason enum."""

    def test_reason_values(self) -> None:
        """Test reason enum values."""
        assert RollbackReason.PERFORMANCE_DEGRADATION.value == "performance_degradation"
        assert RollbackReason.ERROR_RATE_HIGH.value == "error_rate_high"
        assert RollbackReason.LATENCY_HIGH.value == "latency_high"
        assert RollbackReason.MANUAL.value == "manual"
        assert RollbackReason.HEALTH_CHECK_FAILED.value == "health_check_failed"
        assert RollbackReason.DRIFT_DETECTED.value == "drift_detected"


class TestModelVersion:
    """Tests for ModelVersion dataclass."""

    def test_version_creation(self) -> None:
        """Test creating a model version."""
        version = ModelVersion(
            version="v1.0.0",
            model_name="test_model",
            status=ModelStatus.ACTIVE,
            metrics={"f1_score": 0.85},
        )
        assert version.version == "v1.0.0"
        assert version.model_name == "test_model"
        assert version.status == ModelStatus.ACTIVE
        assert version.metrics == {"f1_score": 0.85}

    def test_version_default_fields(self) -> None:
        """Test version with default fields."""
        version = ModelVersion(
            version="v1.0.0",
            model_name="test_model",
        )
        assert version.status == ModelStatus.CHALLENGER
        assert version.metrics == {}
        assert version.metadata == {}
        assert version.promoted_at is None
        assert version.retired_at is None

    def test_version_to_dict(self) -> None:
        """Test converting version to dictionary."""
        version = ModelVersion(
            version="v1.0.0",
            model_name="test_model",
            status=ModelStatus.ACTIVE,
        )
        result = version.to_dict()
        assert result["version"] == "v1.0.0"
        assert result["model_name"] == "test_model"
        assert result["status"] == "active"


class TestRollbackResult:
    """Tests for RollbackResult dataclass."""

    def test_successful_rollback(self) -> None:
        """Test successful rollback result."""
        result = RollbackResult(
            success=True,
            from_version="v2.0.0",
            to_version="v1.0.0",
            reason=RollbackReason.PERFORMANCE_DEGRADATION,
            message="Rollback successful",
        )
        assert result.success is True
        assert result.from_version == "v2.0.0"
        assert result.to_version == "v1.0.0"

    def test_failed_rollback(self) -> None:
        """Test failed rollback result."""
        result = RollbackResult(
            success=False,
            from_version="v2.0.0",
            to_version="",
            reason=RollbackReason.MANUAL,
            message="No previous version available",
        )
        assert result.success is False
        assert result.to_version == ""

    def test_result_to_dict(self) -> None:
        """Test converting result to dictionary."""
        result = RollbackResult(
            success=True,
            from_version="v2.0.0",
            to_version="v1.0.0",
            reason=RollbackReason.MANUAL,
            message="test",
        )
        data = result.to_dict()
        assert data["success"] is True
        assert data["reason"] == "manual"


class TestHealthCheckResult:
    """Tests for HealthCheckResult dataclass."""

    def test_healthy_result(self) -> None:
        """Test healthy check result."""
        result = HealthCheckResult(
            healthy=True,
            version="v1.0.0",
            checks={"f1_score": True, "error_rate": True},
            metrics={"f1_score": 0.9, "error_rate": 0.01},
            errors=[],
        )
        assert result.healthy is True
        assert len(result.checks) == 2
        assert len(result.errors) == 0

    def test_unhealthy_result(self) -> None:
        """Test unhealthy check result."""
        result = HealthCheckResult(
            healthy=False,
            version="v1.0.0",
            checks={"f1_score": False, "error_rate": True},
            metrics={"f1_score": 0.5, "error_rate": 0.01},
            errors=["F1 score below threshold"],
        )
        assert result.healthy is False
        assert len(result.errors) == 1


class TestInMemoryModelRegistry:
    """Tests for InMemoryModelRegistry."""

    @pytest.fixture
    def registry(self) -> InMemoryModelRegistry:
        """Create an in-memory registry."""
        return InMemoryModelRegistry()

    def test_add_version(self, registry: InMemoryModelRegistry) -> None:
        """Test adding a model version."""
        version = ModelVersion(
            version="v1.0.0",
            model_name="test_model",
            status=ModelStatus.ACTIVE,
        )
        registry.add_version(version)
        assert "test_model" in registry._versions
        assert "v1.0.0" in registry._versions["test_model"]

    @pytest.mark.asyncio
    async def test_get_champion(self, registry: InMemoryModelRegistry) -> None:
        """Test getting champion version."""
        version = ModelVersion(
            version="v1.0.0",
            model_name="test_model",
        )
        registry.add_version(version)
        await registry.set_champion("test_model", "v1.0.0")

        champion = await registry.get_champion("test_model")
        assert champion is not None
        assert champion.version == "v1.0.0"

    @pytest.mark.asyncio
    async def test_get_champion_not_set(
        self, registry: InMemoryModelRegistry
    ) -> None:
        """Test getting champion when not set."""
        champion = await registry.get_champion("nonexistent_model")
        assert champion is None

    @pytest.mark.asyncio
    async def test_set_champion(self, registry: InMemoryModelRegistry) -> None:
        """Test setting champion version."""
        version = ModelVersion(
            version="v1.0.0",
            model_name="test_model",
        )
        registry.add_version(version)

        success = await registry.set_champion("test_model", "v1.0.0")
        assert success is True

    @pytest.mark.asyncio
    async def test_set_champion_nonexistent_version(
        self, registry: InMemoryModelRegistry
    ) -> None:
        """Test setting champion with nonexistent version."""
        success = await registry.set_champion("test_model", "v999.0.0")
        assert success is False

    @pytest.mark.asyncio
    async def test_list_versions(self, registry: InMemoryModelRegistry) -> None:
        """Test listing all versions."""
        for i in range(3):
            version = ModelVersion(
                version=f"v{i}.0.0",
                model_name="test_model",
            )
            registry.add_version(version)

        versions = await registry.list_versions("test_model")
        assert len(versions) == 3

    @pytest.mark.asyncio
    async def test_list_versions_with_status_filter(
        self, registry: InMemoryModelRegistry
    ) -> None:
        """Test listing versions with status filter."""
        v1 = ModelVersion(version="v1.0.0", model_name="test_model", status=ModelStatus.ACTIVE)
        v2 = ModelVersion(version="v2.0.0", model_name="test_model", status=ModelStatus.RETIRED)
        registry.add_version(v1)
        registry.add_version(v2)

        active_versions = await registry.list_versions("test_model", status=ModelStatus.ACTIVE)
        assert len(active_versions) == 1
        assert active_versions[0].version == "v1.0.0"

    @pytest.mark.asyncio
    async def test_get_version(self, registry: InMemoryModelRegistry) -> None:
        """Test getting specific version."""
        version = ModelVersion(
            version="v1.0.0",
            model_name="test_model",
        )
        registry.add_version(version)

        retrieved = await registry.get_version("test_model", "v1.0.0")
        assert retrieved is not None
        assert retrieved.version == "v1.0.0"

    @pytest.mark.asyncio
    async def test_get_version_not_found(
        self, registry: InMemoryModelRegistry
    ) -> None:
        """Test getting nonexistent version."""
        retrieved = await registry.get_version("test_model", "v999.0.0")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_retire_version(self, registry: InMemoryModelRegistry) -> None:
        """Test retiring a version."""
        version = ModelVersion(
            version="v1.0.0",
            model_name="test_model",
            status=ModelStatus.ACTIVE,
        )
        registry.add_version(version)

        success = await registry.retire_version("test_model", "v1.0.0")
        assert success is True

        updated = await registry.get_version("test_model", "v1.0.0")
        assert updated is not None
        assert updated.status == ModelStatus.ROLLED_BACK


class TestChampionRollback:
    """Tests for ChampionRollback."""

    @pytest.fixture
    def registry(self) -> InMemoryModelRegistry:
        """Create a registry."""
        return InMemoryModelRegistry()

    @pytest.fixture
    def rollback_handler(
        self, registry: InMemoryModelRegistry
    ) -> ChampionRollback:
        """Create a rollback handler."""
        return ChampionRollback(
            model_name="test_model",
            registry=registry,
        )

    @pytest.mark.asyncio
    async def test_rollback_to_previous(
        self,
        rollback_handler: ChampionRollback,
        registry: InMemoryModelRegistry,
    ) -> None:
        """Test rollback to previous version."""
        # Setup versions
        v1 = ModelVersion(
            version="v1.0.0",
            model_name="test_model",
        )
        v2 = ModelVersion(
            version="v2.0.0",
            model_name="test_model",
        )
        registry.add_version(v1)
        registry.add_version(v2)
        await registry.set_champion("test_model", "v1.0.0")
        await registry.set_champion("test_model", "v2.0.0")

        result = await rollback_handler.rollback_to_previous(
            reason=RollbackReason.PERFORMANCE_DEGRADATION,
            metrics_before={"f1_score": 0.6},
        )

        assert result.success is True
        assert result.from_version == "v2.0.0"
        assert result.to_version == "v1.0.0"

    @pytest.mark.asyncio
    async def test_rollback_no_champion(
        self, rollback_handler: ChampionRollback
    ) -> None:
        """Test rollback when no champion is set."""
        result = await rollback_handler.rollback_to_previous(
            reason=RollbackReason.MANUAL
        )
        assert result.success is False
        assert "No current champion" in result.message

    @pytest.mark.asyncio
    async def test_rollback_no_previous_version(
        self,
        rollback_handler: ChampionRollback,
        registry: InMemoryModelRegistry,
    ) -> None:
        """Test rollback when no previous version available."""
        v1 = ModelVersion(
            version="v1.0.0",
            model_name="test_model",
        )
        registry.add_version(v1)
        await registry.set_champion("test_model", "v1.0.0")

        result = await rollback_handler.rollback_to_previous(
            reason=RollbackReason.MANUAL
        )
        assert result.success is False
        assert "No previous version" in result.message

    @pytest.mark.asyncio
    async def test_rollback_to_version(
        self,
        rollback_handler: ChampionRollback,
        registry: InMemoryModelRegistry,
    ) -> None:
        """Test rollback to specific version."""
        v1 = ModelVersion(version="v1.0.0", model_name="test_model")
        v2 = ModelVersion(version="v2.0.0", model_name="test_model")
        v3 = ModelVersion(version="v3.0.0", model_name="test_model")
        registry.add_version(v1)
        registry.add_version(v2)
        registry.add_version(v3)
        await registry.set_champion("test_model", "v3.0.0")

        result = await rollback_handler.rollback_to_version(
            target_version="v1.0.0",
            reason=RollbackReason.MANUAL,
        )

        assert result.success is True
        assert result.from_version == "v3.0.0"
        assert result.to_version == "v1.0.0"

    @pytest.mark.asyncio
    async def test_rollback_to_nonexistent_version(
        self,
        rollback_handler: ChampionRollback,
        registry: InMemoryModelRegistry,
    ) -> None:
        """Test rollback to nonexistent version."""
        v1 = ModelVersion(version="v1.0.0", model_name="test_model")
        registry.add_version(v1)
        await registry.set_champion("test_model", "v1.0.0")

        result = await rollback_handler.rollback_to_version(
            target_version="v999.0.0",
            reason=RollbackReason.MANUAL,
        )

        assert result.success is False
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_get_rollback_history(
        self,
        rollback_handler: ChampionRollback,
        registry: InMemoryModelRegistry,
    ) -> None:
        """Test getting rollback history."""
        v1 = ModelVersion(version="v1.0.0", model_name="test_model")
        v2 = ModelVersion(version="v2.0.0", model_name="test_model")
        registry.add_version(v1)
        registry.add_version(v2)
        await registry.set_champion("test_model", "v1.0.0")
        await registry.set_champion("test_model", "v2.0.0")

        await rollback_handler.rollback_to_previous(
            reason=RollbackReason.PERFORMANCE_DEGRADATION
        )

        history = rollback_handler.get_rollback_history()
        assert len(history) == 1
        assert history[0].from_version == "v2.0.0"
        assert history[0].to_version == "v1.0.0"

    @pytest.mark.asyncio
    async def test_rollback_cooldown(
        self,
        rollback_handler: ChampionRollback,
        registry: InMemoryModelRegistry,
    ) -> None:
        """Test rollback cooldown is enforced."""
        v1 = ModelVersion(version="v1.0.0", model_name="test_model")
        v2 = ModelVersion(version="v2.0.0", model_name="test_model")
        registry.add_version(v1)
        registry.add_version(v2)
        await registry.set_champion("test_model", "v1.0.0")
        await registry.set_champion("test_model", "v2.0.0")

        # First rollback should succeed
        result1 = await rollback_handler.rollback_to_previous(
            reason=RollbackReason.PERFORMANCE_DEGRADATION
        )
        assert result1.success is True

        # Re-promote v2 for testing
        await registry.set_champion("test_model", "v2.0.0")

        # Second rollback should be blocked by cooldown
        result2 = await rollback_handler.rollback_to_previous(
            reason=RollbackReason.PERFORMANCE_DEGRADATION
        )
        assert result2.success is False
        assert "cooldown" in result2.message.lower()

    def test_get_status(self, rollback_handler: ChampionRollback) -> None:
        """Test get_status returns correct info."""
        status = rollback_handler.get_status()
        assert status["model_name"] == "test_model"
        assert status["can_rollback"] is True
        assert "cooldown_hours" in status


class TestAutoRollbackMonitor:
    """Tests for AutoRollbackMonitor."""

    @pytest.fixture
    def registry(self) -> InMemoryModelRegistry:
        """Create an in-memory registry."""
        return InMemoryModelRegistry()

    @pytest.fixture
    def rollback_manager(
        self, registry: InMemoryModelRegistry
    ) -> ChampionRollback:
        """Create a rollback manager."""
        return ChampionRollback(
            model_name="test_model",
            registry=registry,
        )

    @pytest.fixture
    def monitor(self, rollback_manager: ChampionRollback) -> AutoRollbackMonitor:
        """Create an auto-rollback monitor."""
        return AutoRollbackMonitor(
            rollback_manager=rollback_manager,
            f1_threshold=0.65,
            error_rate_threshold=0.1,
            latency_threshold_ms=500,
            consecutive_failures=3,
        )

    @pytest.mark.asyncio
    async def test_check_healthy_model(
        self, monitor: AutoRollbackMonitor, registry: InMemoryModelRegistry
    ) -> None:
        """Test health check with healthy metrics."""
        v1 = ModelVersion(version="v1.0.0", model_name="test_model")
        registry.add_version(v1)
        await registry.set_champion("test_model", "v1.0.0")

        metrics = {
            "f1_score": 0.9,
            "error_rate": 0.01,
            "latency_p99_ms": 100,
        }
        health, rollback = await monitor.check_and_rollback(metrics)

        assert health.healthy is True
        assert rollback is None

    @pytest.mark.asyncio
    async def test_check_unhealthy_f1(
        self, monitor: AutoRollbackMonitor, registry: InMemoryModelRegistry
    ) -> None:
        """Test health check with low F1 score triggers rollback."""
        v1 = ModelVersion(version="v1.0.0", model_name="test_model")
        v2 = ModelVersion(version="v2.0.0", model_name="test_model")
        registry.add_version(v1)
        registry.add_version(v2)
        await registry.set_champion("test_model", "v1.0.0")
        await registry.set_champion("test_model", "v2.0.0")

        metrics = {
            "f1_score": 0.5,  # Below threshold
            "error_rate": 0.01,
            "latency_p99_ms": 100,
        }

        # Need consecutive failures to trigger rollback
        for _ in range(2):
            health, rollback = await monitor.check_and_rollback(metrics)
            assert health.healthy is False
            assert rollback is None  # Not enough consecutive failures yet

        # Third failure should trigger rollback
        health, rollback = await monitor.check_and_rollback(metrics)
        assert health.healthy is False
        assert rollback is not None
        assert rollback.success is True
        assert rollback.reason == RollbackReason.PERFORMANCE_DEGRADATION

    @pytest.mark.asyncio
    async def test_check_unhealthy_error_rate(
        self, monitor: AutoRollbackMonitor, registry: InMemoryModelRegistry
    ) -> None:
        """Test health check with high error rate."""
        v1 = ModelVersion(version="v1.0.0", model_name="test_model")
        v2 = ModelVersion(version="v2.0.0", model_name="test_model")
        registry.add_version(v1)
        registry.add_version(v2)
        await registry.set_champion("test_model", "v1.0.0")
        await registry.set_champion("test_model", "v2.0.0")

        metrics = {
            "f1_score": 0.9,
            "error_rate": 0.15,  # Above threshold
            "latency_p99_ms": 100,
        }

        for _ in range(3):
            _health, rollback = await monitor.check_and_rollback(metrics)

        assert rollback is not None
        assert rollback.reason == RollbackReason.ERROR_RATE_HIGH

    @pytest.mark.asyncio
    async def test_check_unhealthy_latency(
        self, monitor: AutoRollbackMonitor, registry: InMemoryModelRegistry
    ) -> None:
        """Test health check with high latency."""
        v1 = ModelVersion(version="v1.0.0", model_name="test_model")
        v2 = ModelVersion(version="v2.0.0", model_name="test_model")
        registry.add_version(v1)
        registry.add_version(v2)
        await registry.set_champion("test_model", "v1.0.0")
        await registry.set_champion("test_model", "v2.0.0")

        metrics = {
            "f1_score": 0.9,
            "error_rate": 0.01,
            "latency_p99_ms": 600,  # Above threshold
        }

        for _ in range(3):
            _health, rollback = await monitor.check_and_rollback(metrics)

        assert rollback is not None
        assert rollback.reason == RollbackReason.LATENCY_HIGH

    @pytest.mark.asyncio
    async def test_consecutive_failures_reset(
        self, monitor: AutoRollbackMonitor, registry: InMemoryModelRegistry
    ) -> None:
        """Test that consecutive failures reset on healthy check."""
        v1 = ModelVersion(version="v1.0.0", model_name="test_model")
        registry.add_version(v1)
        await registry.set_champion("test_model", "v1.0.0")

        bad_metrics = {"f1_score": 0.5, "error_rate": 0.01, "latency_p99_ms": 100}
        good_metrics = {"f1_score": 0.9, "error_rate": 0.01, "latency_p99_ms": 100}

        # Two failures
        await monitor.check_and_rollback(bad_metrics)
        await monitor.check_and_rollback(bad_metrics)
        assert monitor._failure_count == 2

        # One healthy check resets counter
        health, _ = await monitor.check_and_rollback(good_metrics)
        assert health.healthy is True
        assert monitor._failure_count == 0

    @pytest.mark.asyncio
    async def test_no_rollback_when_disabled(
        self, monitor: AutoRollbackMonitor, registry: InMemoryModelRegistry
    ) -> None:
        """Test no rollback when monitor is disabled."""
        v1 = ModelVersion(version="v1.0.0", model_name="test_model")
        v2 = ModelVersion(version="v2.0.0", model_name="test_model")
        registry.add_version(v1)
        registry.add_version(v2)
        await registry.set_champion("test_model", "v1.0.0")
        await registry.set_champion("test_model", "v2.0.0")

        monitor.disable()
        metrics = {"f1_score": 0.5, "error_rate": 0.01, "latency_p99_ms": 100}

        for _ in range(5):
            _health, rollback = await monitor.check_and_rollback(metrics)
            assert rollback is None  # Disabled, no rollback

    @pytest.mark.asyncio
    async def test_no_rollback_when_no_previous_version(
        self, monitor: AutoRollbackMonitor, registry: InMemoryModelRegistry
    ) -> None:
        """Test no successful rollback when there's no previous version."""
        v1 = ModelVersion(version="v1.0.0", model_name="test_model")
        registry.add_version(v1)
        await registry.set_champion("test_model", "v1.0.0")

        metrics = {"f1_score": 0.5, "error_rate": 0.01, "latency_p99_ms": 100}

        for _ in range(3):
            _health, rollback = await monitor.check_and_rollback(metrics)

        # Rollback attempted but failed due to no previous version
        assert rollback is not None
        assert rollback.success is False

    def test_enable_disable(self, monitor: AutoRollbackMonitor) -> None:
        """Test enable/disable functionality."""
        assert monitor.enabled is True

        monitor.disable()
        assert monitor.enabled is False

        monitor.enable()
        assert monitor.enabled is True

    def test_get_status(self, monitor: AutoRollbackMonitor) -> None:
        """Test get_status returns correct info."""
        status = monitor.get_status()
        assert status["enabled"] is True
        assert "thresholds" in status
        assert status["consecutive_failures_required"] == 3

    def test_reset(self, monitor: AutoRollbackMonitor) -> None:
        """Test reset clears failure counter."""
        monitor._failure_count = 5
        monitor.reset()
        assert monitor._failure_count == 0
