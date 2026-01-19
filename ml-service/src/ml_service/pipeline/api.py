"""REST API endpoints for Adversarial MLOps Pipeline.

Provides endpoints for:
- Pipeline status and metrics
- Manual trigger
- History retrieval
- Configuration management
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from pydantic import BaseModel, Field

from ml_service.pipeline.config import PipelineConfig, QualityGateConfig
from ml_service.pipeline.orchestrator import (
    AdversarialPipelineOrchestrator,
    PipelineState,
    create_pipeline,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pipeline", tags=["pipeline"])

# Global pipeline instance (initialized on startup)
_pipeline: AdversarialPipelineOrchestrator | None = None
_running_task: asyncio.Task | None = None


class PipelineStatusResponse(BaseModel):
    """Response for pipeline status endpoint."""

    state: str
    current_cycle_id: str | None
    total_cycles: int
    last_evasion_rate: float | None
    last_decision: str | None
    thresholds: dict[str, float]


class TriggerRequest(BaseModel):
    """Request to trigger pipeline execution."""

    cycle_id: str | None = Field(default=None, description="Optional cycle identifier")
    async_mode: bool = Field(default=True, description="Run asynchronously")


class TriggerResponse(BaseModel):
    """Response for pipeline trigger endpoint."""

    success: bool
    message: str
    cycle_id: str | None
    state: str


class CycleHistoryResponse(BaseModel):
    """Response for pipeline history endpoint."""

    total_cycles: int
    cycles: list[dict[str, Any]]


class MetricsResponse(BaseModel):
    """Response for pipeline metrics endpoint."""

    total_cycles: int
    avg_evasion_rate: float
    max_evasion_rate: float
    min_evasion_rate: float
    retraining_count: int
    success_rate: float
    total_samples_collected: int


class ConfigUpdateRequest(BaseModel):
    """Request to update pipeline configuration."""

    max_evasion_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    min_f1_score: float | None = Field(default=None, ge=0.0, le=1.0)
    attack_batch_size: int | None = Field(default=None, ge=1)
    num_variants: int | None = Field(default=None, ge=1)


def get_pipeline() -> AdversarialPipelineOrchestrator:
    """Get or create pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = create_pipeline()
        logger.info("Created pipeline instance with default configuration")
    return _pipeline


def init_pipeline(config: PipelineConfig | None = None) -> None:
    """Initialize pipeline with configuration.

    Args:
        config: Pipeline configuration.
    """
    global _pipeline
    _pipeline = create_pipeline(config)
    logger.info("Initialized pipeline")


@router.get("/status", response_model=PipelineStatusResponse)
async def get_status() -> PipelineStatusResponse:
    """Get current pipeline status.

    Returns status including:
    - Current state (idle, running, etc.)
    - Current cycle information
    - Last cycle results
    - Quality thresholds
    """
    pipeline = get_pipeline()
    status = pipeline.get_status()

    last_cycle = status.get("last_cycle")
    last_evasion_rate = None
    last_decision = None

    if last_cycle:
        attack_result = last_cycle.get("attack_result")
        quality_decision = last_cycle.get("quality_decision")

        if attack_result:
            last_evasion_rate = attack_result.get("evasion_rate")
        if quality_decision:
            last_decision = quality_decision.get("decision")

    return PipelineStatusResponse(
        state=status.get("state", "unknown"),
        current_cycle_id=status.get("current_cycle", {}).get("cycle_id") if status.get("current_cycle") else None,
        total_cycles=status.get("total_cycles", 0),
        last_evasion_rate=last_evasion_rate,
        last_decision=last_decision,
        thresholds=status.get("quality_thresholds", {}),
    )


@router.post("/trigger", response_model=TriggerResponse)
async def trigger_pipeline(
    request: TriggerRequest,
    background_tasks: BackgroundTasks,
) -> TriggerResponse:
    """Manually trigger a pipeline cycle.

    Args:
        request: Trigger request with optional cycle_id and async mode.
        background_tasks: FastAPI background tasks.

    Returns:
        Response with cycle information.
    """
    global _running_task

    pipeline = get_pipeline()

    # Check if already running
    if pipeline.state not in (PipelineState.IDLE, PipelineState.COMPLETED, PipelineState.FAILED):
        return TriggerResponse(
            success=False,
            message=f"Pipeline already running (state: {pipeline.state.value})",
            cycle_id=None,
            state=pipeline.state.value,
        )

    if request.async_mode:
        # Run in background
        async def run_cycle() -> None:
            await pipeline.run_cycle(request.cycle_id)

        _running_task = asyncio.create_task(run_cycle())

        return TriggerResponse(
            success=True,
            message="Pipeline cycle started in background",
            cycle_id=request.cycle_id,
            state="starting",
        )
    else:
        # Run synchronously (may timeout for long cycles)
        try:
            result = await asyncio.wait_for(
                pipeline.run_cycle(request.cycle_id),
                timeout=300.0,  # 5 minute timeout
            )
            return TriggerResponse(
                success=result.success,
                message=f"Cycle completed: {result.state.value}",
                cycle_id=result.cycle_id,
                state=result.state.value,
            )
        except asyncio.TimeoutError:
            return TriggerResponse(
                success=False,
                message="Cycle timed out (still running in background)",
                cycle_id=request.cycle_id,
                state="timeout",
            )


@router.get("/history", response_model=CycleHistoryResponse)
async def get_history(
    limit: int = Query(default=10, ge=1, le=100, description="Number of cycles to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
) -> CycleHistoryResponse:
    """Get pipeline execution history.

    Args:
        limit: Maximum number of cycles to return.
        offset: Offset for pagination.

    Returns:
        Historical cycle results.
    """
    pipeline = get_pipeline()
    history = pipeline.load_history()

    # Apply pagination
    paginated = history[offset:offset + limit]

    return CycleHistoryResponse(
        total_cycles=len(history),
        cycles=paginated,
    )


@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics() -> MetricsResponse:
    """Get pipeline metrics for monitoring.

    Returns aggregated metrics from recent cycles including:
    - Evasion rate statistics
    - Retraining frequency
    - Success rate
    """
    pipeline = get_pipeline()
    metrics = pipeline.get_metrics()

    return MetricsResponse(
        total_cycles=metrics.get("total_cycles", 0),
        avg_evasion_rate=metrics.get("avg_evasion_rate", 0.0),
        max_evasion_rate=metrics.get("max_evasion_rate", 0.0),
        min_evasion_rate=metrics.get("min_evasion_rate", 0.0),
        retraining_count=metrics.get("retraining_count", 0),
        success_rate=metrics.get("success_rate", 0.0),
        total_samples_collected=metrics.get("total_samples_collected", 0),
    )


@router.put("/config")
async def update_config(request: ConfigUpdateRequest) -> dict[str, Any]:
    """Update pipeline configuration.

    Allows updating quality gate thresholds and attack settings.

    Args:
        request: Configuration updates.

    Returns:
        Updated configuration.
    """
    pipeline = get_pipeline()

    updates: dict[str, Any] = {}

    # Update quality gate thresholds
    if request.max_evasion_rate is not None:
        pipeline._quality_gate.config.max_evasion_rate = request.max_evasion_rate
        updates["max_evasion_rate"] = request.max_evasion_rate

    if request.min_f1_score is not None:
        pipeline._quality_gate.config.min_f1_score = request.min_f1_score
        updates["min_f1_score"] = request.min_f1_score

    # Update attack settings
    if request.attack_batch_size is not None:
        pipeline.config.attack.batch_size = request.attack_batch_size
        updates["attack_batch_size"] = request.attack_batch_size

    if request.num_variants is not None:
        pipeline.config.attack.num_variants = request.num_variants
        updates["num_variants"] = request.num_variants

    logger.info(f"Updated pipeline configuration: {updates}")

    return {
        "success": True,
        "updates": updates,
        "current_thresholds": pipeline._quality_gate.get_threshold_summary(),
    }


@router.get("/config")
async def get_config() -> dict[str, Any]:
    """Get current pipeline configuration.

    Returns:
        Current configuration values.
    """
    pipeline = get_pipeline()

    return {
        "attack": {
            "batch_size": pipeline.config.attack.batch_size,
            "num_variants": pipeline.config.attack.num_variants,
            "include_llm": pipeline.config.attack.include_llm_strategies,
        },
        "quality_gate": pipeline._quality_gate.get_threshold_summary(),
        "retrain": {
            "min_failed_samples": pipeline.config.retrain.min_failed_samples,
            "augmentation_multiplier": pipeline.config.retrain.augmentation_multiplier,
        },
        "evaluation": {
            "traffic_ratio": pipeline.config.evaluation.traffic_ratio,
            "min_samples": pipeline.config.evaluation.min_samples,
            "min_improvement": pipeline.config.evaluation.min_improvement,
        },
    }


@router.get("/samples/statistics")
async def get_sample_statistics() -> dict[str, Any]:
    """Get statistics about collected failed samples.

    Returns:
        Sample collection statistics.
    """
    pipeline = get_pipeline()
    return pipeline._sample_collector.get_statistics()


@router.delete("/samples")
async def clear_samples() -> dict[str, Any]:
    """Clear collected samples from memory.

    Returns:
        Number of samples cleared.
    """
    pipeline = get_pipeline()
    count = pipeline._sample_collector.clear()
    return {
        "success": True,
        "samples_cleared": count,
    }


@router.get("/health")
async def health_check() -> dict[str, Any]:
    """Health check endpoint.

    Returns:
        Health status.
    """
    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "pipeline_state": pipeline.state.value,
        "timestamp": datetime.now(UTC).isoformat(),
    }


# Export router for inclusion in main app
__all__ = ["router", "init_pipeline", "get_pipeline"]
