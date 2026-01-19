"""Scheduler for continuous pipeline execution.

This module provides scheduling capabilities for running the pipeline
continuously over a specified duration or indefinitely.
"""

from __future__ import annotations

import asyncio
import logging
import signal
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Callable

from ml_service.pipeline.config import PipelineConfig
from ml_service.pipeline.orchestrator import (
    AdversarialPipelineOrchestrator,
    CycleResult,
    PipelineState,
)

logger = logging.getLogger(__name__)


class SchedulerState(Enum):
    """Scheduler state."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


@dataclass
class SchedulerStats:
    """Statistics for scheduler execution."""

    started_at: datetime | None = None
    stopped_at: datetime | None = None
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    retraining_triggered: int = 0
    total_evasions: int = 0
    total_samples_collected: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "total_cycles": self.total_cycles,
            "successful_cycles": self.successful_cycles,
            "failed_cycles": self.failed_cycles,
            "retraining_triggered": self.retraining_triggered,
            "total_evasions": self.total_evasions,
            "total_samples_collected": self.total_samples_collected,
            "success_rate": self.successful_cycles / self.total_cycles if self.total_cycles > 0 else 0,
        }


class PipelineScheduler:
    """Scheduler for running pipeline continuously.

    Supports:
    - Running for a fixed duration (N hours)
    - Running indefinitely until stopped
    - Configurable interval between cycles
    - Graceful shutdown handling

    Example:
        >>> scheduler = PipelineScheduler(pipeline, interval_minutes=30)
        >>> await scheduler.run_for_duration(hours=8)  # Run for 8 hours

        >>> # Or run indefinitely
        >>> await scheduler.start()
        >>> # Later...
        >>> scheduler.stop()
    """

    def __init__(
        self,
        pipeline: AdversarialPipelineOrchestrator,
        interval_minutes: int = 60,
        on_cycle_complete: Callable[[CycleResult], None] | None = None,
    ) -> None:
        """Initialize scheduler.

        Args:
            pipeline: Pipeline orchestrator to run.
            interval_minutes: Minutes between cycles.
            on_cycle_complete: Optional callback after each cycle.
        """
        self.pipeline = pipeline
        self.interval_minutes = interval_minutes
        self.on_cycle_complete = on_cycle_complete

        self._state = SchedulerState.STOPPED
        self._stats = SchedulerStats()
        self._stop_event = asyncio.Event()
        self._current_task: asyncio.Task | None = None

    @property
    def state(self) -> SchedulerState:
        """Get current scheduler state."""
        return self._state

    @property
    def stats(self) -> SchedulerStats:
        """Get scheduler statistics."""
        return self._stats

    async def run_for_duration(
        self,
        hours: float | None = None,
        minutes: float | None = None,
    ) -> SchedulerStats:
        """Run pipeline for a specified duration.

        Args:
            hours: Duration in hours.
            minutes: Duration in minutes (added to hours).

        Returns:
            SchedulerStats with execution summary.
        """
        total_minutes = (hours or 0) * 60 + (minutes or 0)
        if total_minutes <= 0:
            raise ValueError("Duration must be positive")

        end_time = datetime.now(UTC) + timedelta(minutes=total_minutes)

        logger.info(
            f"Starting scheduler for {total_minutes:.0f} minutes "
            f"(until {end_time.isoformat()})"
        )

        return await self._run_until(end_time)

    async def start(self) -> None:
        """Start running indefinitely.

        Use stop() to stop the scheduler.
        """
        if self._state == SchedulerState.RUNNING:
            logger.warning("Scheduler already running")
            return

        self._stop_event.clear()
        self._current_task = asyncio.create_task(self._run_until(None))

    def stop(self) -> None:
        """Signal the scheduler to stop after current cycle."""
        if self._state != SchedulerState.RUNNING:
            return

        logger.info("Stopping scheduler after current cycle...")
        self._state = SchedulerState.STOPPING
        self._stop_event.set()

    def stop_immediately(self) -> None:
        """Stop the scheduler immediately, cancelling current cycle."""
        self.stop()
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()

    async def wait(self) -> SchedulerStats:
        """Wait for scheduler to complete.

        Returns:
            Final statistics.
        """
        if self._current_task:
            try:
                await self._current_task
            except asyncio.CancelledError:
                pass
        return self._stats

    async def _run_until(self, end_time: datetime | None) -> SchedulerStats:
        """Internal method to run until a specific time or indefinitely.

        Args:
            end_time: When to stop (None for indefinitely).

        Returns:
            SchedulerStats.
        """
        self._state = SchedulerState.RUNNING
        self._stats = SchedulerStats(started_at=datetime.now(UTC))

        cycle_num = 0

        try:
            while not self._stop_event.is_set():
                # Check if we've reached the end time
                if end_time and datetime.now(UTC) >= end_time:
                    logger.info("Duration reached, stopping scheduler")
                    break

                cycle_num += 1
                cycle_id = f"sched_{cycle_num:04d}"

                logger.info(f"Starting scheduled cycle {cycle_num} ({cycle_id})")

                # Run cycle
                try:
                    result = await self.pipeline.run_cycle(cycle_id)
                    self._update_stats(result)

                    if self.on_cycle_complete:
                        self.on_cycle_complete(result)

                except Exception as e:
                    logger.error(f"Cycle {cycle_num} failed: {e}")
                    self._stats.total_cycles += 1
                    self._stats.failed_cycles += 1

                # Check stop signal before waiting
                if self._stop_event.is_set():
                    break

                # Calculate wait time
                wait_seconds = self.interval_minutes * 60

                # Adjust if we have an end time
                if end_time:
                    remaining = (end_time - datetime.now(UTC)).total_seconds()
                    if remaining <= 0:
                        break
                    wait_seconds = min(wait_seconds, remaining)

                logger.info(f"Next cycle in {wait_seconds / 60:.1f} minutes")

                # Wait with stop event
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=wait_seconds,
                    )
                    # If we get here, stop was requested
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, continue to next cycle
                    pass

        finally:
            self._stats.stopped_at = datetime.now(UTC)
            self._state = SchedulerState.STOPPED

            duration = self._stats.stopped_at - self._stats.started_at
            logger.info(
                f"Scheduler stopped after {duration.total_seconds() / 3600:.1f} hours, "
                f"{self._stats.total_cycles} cycles "
                f"({self._stats.successful_cycles} successful, {self._stats.failed_cycles} failed)"
            )

        return self._stats

    def _update_stats(self, result: CycleResult) -> None:
        """Update statistics from cycle result."""
        self._stats.total_cycles += 1

        if result.success:
            self._stats.successful_cycles += 1
        else:
            self._stats.failed_cycles += 1

        if result.retraining_triggered:
            self._stats.retraining_triggered += 1

        if result.attack_result:
            self._stats.total_evasions += result.attack_result.total_evasions

        self._stats.total_samples_collected += result.samples_collected

    def get_status(self) -> dict[str, Any]:
        """Get scheduler status.

        Returns:
            Status dictionary.
        """
        elapsed = None
        if self._stats.started_at:
            end = self._stats.stopped_at or datetime.now(UTC)
            elapsed = (end - self._stats.started_at).total_seconds()

        return {
            "state": self._state.value,
            "interval_minutes": self.interval_minutes,
            "elapsed_seconds": elapsed,
            "elapsed_hours": elapsed / 3600 if elapsed else None,
            "stats": self._stats.to_dict(),
        }


def create_scheduled_pipeline(
    config: PipelineConfig | None = None,
    interval_minutes: int = 60,
) -> tuple[AdversarialPipelineOrchestrator, PipelineScheduler]:
    """Create a pipeline with scheduler.

    Args:
        config: Pipeline configuration.
        interval_minutes: Minutes between cycles.

    Returns:
        Tuple of (pipeline, scheduler).
    """
    from ml_service.pipeline.orchestrator import create_pipeline

    pipeline = create_pipeline(config)
    scheduler = PipelineScheduler(pipeline, interval_minutes)

    return pipeline, scheduler


async def run_continuous(
    hours: float,
    interval_minutes: int = 60,
    config: PipelineConfig | None = None,
) -> SchedulerStats:
    """Convenience function to run pipeline for N hours.

    Args:
        hours: Duration in hours.
        interval_minutes: Minutes between cycles.
        config: Pipeline configuration.

    Returns:
        SchedulerStats.
    """
    pipeline, scheduler = create_scheduled_pipeline(config, interval_minutes)
    return await scheduler.run_for_duration(hours=hours)
