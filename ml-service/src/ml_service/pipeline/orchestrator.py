"""Main orchestrator for Adversarial MLOps Pipeline.

This module coordinates the entire pipeline:
1. Run adversarial attacks
2. Evaluate quality through quality gate
3. Collect failed samples if threshold exceeded
4. Augment data and trigger retraining if needed
5. Promote model through Champion/Challenger evaluation
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol
import uuid

from ml_service.pipeline.config import PipelineConfig
from ml_service.pipeline.attack_runner import AttackRunner, AttackBatchResult
from ml_service.pipeline.quality_gate import QualityGate, QualityGateDecision, DecisionType
from ml_service.pipeline.sample_collector import FailedSampleCollector
from ml_service.pipeline.data_augmentor import TrainingDataAugmentor, AugmentedDataset
from ml_service.pipeline.model_promoter import (
    ModelPromoter,
    SimpleModelPromoter,
    PromotionDecision,
    PromotionStatus,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ml_service.training.trainer import QLoRATrainer


class ClassifierProtocol(Protocol):
    """Protocol for classifier models."""

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict labels for texts."""
        ...


class PipelineState(Enum):
    """State of the pipeline."""

    IDLE = "idle"
    RUNNING_ATTACK = "running_attack"
    EVALUATING_QUALITY = "evaluating_quality"
    COLLECTING_SAMPLES = "collecting_samples"
    AUGMENTING_DATA = "augmenting_data"
    RETRAINING = "retraining"
    PROMOTING = "promoting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CycleResult:
    """Result of a single pipeline cycle."""

    cycle_id: str
    started_at: datetime
    completed_at: datetime | None
    state: PipelineState
    attack_result: AttackBatchResult | None
    quality_decision: QualityGateDecision | None
    samples_collected: int
    retraining_triggered: bool
    retraining_metrics: dict[str, float] | None
    promotion_decision: PromotionDecision | None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cycle_id": self.cycle_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "state": self.state.value,
            "attack_result": self.attack_result.to_dict() if self.attack_result else None,
            "quality_decision": self.quality_decision.to_dict() if self.quality_decision else None,
            "samples_collected": self.samples_collected,
            "retraining_triggered": self.retraining_triggered,
            "retraining_metrics": self.retraining_metrics,
            "promotion_decision": self.promotion_decision.to_dict() if self.promotion_decision else None,
            "error": self.error,
            "metadata": self.metadata,
        }

    @property
    def success(self) -> bool:
        """Check if cycle completed successfully."""
        return self.state == PipelineState.COMPLETED and self.error is None

    @property
    def duration_seconds(self) -> float:
        """Get cycle duration in seconds."""
        if self.completed_at is None:
            return (datetime.now(UTC) - self.started_at).total_seconds()
        return (self.completed_at - self.started_at).total_seconds()


class AdversarialPipelineOrchestrator:
    """Main orchestrator for the Adversarial MLOps Pipeline.

    Coordinates the full pipeline cycle:
    1. Attack Phase: Run adversarial attacks against current model
    2. Quality Gate: Evaluate if model meets quality thresholds
    3. Collection: Collect failed samples if threshold exceeded
    4. Retraining: Augment data and retrain if needed
    5. Promotion: Evaluate and promote challenger through A/B testing

    Example:
        >>> config = PipelineConfig()
        >>> orchestrator = AdversarialPipelineOrchestrator(config, classifier)
        >>> result = await orchestrator.run_cycle()
        >>> if result.retraining_triggered:
        ...     print(f"Retrained model with {result.samples_collected} samples")
    """

    def __init__(
        self,
        config: PipelineConfig,
        classifier: ClassifierProtocol,
    ) -> None:
        """Initialize pipeline orchestrator.

        Args:
            config: Pipeline configuration.
            classifier: Classifier model to attack and evaluate.
        """
        self.config = config
        self.classifier = classifier
        self._state = PipelineState.IDLE
        self._current_cycle: CycleResult | None = None
        self._history: list[CycleResult] = []

        # Initialize components
        self._attack_runner = AttackRunner(config.attack, classifier)
        self._quality_gate = QualityGate(config.quality_gate)
        self._sample_collector = FailedSampleCollector(config.storage)
        self._data_augmentor = TrainingDataAugmentor(config.retrain, config.storage)
        self._model_promoter = SimpleModelPromoter(config.evaluation, config.mlflow)

        # Ensure storage directories exist
        config.storage.failed_samples_dir.mkdir(parents=True, exist_ok=True)
        config.storage.augmented_data_dir.mkdir(parents=True, exist_ok=True)
        config.storage.model_output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def state(self) -> PipelineState:
        """Get current pipeline state."""
        return self._state

    def _set_state(self, state: PipelineState) -> None:
        """Set pipeline state with logging."""
        old_state = self._state
        self._state = state
        if self.config.verbose:
            logger.info(f"Pipeline state: {old_state.value} -> {state.value}")

    async def run_cycle(self, cycle_id: str | None = None) -> CycleResult:
        """Run a complete pipeline cycle.

        Args:
            cycle_id: Optional cycle identifier.

        Returns:
            CycleResult with all results.
        """
        cycle_id = cycle_id or str(uuid.uuid4())[:8]
        started_at = datetime.now(UTC)

        logger.info(f"Starting pipeline cycle: {cycle_id}")

        # Initialize result
        result = CycleResult(
            cycle_id=cycle_id,
            started_at=started_at,
            completed_at=None,
            state=PipelineState.RUNNING_ATTACK,
            attack_result=None,
            quality_decision=None,
            samples_collected=0,
            retraining_triggered=False,
            retraining_metrics=None,
            promotion_decision=None,
        )

        self._current_cycle = result

        try:
            # Phase 1: Run adversarial attacks
            self._set_state(PipelineState.RUNNING_ATTACK)
            attack_result = await self._run_attacks()
            result.attack_result = attack_result

            # Phase 2: Evaluate quality
            self._set_state(PipelineState.EVALUATING_QUALITY)
            quality_decision = self._evaluate_quality(attack_result)
            result.quality_decision = quality_decision

            # Phase 3: Collect samples if quality failed
            if quality_decision.needs_retraining:
                self._set_state(PipelineState.COLLECTING_SAMPLES)
                samples_collected = self._collect_samples(attack_result)
                result.samples_collected = samples_collected

                # Phase 4: Check if enough samples for retraining
                if samples_collected >= self.config.retrain.min_failed_samples:
                    self._set_state(PipelineState.AUGMENTING_DATA)
                    augmented = await self._augment_and_retrain(cycle_id)
                    result.retraining_triggered = True
                    result.retraining_metrics = augmented

                    # Phase 5: Evaluate for promotion
                    if augmented:
                        self._set_state(PipelineState.PROMOTING)
                        promotion = self._evaluate_promotion(augmented)
                        result.promotion_decision = promotion
                else:
                    logger.info(
                        f"Only {samples_collected} failed samples, "
                        f"need {self.config.retrain.min_failed_samples} for retraining"
                    )

            # Mark completed
            self._set_state(PipelineState.COMPLETED)
            result.state = PipelineState.COMPLETED
            result.completed_at = datetime.now(UTC)

        except Exception as e:
            logger.error(f"Pipeline cycle failed: {e}", exc_info=True)
            self._set_state(PipelineState.FAILED)
            result.state = PipelineState.FAILED
            result.error = str(e)
            result.completed_at = datetime.now(UTC)

        # Save to history
        self._history.append(result)
        self._save_history()

        evasion_str = f"{result.attack_result.evasion_rate:.1%}" if result.attack_result else "N/A"
        logger.info(
            f"Pipeline cycle {cycle_id} completed: "
            f"state={result.state.value}, "
            f"evasion_rate={evasion_str}, "
            f"retraining={result.retraining_triggered}"
        )

        return result

    async def _run_attacks(self) -> AttackBatchResult:
        """Run attack phase."""
        logger.info("Running adversarial attacks...")
        return await self._attack_runner.run_batch_async()

    def _evaluate_quality(self, attack_result: AttackBatchResult) -> QualityGateDecision:
        """Evaluate quality gate."""
        logger.info("Evaluating quality gate...")
        return self._quality_gate.evaluate(attack_result)

    def _collect_samples(self, attack_result: AttackBatchResult) -> int:
        """Collect failed samples."""
        logger.info("Collecting failed samples...")
        collection = self._sample_collector.collect(attack_result)
        return collection.total_collected

    async def _augment_and_retrain(self, cycle_id: str) -> dict[str, float] | None:
        """Augment data and retrain model."""
        logger.info("Augmenting data and retraining...")

        # Get collected samples
        samples = self._sample_collector.get_samples()

        # Augment data
        augmented = self._data_augmentor.augment(samples)
        self._data_augmentor.save(augmented, cycle_id)

        # Save collected samples
        self._sample_collector.save(cycle_id)

        # Try to retrain
        try:
            metrics = await self._retrain_model(augmented, cycle_id)
            return metrics
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            return None

    async def _retrain_model(
        self,
        augmented: AugmentedDataset,
        cycle_id: str,
    ) -> dict[str, float]:
        """Retrain the model with augmented data.

        Args:
            augmented: Augmented dataset.
            cycle_id: Cycle identifier.

        Returns:
            Training metrics.
        """
        try:
            from ml_service.training.trainer import QLoRATrainer
            from ml_service.training.config import TrainingConfig
        except ImportError:
            logger.warning("Training dependencies not available, skipping retraining")
            return {"status": "skipped", "reason": "dependencies_missing"}

        # Prepare training config
        training_config = TrainingConfig(
            model_name=self.config.model_name,
            output_dir=self.config.storage.model_output_dir / cycle_id,
            mlflow_experiment_name=self.config.mlflow.experiment_name,
            mlflow_tracking_uri=self.config.mlflow.tracking_uri,
        )

        # Prepare data
        tokenized = self._data_augmentor.prepare_for_training(
            augmented,
            tokenizer_name=self.config.model_name,
        )

        # Train
        trainer = QLoRATrainer(training_config)
        trainer.setup_model()

        # Run training in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: trainer.train(tokenized, run_name=f"pipeline_{cycle_id}"),
        )

        return result["eval_metrics"]

    def _evaluate_promotion(self, metrics: dict[str, float]) -> PromotionDecision:
        """Evaluate model for promotion."""
        logger.info("Evaluating model for promotion...")

        # Set challenger metrics from training
        self._model_promoter.set_challenger_metrics(
            metrics,
            version=self._current_cycle.cycle_id if self._current_cycle else "unknown",
        )

        return self._model_promoter.evaluate()

    def _save_history(self) -> None:
        """Save pipeline history to file."""
        history_path = self.config.storage.history_file
        history_path.parent.mkdir(parents=True, exist_ok=True)

        history_data = [r.to_dict() for r in self._history[-100:]]  # Keep last 100

        with open(history_path, "w") as f:
            json.dump(history_data, f, indent=2)

    def load_history(self) -> list[dict[str, Any]]:
        """Load pipeline history from file.

        Returns:
            List of historical cycle results.
        """
        history_path = self.config.storage.history_file
        if not history_path.exists():
            return []

        with open(history_path) as f:
            return json.load(f)

    def get_status(self) -> dict[str, Any]:
        """Get current pipeline status.

        Returns:
            Dictionary with status information.
        """
        return {
            "state": self._state.value,
            "current_cycle": self._current_cycle.to_dict() if self._current_cycle else None,
            "total_cycles": len(self._history),
            "last_cycle": self._history[-1].to_dict() if self._history else None,
            "quality_thresholds": self._quality_gate.get_threshold_summary(),
            "sample_statistics": self._sample_collector.get_statistics(),
        }

    def get_metrics(self) -> dict[str, Any]:
        """Get pipeline metrics for monitoring.

        Returns:
            Dictionary of metrics.
        """
        if not self._history:
            return {}

        # Calculate aggregates from recent cycles
        recent = self._history[-10:]

        evasion_rates = [
            r.attack_result.evasion_rate
            for r in recent
            if r.attack_result
        ]
        retraining_count = sum(1 for r in recent if r.retraining_triggered)
        success_count = sum(1 for r in recent if r.success)

        return {
            "total_cycles": len(self._history),
            "recent_cycles": len(recent),
            "avg_evasion_rate": sum(evasion_rates) / len(evasion_rates) if evasion_rates else 0,
            "max_evasion_rate": max(evasion_rates) if evasion_rates else 0,
            "min_evasion_rate": min(evasion_rates) if evasion_rates else 0,
            "retraining_count": retraining_count,
            "success_rate": success_count / len(recent) if recent else 0,
            "total_samples_collected": sum(r.samples_collected for r in self._history),
        }

    def trigger(self) -> CycleResult:
        """Trigger a pipeline cycle synchronously.

        Returns:
            CycleResult.
        """
        return asyncio.run(self.run_cycle())


def create_pipeline(
    config: PipelineConfig | None = None,
    classifier: ClassifierProtocol | None = None,
) -> AdversarialPipelineOrchestrator:
    """Create a pipeline orchestrator with default or provided config.

    Args:
        config: Pipeline configuration.
        classifier: Classifier model.

    Returns:
        Configured AdversarialPipelineOrchestrator.
    """
    if config is None:
        config = PipelineConfig()

    if classifier is None:
        from ml_service.pipeline.attack_runner import MockClassifier
        classifier = MockClassifier(evasion_rate=0.15)
        logger.warning("Using MockClassifier - replace with real model for production")

    return AdversarialPipelineOrchestrator(config, classifier)
