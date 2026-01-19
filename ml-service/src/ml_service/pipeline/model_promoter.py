"""Model promoter for adversarial pipeline.

This module handles Champion/Challenger model evaluation and promotion
using A/B testing and statistical significance testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from ml_service.pipeline.config import EvaluationConfig, MLflowConfig

logger = logging.getLogger(__name__)


class PromotionStatus(Enum):
    """Status of promotion decision."""

    PENDING = "pending"  # Not enough data yet
    PROMOTE = "promote"  # Challenger significantly better
    KEEP = "keep"  # Champion significantly better or no difference
    TIMEOUT = "timeout"  # Max evaluation time reached


@dataclass
class PromotionDecision:
    """Result of promotion evaluation."""

    status: PromotionStatus
    champion_version: str | int | None
    challenger_version: str | int | None
    champion_accuracy: float
    challenger_accuracy: float
    improvement: float
    is_significant: bool
    p_value: float | None
    samples_evaluated: int
    timestamp: datetime
    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "status": self.status.value,
            "champion_version": self.champion_version,
            "challenger_version": self.challenger_version,
            "champion_accuracy": self.champion_accuracy,
            "challenger_accuracy": self.challenger_accuracy,
            "improvement": self.improvement,
            "is_significant": self.is_significant,
            "p_value": self.p_value,
            "samples_evaluated": self.samples_evaluated,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "metadata": self.metadata,
        }


class ModelPromoter:
    """Manages Champion/Challenger model evaluation and promotion.

    Uses A/B testing to compare model performance and promotes the
    challenger to champion if it shows statistically significant improvement.

    Example:
        >>> promoter = ModelPromoter(eval_config, mlflow_config)
        >>> promoter.register_challenger(model_path, "v2")
        >>> decision = promoter.evaluate()
        >>> if decision.status == PromotionStatus.PROMOTE:
        ...     promoter.promote_challenger()
    """

    def __init__(
        self,
        eval_config: EvaluationConfig,
        mlflow_config: MLflowConfig,
    ) -> None:
        """Initialize model promoter.

        Args:
            eval_config: Evaluation configuration.
            mlflow_config: MLflow configuration.
        """
        self.eval_config = eval_config
        self.mlflow_config = mlflow_config
        self._experiment = None
        self._challenger_version: str | int | None = None
        self._champion_version: str | int | None = None
        self._mlflow_client = None

    def _get_mlflow_client(self) -> Any:
        """Get MLflow client."""
        if self._mlflow_client is None:
            try:
                import mlflow
                mlflow.set_tracking_uri(self.mlflow_config.tracking_uri)
                self._mlflow_client = mlflow.MlflowClient()
            except ImportError:
                logger.warning("MLflow not available")
                self._mlflow_client = None
        return self._mlflow_client

    def _get_experiment(self) -> Any:
        """Get or create A/B experiment."""
        if self._experiment is None:
            from ml_service.monitoring.experimental.ab_testing import ABExperiment

            experiment_id = f"pipeline_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}"
            self._experiment = ABExperiment(
                experiment_id=experiment_id,
                challenger_ratio=self.eval_config.traffic_ratio,
                min_samples_per_variant=self.eval_config.min_samples,
                significance_level=self.eval_config.significance_level,
                auto_stop=True,
                max_duration_hours=self.eval_config.max_evaluation_hours,
            )
        return self._experiment

    def register_challenger(
        self,
        model_path: str,
        version: str | int,
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Register a new challenger model.

        Args:
            model_path: Path to the model.
            version: Version identifier.
            metrics: Optional training metrics.
        """
        self._challenger_version = version

        # Register in MLflow if available
        client = self._get_mlflow_client()
        if client is not None:
            try:
                from ml_service.training.mlflow_utils import MLflowTracker
                from ml_service.training.config import TrainingConfig

                config = TrainingConfig(
                    mlflow_experiment_name=self.mlflow_config.experiment_name,
                    mlflow_tracking_uri=self.mlflow_config.tracking_uri,
                )
                tracker = MLflowTracker(config)
                tracker.register_challenger(self.mlflow_config.model_name, version)
            except Exception as e:
                logger.warning(f"Failed to register challenger in MLflow: {e}")

        # Reset experiment for new evaluation
        self._experiment = None

        logger.info(f"Registered challenger model version {version}")

    def start_evaluation(self) -> None:
        """Start A/B testing evaluation."""
        experiment = self._get_experiment()
        experiment.start()
        logger.info(
            f"Started A/B evaluation: {self.eval_config.traffic_ratio:.0%} "
            f"traffic to challenger"
        )

    def record_prediction(
        self,
        is_challenger: bool,
        prediction: int,
        confidence: float,
        latency_ms: float,
        actual_label: int | None = None,
    ) -> None:
        """Record a prediction result for evaluation.

        Args:
            is_challenger: Whether this was from the challenger model.
            prediction: Model prediction.
            confidence: Prediction confidence.
            latency_ms: Prediction latency.
            actual_label: Actual label if known.
        """
        experiment = self._get_experiment()

        from ml_service.monitoring.experimental.ab_testing import VariantAssignment

        variant = (
            VariantAssignment.CHALLENGER if is_challenger
            else VariantAssignment.CHAMPION
        )

        experiment.record_prediction(
            variant=variant,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            actual_label=actual_label,
        )

    def evaluate(self) -> PromotionDecision:
        """Evaluate current A/B test results.

        Returns:
            PromotionDecision with current status.
        """
        experiment = self._get_experiment()
        result = experiment.get_result()

        timestamp = datetime.now(UTC)

        champion_acc = result.champion_metrics.accuracy
        challenger_acc = result.challenger_metrics.accuracy
        improvement = challenger_acc - champion_acc

        # Get statistical test result
        stat_result = result.statistical_test
        is_significant = stat_result.is_significant if stat_result else False
        p_value = stat_result.p_value if stat_result else None

        total_samples = (
            result.champion_metrics.total_predictions +
            result.challenger_metrics.total_predictions
        )

        # Determine status
        if total_samples < self.eval_config.min_samples * 2:
            status = PromotionStatus.PENDING
            message = f"Need more samples ({total_samples}/{self.eval_config.min_samples * 2})"
        elif is_significant and stat_result.challenger_better:
            if improvement >= self.eval_config.min_improvement:
                status = PromotionStatus.PROMOTE
                message = f"Challenger significantly better by {improvement:.1%}"
            else:
                status = PromotionStatus.KEEP
                message = f"Improvement {improvement:.1%} below threshold {self.eval_config.min_improvement:.1%}"
        elif is_significant and not stat_result.challenger_better:
            status = PromotionStatus.KEEP
            message = "Champion significantly better"
        else:
            status = PromotionStatus.PENDING
            message = "No significant difference yet"

        return PromotionDecision(
            status=status,
            champion_version=self._champion_version,
            challenger_version=self._challenger_version,
            champion_accuracy=champion_acc,
            challenger_accuracy=challenger_acc,
            improvement=improvement,
            is_significant=is_significant,
            p_value=p_value,
            samples_evaluated=total_samples,
            timestamp=timestamp,
            message=message,
            metadata={
                "champion_predictions": result.champion_metrics.total_predictions,
                "challenger_predictions": result.challenger_metrics.total_predictions,
                "champion_f1": result.champion_metrics.f1_score,
                "challenger_f1": result.challenger_metrics.f1_score,
            },
        )

    def promote_challenger(self) -> bool:
        """Promote challenger to champion.

        Returns:
            True if promotion successful.
        """
        if self._challenger_version is None:
            logger.error("No challenger version registered")
            return False

        # Update MLflow aliases
        client = self._get_mlflow_client()
        if client is not None:
            try:
                from ml_service.training.mlflow_utils import MLflowTracker
                from ml_service.training.config import TrainingConfig

                config = TrainingConfig(
                    mlflow_experiment_name=self.mlflow_config.experiment_name,
                    mlflow_tracking_uri=self.mlflow_config.tracking_uri,
                )
                tracker = MLflowTracker(config)
                tracker.promote_to_champion(
                    self.mlflow_config.model_name,
                    self._challenger_version,
                )
            except Exception as e:
                logger.warning(f"Failed to promote in MLflow: {e}")

        # Update internal state
        old_champion = self._champion_version
        self._champion_version = self._challenger_version
        self._challenger_version = None

        logger.info(
            f"Promoted challenger {self._champion_version} to champion "
            f"(previous: {old_champion})"
        )

        return True

    def stop_evaluation(self) -> None:
        """Stop current A/B evaluation."""
        if self._experiment is not None:
            from ml_service.monitoring.experimental.ab_testing import ExperimentStatus
            self._experiment.stop(ExperimentStatus.COMPLETED)
            logger.info("Stopped A/B evaluation")

    def get_status(self) -> dict[str, Any]:
        """Get current promotion status.

        Returns:
            Dictionary of status information.
        """
        experiment = self._get_experiment() if self._experiment else None

        return {
            "champion_version": self._champion_version,
            "challenger_version": self._challenger_version,
            "evaluation_active": experiment.status.value if experiment else "none",
            "traffic_ratio": self.eval_config.traffic_ratio,
            "min_samples": self.eval_config.min_samples,
            "min_improvement": self.eval_config.min_improvement,
        }


class SimpleModelPromoter:
    """Simplified model promoter without A/B testing.

    Uses direct metric comparison for promotion decisions.
    Useful when A/B testing is not feasible or desired.
    """

    def __init__(
        self,
        eval_config: EvaluationConfig,
        mlflow_config: MLflowConfig,
    ) -> None:
        """Initialize simple promoter.

        Args:
            eval_config: Evaluation configuration.
            mlflow_config: MLflow configuration.
        """
        self.eval_config = eval_config
        self.mlflow_config = mlflow_config
        self._champion_metrics: dict[str, float] | None = None
        self._challenger_metrics: dict[str, float] | None = None
        self._champion_version: str | int | None = None
        self._challenger_version: str | int | None = None

    def set_champion_metrics(
        self,
        metrics: dict[str, float],
        version: str | int,
    ) -> None:
        """Set champion model metrics.

        Args:
            metrics: Champion metrics (f1, precision, recall, etc.).
            version: Champion version.
        """
        self._champion_metrics = metrics
        self._champion_version = version

    def set_challenger_metrics(
        self,
        metrics: dict[str, float],
        version: str | int,
    ) -> None:
        """Set challenger model metrics.

        Args:
            metrics: Challenger metrics.
            version: Challenger version.
        """
        self._challenger_metrics = metrics
        self._challenger_version = version

    def evaluate(self) -> PromotionDecision:
        """Evaluate challenger against champion.

        Returns:
            PromotionDecision based on metric comparison.
        """
        timestamp = datetime.now(UTC)

        if self._champion_metrics is None or self._challenger_metrics is None:
            return PromotionDecision(
                status=PromotionStatus.PENDING,
                champion_version=self._champion_version,
                challenger_version=self._challenger_version,
                champion_accuracy=0.0,
                challenger_accuracy=0.0,
                improvement=0.0,
                is_significant=False,
                p_value=None,
                samples_evaluated=0,
                timestamp=timestamp,
                message="Missing metrics for comparison",
            )

        # Compare F1 scores (primary metric)
        champion_f1 = self._champion_metrics.get("f1", 0.0)
        challenger_f1 = self._challenger_metrics.get("f1", 0.0)
        improvement = challenger_f1 - champion_f1

        # Use accuracy as fallback
        champion_acc = self._champion_metrics.get("accuracy", champion_f1)
        challenger_acc = self._challenger_metrics.get("accuracy", challenger_f1)

        if improvement >= self.eval_config.min_improvement:
            status = PromotionStatus.PROMOTE
            message = f"Challenger F1 {challenger_f1:.3f} vs Champion {champion_f1:.3f} (+{improvement:.3f})"
        elif improvement > 0:
            status = PromotionStatus.KEEP
            message = f"Improvement {improvement:.3f} below threshold {self.eval_config.min_improvement}"
        else:
            status = PromotionStatus.KEEP
            message = f"Champion F1 {champion_f1:.3f} >= Challenger {challenger_f1:.3f}"

        return PromotionDecision(
            status=status,
            champion_version=self._champion_version,
            challenger_version=self._challenger_version,
            champion_accuracy=champion_acc,
            challenger_accuracy=challenger_acc,
            improvement=improvement,
            is_significant=improvement >= self.eval_config.min_improvement,
            p_value=None,
            samples_evaluated=0,
            timestamp=timestamp,
            message=message,
            metadata={
                "champion_metrics": self._champion_metrics,
                "challenger_metrics": self._challenger_metrics,
            },
        )

    def promote_challenger(self) -> bool:
        """Promote challenger to champion.

        Returns:
            True if successful.
        """
        if self._challenger_metrics is None:
            return False

        self._champion_metrics = self._challenger_metrics
        self._champion_version = self._challenger_version
        self._challenger_metrics = None
        self._challenger_version = None

        logger.info(f"Promoted challenger to champion (v{self._champion_version})")
        return True
