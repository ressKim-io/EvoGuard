"""A/B Testing - Model experimentation and traffic splitting.

This module provides A/B testing capabilities for ML models:
- Traffic splitting between champion and challenger models
- Metrics collection per variant
- Statistical significance testing
- Automatic promotion decisions
"""

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Status of an A/B experiment."""

    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class VariantAssignment(Enum):
    """Model variant assignment."""

    CHAMPION = "champion"
    CHALLENGER = "challenger"


@dataclass
class VariantMetrics:
    """Metrics collected for a variant."""

    variant: VariantAssignment
    total_predictions: int = 0
    correct_predictions: int = 0
    total_latency_ms: float = 0.0
    total_confidence: float = 0.0
    positive_predictions: int = 0
    negative_predictions: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def accuracy(self) -> float:
        """Calculate accuracy."""
        if self.total_predictions == 0:
            return 0.0
        return self.correct_predictions / self.total_predictions

    @property
    def precision(self) -> float:
        """Calculate precision."""
        if (self.true_positives + self.false_positives) == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        """Calculate recall."""
        if (self.true_positives + self.false_negatives) == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)

    @property
    def f1_score(self) -> float:
        """Calculate F1 score."""
        if (self.precision + self.recall) == 0:
            return 0.0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def mean_latency_ms(self) -> float:
        """Calculate mean latency."""
        if self.total_predictions == 0:
            return 0.0
        return self.total_latency_ms / self.total_predictions

    @property
    def mean_confidence(self) -> float:
        """Calculate mean confidence."""
        if self.total_predictions == 0:
            return 0.0
        return self.total_confidence / self.total_predictions

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "variant": self.variant.value,
            "total_predictions": self.total_predictions,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "mean_latency_ms": self.mean_latency_ms,
            "mean_confidence": self.mean_confidence,
        }


@dataclass
class StatisticalTestResult:
    """Result of statistical significance test."""

    test_name: str
    is_significant: bool
    p_value: float
    confidence_level: float
    effect_size: float
    challenger_better: bool
    message: str


@dataclass
class ExperimentResult:
    """Result of an A/B experiment."""

    experiment_id: str
    status: ExperimentStatus
    champion_metrics: VariantMetrics
    challenger_metrics: VariantMetrics
    statistical_test: StatisticalTestResult | None
    recommendation: str
    started_at: datetime
    ended_at: datetime | None = None
    duration: timedelta | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "champion_metrics": self.champion_metrics.to_dict(),
            "challenger_metrics": self.challenger_metrics.to_dict(),
            "statistical_test": {
                "test_name": self.statistical_test.test_name,
                "is_significant": self.statistical_test.is_significant,
                "p_value": self.statistical_test.p_value,
                "challenger_better": self.statistical_test.challenger_better,
            }
            if self.statistical_test
            else None,
            "recommendation": self.recommendation,
            "started_at": self.started_at.isoformat(),
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
        }


class TrafficSplitter:
    """Split traffic between variants using consistent hashing.

    Ensures the same user/request always gets the same variant.

    Example:
        >>> splitter = TrafficSplitter(challenger_ratio=0.1)
        >>> variant = splitter.assign("user_123")
        >>> # user_123 will always get the same variant
    """

    def __init__(self, challenger_ratio: float = 0.1) -> None:
        """Initialize the traffic splitter.

        Args:
            challenger_ratio: Ratio of traffic to send to challenger (0.0 to 1.0).
        """
        if not 0.0 <= challenger_ratio <= 1.0:
            raise ValueError("challenger_ratio must be between 0.0 and 1.0")
        self.challenger_ratio = challenger_ratio

    def assign(self, identifier: str) -> VariantAssignment:
        """Assign a variant based on identifier.

        Uses consistent hashing to ensure deterministic assignment.

        Args:
            identifier: Unique identifier (user_id, request_id, etc.).

        Returns:
            VariantAssignment (CHAMPION or CHALLENGER).
        """
        hash_value = int(hashlib.md5(identifier.encode()).hexdigest(), 16)
        normalized = (hash_value % 10000) / 10000.0

        if normalized < self.challenger_ratio:
            return VariantAssignment.CHALLENGER
        return VariantAssignment.CHAMPION

    def update_ratio(self, new_ratio: float) -> None:
        """Update the challenger traffic ratio.

        Args:
            new_ratio: New ratio (0.0 to 1.0).
        """
        if not 0.0 <= new_ratio <= 1.0:
            raise ValueError("new_ratio must be between 0.0 and 1.0")
        self.challenger_ratio = new_ratio


class StatisticalSignificance:
    """Statistical significance testing for A/B experiments."""

    @staticmethod
    def z_test_proportions(
        p1: float,
        n1: int,
        p2: float,
        n2: int,
        alpha: float = 0.05,
    ) -> StatisticalTestResult:
        """Two-proportion z-test for comparing conversion rates.

        Args:
            p1: Proportion for group 1 (champion).
            n1: Sample size for group 1.
            p2: Proportion for group 2 (challenger).
            n2: Sample size for group 2.
            alpha: Significance level.

        Returns:
            StatisticalTestResult with test outcome.
        """
        if n1 == 0 or n2 == 0:
            return StatisticalTestResult(
                test_name="two_proportion_z_test",
                is_significant=False,
                p_value=1.0,
                confidence_level=1 - alpha,
                effect_size=0.0,
                challenger_better=False,
                message="Insufficient samples for statistical test",
            )

        # Pooled proportion
        p_pool = (p1 * n1 + p2 * n2) / (n1 + n2)

        if p_pool == 0 or p_pool == 1:
            return StatisticalTestResult(
                test_name="two_proportion_z_test",
                is_significant=False,
                p_value=1.0,
                confidence_level=1 - alpha,
                effect_size=0.0,
                challenger_better=False,
                message="Cannot compute z-test: no variance in data",
            )

        # Standard error
        se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))

        if se == 0:
            return StatisticalTestResult(
                test_name="two_proportion_z_test",
                is_significant=False,
                p_value=1.0,
                confidence_level=1 - alpha,
                effect_size=0.0,
                challenger_better=False,
                message="Cannot compute z-test: zero standard error",
            )

        # Z-statistic
        z = (p2 - p1) / se

        # Two-tailed p-value using normal CDF approximation
        p_value = 2 * (1 - StatisticalSignificance._normal_cdf(abs(z)))

        # Effect size (difference in proportions)
        effect_size = p2 - p1

        is_significant = p_value < alpha
        challenger_better = effect_size > 0

        if is_significant:
            if challenger_better:
                message = f"Challenger significantly better (p={p_value:.4f})"
            else:
                message = f"Champion significantly better (p={p_value:.4f})"
        else:
            message = f"No significant difference (p={p_value:.4f})"

        return StatisticalTestResult(
            test_name="two_proportion_z_test",
            is_significant=is_significant,
            p_value=p_value,
            confidence_level=1 - alpha,
            effect_size=effect_size,
            challenger_better=challenger_better,
            message=message,
        )

    @staticmethod
    def _normal_cdf(z: float) -> float:
        """Approximate the normal CDF using error function approximation."""
        return 0.5 * (1 + math.erf(z / math.sqrt(2)))

    @staticmethod
    def minimum_sample_size(
        baseline_rate: float,
        minimum_detectable_effect: float,
        alpha: float = 0.05,
        power: float = 0.8,
    ) -> int:
        """Calculate minimum sample size for each variant.

        Args:
            baseline_rate: Expected baseline conversion rate.
            minimum_detectable_effect: Minimum effect size to detect.
            alpha: Significance level (Type I error rate).
            power: Statistical power (1 - Type II error rate).

        Returns:
            Required sample size per variant.
        """
        # Z-scores for alpha and power
        z_alpha = 1.96 if alpha == 0.05 else 2.576 if alpha == 0.01 else 1.645
        z_beta = 0.84 if power == 0.8 else 1.28 if power == 0.9 else 0.52

        p1 = baseline_rate
        p2 = baseline_rate + minimum_detectable_effect

        # Pooled standard deviation
        p_avg = (p1 + p2) / 2
        sd = math.sqrt(2 * p_avg * (1 - p_avg))

        # Sample size formula
        n = ((z_alpha + z_beta) ** 2 * sd**2) / (minimum_detectable_effect**2)

        return max(math.ceil(n), 100)


class ModelSelectorProtocol(Protocol):
    """Protocol for model selection in A/B tests."""

    async def predict_champion(
        self, input_data: Any
    ) -> tuple[int, float, float]:
        """Get prediction from champion model.

        Returns:
            Tuple of (prediction, confidence, latency_ms).
        """
        ...

    async def predict_challenger(
        self, input_data: Any
    ) -> tuple[int, float, float]:
        """Get prediction from challenger model.

        Returns:
            Tuple of (prediction, confidence, latency_ms).
        """
        ...


@dataclass
class ABExperiment:
    """A/B experiment for comparing champion and challenger models.

    Example:
        >>> experiment = ABExperiment(
        ...     experiment_id="exp_001",
        ...     challenger_ratio=0.1,
        ...     min_samples_per_variant=1000,
        ... )
        >>> experiment.start()
        >>> variant = experiment.assign_variant("user_123")
        >>> # After getting prediction and feedback
        >>> experiment.record_prediction(
        ...     variant=variant,
        ...     prediction=1,
        ...     confidence=0.95,
        ...     latency_ms=50.0,
        ...     actual_label=1,
        ... )
        >>> result = experiment.get_result()
    """

    experiment_id: str
    challenger_ratio: float = 0.1
    min_samples_per_variant: int = 1000
    significance_level: float = 0.05
    auto_stop: bool = True
    max_duration_hours: int | None = None
    status: ExperimentStatus = ExperimentStatus.DRAFT
    started_at: datetime | None = None
    ended_at: datetime | None = None
    _splitter: TrafficSplitter = field(init=False)
    _champion_metrics: VariantMetrics = field(init=False)
    _challenger_metrics: VariantMetrics = field(init=False)

    def __post_init__(self) -> None:
        """Initialize internal state."""
        self._splitter = TrafficSplitter(self.challenger_ratio)
        self._champion_metrics = VariantMetrics(variant=VariantAssignment.CHAMPION)
        self._challenger_metrics = VariantMetrics(variant=VariantAssignment.CHALLENGER)

    def start(self) -> None:
        """Start the experiment."""
        if self.status != ExperimentStatus.DRAFT:
            raise ValueError(f"Cannot start experiment in {self.status.value} status")
        self.status = ExperimentStatus.RUNNING
        self.started_at = datetime.now(UTC)
        logger.info(f"Started A/B experiment: {self.experiment_id}")

    def pause(self) -> None:
        """Pause the experiment."""
        if self.status != ExperimentStatus.RUNNING:
            raise ValueError("Can only pause running experiments")
        self.status = ExperimentStatus.PAUSED
        logger.info(f"Paused A/B experiment: {self.experiment_id}")

    def resume(self) -> None:
        """Resume the experiment."""
        if self.status != ExperimentStatus.PAUSED:
            raise ValueError("Can only resume paused experiments")
        self.status = ExperimentStatus.RUNNING
        logger.info(f"Resumed A/B experiment: {self.experiment_id}")

    def stop(self, status: ExperimentStatus = ExperimentStatus.COMPLETED) -> None:
        """Stop the experiment.

        Args:
            status: Final status (COMPLETED or CANCELLED).
        """
        if self.status not in (ExperimentStatus.RUNNING, ExperimentStatus.PAUSED):
            raise ValueError("Can only stop running or paused experiments")
        self.status = status
        self.ended_at = datetime.now(UTC)
        logger.info(f"Stopped A/B experiment: {self.experiment_id} ({status.value})")

    def assign_variant(self, identifier: str) -> VariantAssignment:
        """Assign a variant to an identifier.

        Args:
            identifier: Unique identifier for consistent assignment.

        Returns:
            VariantAssignment.
        """
        if self.status != ExperimentStatus.RUNNING:
            return VariantAssignment.CHAMPION  # Default to champion when not running

        return self._splitter.assign(identifier)

    def record_prediction(
        self,
        variant: VariantAssignment,
        prediction: int,
        confidence: float,
        latency_ms: float,
        actual_label: int | None = None,
    ) -> None:
        """Record a prediction result.

        Args:
            variant: Which variant made the prediction.
            prediction: Model prediction (0 or 1).
            confidence: Prediction confidence.
            latency_ms: Prediction latency in milliseconds.
            actual_label: Actual label if known (for accuracy calculation).
        """
        if self.status != ExperimentStatus.RUNNING:
            return

        metrics = (
            self._champion_metrics
            if variant == VariantAssignment.CHAMPION
            else self._challenger_metrics
        )

        metrics.total_predictions += 1
        metrics.total_latency_ms += latency_ms
        metrics.total_confidence += confidence

        if prediction == 1:
            metrics.positive_predictions += 1
        else:
            metrics.negative_predictions += 1

        if actual_label is not None:
            is_correct = prediction == actual_label
            if is_correct:
                metrics.correct_predictions += 1

            # Update confusion matrix
            if prediction == 1 and actual_label == 1:
                metrics.true_positives += 1
            elif prediction == 1 and actual_label == 0:
                metrics.false_positives += 1
            elif prediction == 0 and actual_label == 0:
                metrics.true_negatives += 1
            else:
                metrics.false_negatives += 1

        # Check auto-stop conditions
        if self.auto_stop:
            self._check_auto_stop()

    def _check_auto_stop(self) -> None:
        """Check if experiment should auto-stop."""
        # Check sample size
        if (
            self._champion_metrics.total_predictions >= self.min_samples_per_variant
            and self._challenger_metrics.total_predictions >= self.min_samples_per_variant
        ):
            result = self.get_statistical_result()
            if result and result.is_significant:
                self.stop(ExperimentStatus.COMPLETED)
                logger.info(
                    f"Auto-stopped experiment {self.experiment_id}: "
                    f"Significant result found"
                )
                return

        # Check max duration
        if self.max_duration_hours and self.started_at:
            elapsed = datetime.now(UTC) - self.started_at
            if elapsed >= timedelta(hours=self.max_duration_hours):
                self.stop(ExperimentStatus.COMPLETED)
                logger.info(
                    f"Auto-stopped experiment {self.experiment_id}: "
                    f"Max duration reached"
                )

    def get_statistical_result(self) -> StatisticalTestResult | None:
        """Get statistical significance test result.

        Returns:
            StatisticalTestResult or None if insufficient data.
        """
        champ = self._champion_metrics
        chall = self._challenger_metrics

        if champ.total_predictions < 30 or chall.total_predictions < 30:
            return None

        # Use accuracy as the primary metric for comparison
        return StatisticalSignificance.z_test_proportions(
            p1=champ.accuracy,
            n1=champ.total_predictions,
            p2=chall.accuracy,
            n2=chall.total_predictions,
            alpha=self.significance_level,
        )

    def get_result(self) -> ExperimentResult:
        """Get the current experiment result.

        Returns:
            ExperimentResult with current state.
        """
        stat_result = self.get_statistical_result()

        # Generate recommendation
        if self.status == ExperimentStatus.DRAFT:
            recommendation = "Experiment not started"
        elif self._champion_metrics.total_predictions < self.min_samples_per_variant:
            recommendation = f"Need more champion samples ({self._champion_metrics.total_predictions}/{self.min_samples_per_variant})"
        elif self._challenger_metrics.total_predictions < self.min_samples_per_variant:
            recommendation = f"Need more challenger samples ({self._challenger_metrics.total_predictions}/{self.min_samples_per_variant})"
        elif stat_result is None:
            recommendation = "Insufficient labeled data for statistical test"
        elif stat_result.is_significant and stat_result.challenger_better:
            recommendation = "PROMOTE: Challenger significantly outperforms champion"
        elif stat_result.is_significant and not stat_result.challenger_better:
            recommendation = "KEEP: Champion significantly outperforms challenger"
        else:
            recommendation = "CONTINUE: No significant difference detected yet"

        duration = None
        if self.started_at:
            end = self.ended_at or datetime.now(UTC)
            duration = end - self.started_at

        return ExperimentResult(
            experiment_id=self.experiment_id,
            status=self.status,
            champion_metrics=self._champion_metrics,
            challenger_metrics=self._challenger_metrics,
            statistical_test=stat_result,
            recommendation=recommendation,
            started_at=self.started_at or datetime.now(UTC),
            ended_at=self.ended_at,
            duration=duration,
        )

    def get_status(self) -> dict[str, Any]:
        """Get experiment status summary.

        Returns:
            Status dictionary.
        """
        return {
            "experiment_id": self.experiment_id,
            "status": self.status.value,
            "challenger_ratio": self.challenger_ratio,
            "champion_predictions": self._champion_metrics.total_predictions,
            "challenger_predictions": self._challenger_metrics.total_predictions,
            "champion_accuracy": self._champion_metrics.accuracy,
            "challenger_accuracy": self._challenger_metrics.accuracy,
            "started_at": self.started_at.isoformat() if self.started_at else None,
        }

    def update_traffic_ratio(self, new_ratio: float) -> None:
        """Update the challenger traffic ratio.

        Args:
            new_ratio: New ratio (0.0 to 1.0).
        """
        self._splitter.update_ratio(new_ratio)
        self.challenger_ratio = new_ratio
        logger.info(
            f"Updated traffic ratio for {self.experiment_id}: "
            f"challenger={new_ratio:.1%}"
        )


class ExperimentManager:
    """Manage multiple A/B experiments.

    Example:
        >>> manager = ExperimentManager()
        >>> exp = manager.create_experiment(
        ...     experiment_id="exp_001",
        ...     challenger_ratio=0.1,
        ... )
        >>> manager.start_experiment("exp_001")
        >>> variant = manager.assign_variant("exp_001", "user_123")
    """

    def __init__(self) -> None:
        """Initialize the experiment manager."""
        self._experiments: dict[str, ABExperiment] = {}

    def create_experiment(
        self,
        experiment_id: str,
        challenger_ratio: float = 0.1,
        min_samples_per_variant: int = 1000,
        significance_level: float = 0.05,
        auto_stop: bool = True,
        max_duration_hours: int | None = None,
    ) -> ABExperiment:
        """Create a new experiment.

        Args:
            experiment_id: Unique experiment identifier.
            challenger_ratio: Traffic ratio for challenger.
            min_samples_per_variant: Minimum samples before significance test.
            significance_level: Alpha for statistical test.
            auto_stop: Whether to auto-stop on significant result.
            max_duration_hours: Maximum experiment duration.

        Returns:
            Created ABExperiment.
        """
        if experiment_id in self._experiments:
            raise ValueError(f"Experiment {experiment_id} already exists")

        experiment = ABExperiment(
            experiment_id=experiment_id,
            challenger_ratio=challenger_ratio,
            min_samples_per_variant=min_samples_per_variant,
            significance_level=significance_level,
            auto_stop=auto_stop,
            max_duration_hours=max_duration_hours,
        )

        self._experiments[experiment_id] = experiment
        logger.info(f"Created experiment: {experiment_id}")
        return experiment

    def get_experiment(self, experiment_id: str) -> ABExperiment | None:
        """Get an experiment by ID.

        Args:
            experiment_id: Experiment identifier.

        Returns:
            ABExperiment or None.
        """
        return self._experiments.get(experiment_id)

    def start_experiment(self, experiment_id: str) -> None:
        """Start an experiment.

        Args:
            experiment_id: Experiment identifier.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment {experiment_id} not found")
        exp.start()

    def stop_experiment(
        self,
        experiment_id: str,
        status: ExperimentStatus = ExperimentStatus.COMPLETED,
    ) -> None:
        """Stop an experiment.

        Args:
            experiment_id: Experiment identifier.
            status: Final status.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment {experiment_id} not found")
        exp.stop(status)

    def assign_variant(
        self,
        experiment_id: str,
        identifier: str,
    ) -> VariantAssignment:
        """Assign a variant for an experiment.

        Args:
            experiment_id: Experiment identifier.
            identifier: User/request identifier.

        Returns:
            VariantAssignment.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            return VariantAssignment.CHAMPION
        return exp.assign_variant(identifier)

    def record_prediction(
        self,
        experiment_id: str,
        variant: VariantAssignment,
        prediction: int,
        confidence: float,
        latency_ms: float,
        actual_label: int | None = None,
    ) -> None:
        """Record a prediction for an experiment.

        Args:
            experiment_id: Experiment identifier.
            variant: Variant that made prediction.
            prediction: Model prediction.
            confidence: Prediction confidence.
            latency_ms: Prediction latency.
            actual_label: Actual label if known.
        """
        exp = self._experiments.get(experiment_id)
        if exp is None:
            return
        exp.record_prediction(
            variant=variant,
            prediction=prediction,
            confidence=confidence,
            latency_ms=latency_ms,
            actual_label=actual_label,
        )

    def get_all_results(self) -> dict[str, ExperimentResult]:
        """Get results for all experiments.

        Returns:
            Dictionary of experiment_id to ExperimentResult.
        """
        return {
            exp_id: exp.get_result() for exp_id, exp in self._experiments.items()
        }

    def get_running_experiments(self) -> list[str]:
        """Get IDs of running experiments.

        Returns:
            List of running experiment IDs.
        """
        return [
            exp_id
            for exp_id, exp in self._experiments.items()
            if exp.status == ExperimentStatus.RUNNING
        ]

    def cleanup_completed(self, keep_last: int = 10) -> int:
        """Remove old completed experiments.

        Args:
            keep_last: Number of completed experiments to keep.

        Returns:
            Number of experiments removed.
        """
        completed = [
            (exp_id, exp)
            for exp_id, exp in self._experiments.items()
            if exp.status in (ExperimentStatus.COMPLETED, ExperimentStatus.CANCELLED)
        ]

        # Sort by end time
        completed.sort(key=lambda x: x[1].ended_at or datetime.min, reverse=True)

        to_remove = completed[keep_last:]
        for exp_id, _ in to_remove:
            del self._experiments[exp_id]

        return len(to_remove)
