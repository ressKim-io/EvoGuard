"""Quality gate for adversarial pipeline.

This module evaluates model performance against configurable thresholds
and decides whether retraining is required.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from ml_service.pipeline.config import QualityGateConfig
from ml_service.pipeline.attack_runner import AttackBatchResult

logger = logging.getLogger(__name__)


class DecisionType(Enum):
    """Quality gate decision types."""

    PASS = "pass"  # Model meets quality thresholds
    FAIL = "fail"  # Model fails thresholds, needs retraining
    WARN = "warn"  # Model close to thresholds, monitor closely


class FailureReason(Enum):
    """Reasons for quality gate failure."""

    EVASION_RATE_HIGH = "evasion_rate_exceeded"
    F1_SCORE_LOW = "f1_score_below_threshold"
    F1_DROP_HIGH = "f1_score_dropped"
    PRECISION_LOW = "precision_below_threshold"
    RECALL_LOW = "recall_below_threshold"


@dataclass
class QualityGateDecision:
    """Result of quality gate evaluation."""

    decision: DecisionType
    needs_retraining: bool
    evasion_rate: float
    evasion_rate_threshold: float
    f1_score: float | None
    f1_threshold: float
    precision: float | None
    recall: float | None
    failure_reasons: list[FailureReason]
    warnings: list[str]
    timestamp: datetime
    metrics: dict[str, float] = field(default_factory=dict)
    baseline_f1: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision": self.decision.value,
            "needs_retraining": self.needs_retraining,
            "evasion_rate": self.evasion_rate,
            "evasion_rate_threshold": self.evasion_rate_threshold,
            "f1_score": self.f1_score,
            "f1_threshold": self.f1_threshold,
            "precision": self.precision,
            "recall": self.recall,
            "failure_reasons": [r.value for r in self.failure_reasons],
            "warnings": self.warnings,
            "timestamp": self.timestamp.isoformat(),
            "metrics": self.metrics,
            "baseline_f1": self.baseline_f1,
        }

    @property
    def summary(self) -> str:
        """Get human-readable summary."""
        status = "PASS" if self.decision == DecisionType.PASS else "FAIL"
        reasons = ", ".join(r.value for r in self.failure_reasons)
        return (
            f"Quality Gate {status}: evasion_rate={self.evasion_rate:.1%} "
            f"(threshold={self.evasion_rate_threshold:.1%})"
            + (f", failures=[{reasons}]" if reasons else "")
        )


class QualityGate:
    """Evaluates model quality against configurable thresholds.

    The quality gate checks:
    - Evasion rate (% of adversarial examples that fool the model)
    - F1 score on evaluation dataset
    - F1 score drop from baseline
    - Precision and recall thresholds

    Example:
        >>> config = QualityGateConfig(max_evasion_rate=0.15)
        >>> gate = QualityGate(config)
        >>> decision = gate.evaluate(attack_result)
        >>> if decision.needs_retraining:
        ...     trigger_retraining()
    """

    def __init__(
        self,
        config: QualityGateConfig,
        baseline_f1: float | None = None,
    ) -> None:
        """Initialize quality gate.

        Args:
            config: Quality gate configuration.
            baseline_f1: Baseline F1 score for comparison.
        """
        self.config = config
        self.baseline_f1 = baseline_f1

    def evaluate(
        self,
        attack_result: AttackBatchResult,
        eval_metrics: dict[str, float] | None = None,
    ) -> QualityGateDecision:
        """Evaluate model quality based on attack results and metrics.

        Args:
            attack_result: Results from attack batch.
            eval_metrics: Optional evaluation metrics (f1, precision, recall).

        Returns:
            QualityGateDecision with pass/fail status.
        """
        timestamp = datetime.now(UTC)
        failure_reasons: list[FailureReason] = []
        warnings: list[str] = []

        # Extract metrics
        evasion_rate = attack_result.evasion_rate
        f1_score = eval_metrics.get("f1") if eval_metrics else None
        precision = eval_metrics.get("precision") if eval_metrics else None
        recall = eval_metrics.get("recall") if eval_metrics else None

        # Check evasion rate threshold
        if evasion_rate > self.config.max_evasion_rate:
            failure_reasons.append(FailureReason.EVASION_RATE_HIGH)
            logger.warning(
                f"Evasion rate {evasion_rate:.1%} exceeds threshold "
                f"{self.config.max_evasion_rate:.1%}"
            )
        elif evasion_rate > self.config.max_evasion_rate * 0.8:
            warnings.append(
                f"Evasion rate {evasion_rate:.1%} approaching threshold "
                f"{self.config.max_evasion_rate:.1%}"
            )

        # Check F1 score threshold
        if f1_score is not None:
            if f1_score < self.config.min_f1_score:
                failure_reasons.append(FailureReason.F1_SCORE_LOW)
                logger.warning(
                    f"F1 score {f1_score:.3f} below threshold "
                    f"{self.config.min_f1_score:.3f}"
                )
            elif f1_score < self.config.min_f1_score + 0.05:
                warnings.append(
                    f"F1 score {f1_score:.3f} approaching threshold "
                    f"{self.config.min_f1_score:.3f}"
                )

            # Check F1 drop from baseline
            if self.baseline_f1 is not None:
                f1_drop = self.baseline_f1 - f1_score
                if f1_drop > self.config.min_f1_drop:
                    failure_reasons.append(FailureReason.F1_DROP_HIGH)
                    logger.warning(
                        f"F1 drop {f1_drop:.3f} exceeds threshold "
                        f"{self.config.min_f1_drop:.3f}"
                    )

        # Check precision threshold
        if precision is not None and precision < self.config.min_precision:
            failure_reasons.append(FailureReason.PRECISION_LOW)
            logger.warning(
                f"Precision {precision:.3f} below threshold "
                f"{self.config.min_precision:.3f}"
            )

        # Check recall threshold
        if recall is not None and recall < self.config.min_recall:
            failure_reasons.append(FailureReason.RECALL_LOW)
            logger.warning(
                f"Recall {recall:.3f} below threshold {self.config.min_recall:.3f}"
            )

        # Determine decision
        if failure_reasons:
            decision = DecisionType.FAIL
            needs_retraining = True
        elif warnings:
            decision = DecisionType.WARN
            needs_retraining = False
        else:
            decision = DecisionType.PASS
            needs_retraining = False

        # Compile metrics
        all_metrics = {
            "evasion_rate": evasion_rate,
            "total_variants": attack_result.total_variants,
            "total_evasions": attack_result.total_evasions,
        }
        if eval_metrics:
            all_metrics.update(eval_metrics)

        return QualityGateDecision(
            decision=decision,
            needs_retraining=needs_retraining,
            evasion_rate=evasion_rate,
            evasion_rate_threshold=self.config.max_evasion_rate,
            f1_score=f1_score,
            f1_threshold=self.config.min_f1_score,
            precision=precision,
            recall=recall,
            failure_reasons=failure_reasons,
            warnings=warnings,
            timestamp=timestamp,
            metrics=all_metrics,
            baseline_f1=self.baseline_f1,
        )

    def evaluate_simple(self, evasion_rate: float) -> QualityGateDecision:
        """Simple evaluation based only on evasion rate.

        Args:
            evasion_rate: Calculated evasion rate.

        Returns:
            QualityGateDecision.
        """
        timestamp = datetime.now(UTC)
        failure_reasons: list[FailureReason] = []
        warnings: list[str] = []

        if evasion_rate > self.config.max_evasion_rate:
            failure_reasons.append(FailureReason.EVASION_RATE_HIGH)
        elif evasion_rate > self.config.max_evasion_rate * 0.8:
            warnings.append("Evasion rate approaching threshold")

        decision = DecisionType.FAIL if failure_reasons else (
            DecisionType.WARN if warnings else DecisionType.PASS
        )

        return QualityGateDecision(
            decision=decision,
            needs_retraining=bool(failure_reasons),
            evasion_rate=evasion_rate,
            evasion_rate_threshold=self.config.max_evasion_rate,
            f1_score=None,
            f1_threshold=self.config.min_f1_score,
            precision=None,
            recall=None,
            failure_reasons=failure_reasons,
            warnings=warnings,
            timestamp=timestamp,
            baseline_f1=self.baseline_f1,
        )

    def update_baseline(self, new_baseline_f1: float) -> None:
        """Update the baseline F1 score.

        Args:
            new_baseline_f1: New baseline F1 score.
        """
        old_baseline = self.baseline_f1
        self.baseline_f1 = new_baseline_f1
        logger.info(f"Updated baseline F1: {old_baseline} -> {new_baseline_f1}")

    def get_threshold_summary(self) -> dict[str, Any]:
        """Get summary of current thresholds.

        Returns:
            Dictionary of thresholds.
        """
        return {
            "max_evasion_rate": self.config.max_evasion_rate,
            "min_f1_score": self.config.min_f1_score,
            "min_f1_drop": self.config.min_f1_drop,
            "min_precision": self.config.min_precision,
            "min_recall": self.config.min_recall,
            "baseline_f1": self.baseline_f1,
        }
