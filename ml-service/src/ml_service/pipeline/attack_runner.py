"""Attack runner for adversarial pipeline.

This module handles batch execution of adversarial attacks against the model,
generating evasion variants and evaluating model responses.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd

from ml_service.pipeline.config import AttackConfig

logger = logging.getLogger(__name__)

# Add attacker module to path if needed
ATTACKER_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "attacker"
if str(ATTACKER_PATH) not in sys.path:
    sys.path.insert(0, str(ATTACKER_PATH.parent))


class ClassifierProtocol(Protocol):
    """Protocol for classifier models."""

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict labels for texts.

        Returns:
            List of dicts with keys: label, label_name, confidence
        """
        ...


@dataclass
class AttackResult:
    """Result of a single attack."""

    original_text: str
    original_label: int
    variant_text: str
    strategy_name: str
    model_prediction: int
    model_confidence: float
    evaded: bool  # True if model was fooled (toxic classified as non-toxic)

    @property
    def is_evasion(self) -> bool:
        """Check if this is a successful evasion."""
        return self.evaded


@dataclass
class AttackBatchResult:
    """Result of a batch attack run."""

    timestamp: datetime
    total_originals: int
    total_variants: int
    total_evasions: int
    evasion_rate: float
    attack_results: list[AttackResult]
    strategies_used: list[str]
    duration_seconds: float
    metrics: dict[str, float] = field(default_factory=dict)

    def get_failed_samples(self) -> list[AttackResult]:
        """Get samples that evaded the model."""
        return [r for r in self.attack_results if r.evaded]

    def get_strategy_breakdown(self) -> dict[str, dict[str, int]]:
        """Get evasion breakdown by strategy."""
        breakdown: dict[str, dict[str, int]] = {}
        for result in self.attack_results:
            if result.strategy_name not in breakdown:
                breakdown[result.strategy_name] = {"total": 0, "evasions": 0}
            breakdown[result.strategy_name]["total"] += 1
            if result.evaded:
                breakdown[result.strategy_name]["evasions"] += 1
        return breakdown

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_originals": self.total_originals,
            "total_variants": self.total_variants,
            "total_evasions": self.total_evasions,
            "evasion_rate": self.evasion_rate,
            "strategies_used": self.strategies_used,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "strategy_breakdown": self.get_strategy_breakdown(),
        }


class AttackRunner:
    """Runs adversarial attacks against a classifier model.

    Uses the AttackOrchestrator from the attacker module to generate
    evasion variants and evaluates them against the classifier.

    Example:
        >>> config = AttackConfig(batch_size=100, num_variants=5)
        >>> runner = AttackRunner(config, classifier)
        >>> result = await runner.run_batch(corpus_df)
        >>> print(f"Evasion rate: {result.evasion_rate:.1%}")
    """

    def __init__(
        self,
        config: AttackConfig,
        classifier: ClassifierProtocol,
    ) -> None:
        """Initialize attack runner.

        Args:
            config: Attack configuration.
            classifier: Classifier model to attack.
        """
        self.config = config
        self.classifier = classifier
        self._orchestrator = None

    def _get_orchestrator(self) -> Any:
        """Get or create attack orchestrator."""
        if self._orchestrator is None:
            try:
                from attacker.orchestrator import (
                    AttackOrchestrator,
                    OrchestratorConfig,
                    SelectionMode,
                )

                orch_config = OrchestratorConfig(
                    selection_mode=SelectionMode.WEIGHTED,
                    include_llm=self.config.include_llm_strategies,
                )
                self._orchestrator = AttackOrchestrator(orch_config)
                logger.info(
                    f"Initialized AttackOrchestrator with "
                    f"{len(self._orchestrator.enabled_strategies)} strategies"
                )
            except ImportError as e:
                logger.error(f"Failed to import attacker module: {e}")
                raise ImportError(
                    "Attacker module not found. Ensure the attacker package is installed."
                ) from e
        return self._orchestrator

    def load_corpus(self, corpus_path: Path | str | None = None) -> pd.DataFrame:
        """Load attack corpus from file.

        Args:
            corpus_path: Path to corpus file (CSV). Uses config if not provided.

        Returns:
            DataFrame with text and label columns.
        """
        path = Path(corpus_path) if corpus_path else self.config.corpus_path

        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        df = pd.read_csv(path)

        # Validate columns
        if self.config.text_column not in df.columns:
            raise ValueError(f"Text column '{self.config.text_column}' not found")
        if self.config.label_column not in df.columns:
            raise ValueError(f"Label column '{self.config.label_column}' not found")

        # Filter to toxic samples only (label == 1)
        toxic_df = df[df[self.config.label_column] == 1].copy()
        logger.info(f"Loaded {len(toxic_df)} toxic samples from corpus")

        return toxic_df

    def run_batch(
        self,
        corpus_df: pd.DataFrame | None = None,
        sample_size: int | None = None,
    ) -> AttackBatchResult:
        """Run a batch of attacks.

        Args:
            corpus_df: DataFrame with texts to attack. Loads from config if None.
            sample_size: Number of samples to attack. Uses config if None.

        Returns:
            AttackBatchResult with all attack results.
        """
        start_time = datetime.now(UTC)

        # Load corpus if not provided
        if corpus_df is None:
            corpus_df = self.load_corpus()

        # Sample texts
        sample_size = sample_size or self.config.batch_size
        if len(corpus_df) > sample_size:
            corpus_df = corpus_df.sample(n=sample_size, random_state=42)

        logger.info(f"Running attacks on {len(corpus_df)} samples")

        # Get orchestrator
        orchestrator = self._get_orchestrator()

        # Generate variants and evaluate
        all_results: list[AttackResult] = []
        strategies_used: set[str] = set()

        for _, row in corpus_df.iterrows():
            text = row[self.config.text_column]
            label = int(row[self.config.label_column])

            # Generate variants
            orch_result = orchestrator.generate(text, num_variants=self.config.num_variants)

            strategies_used.update(orch_result.strategies_used)

            # Evaluate each variant
            variant_texts = [r.evasion for r in orch_result.results]
            if not variant_texts:
                continue

            # Batch predict
            predictions = self.classifier.predict(variant_texts)

            for evasion_result, prediction in zip(orch_result.results, predictions):
                # Check if evasion was successful
                # Toxic (1) classified as non-toxic (0) = successful evasion
                evaded = label == 1 and prediction["label"] == 0

                attack_result = AttackResult(
                    original_text=text,
                    original_label=label,
                    variant_text=evasion_result.evasion,
                    strategy_name=evasion_result.strategy,
                    model_prediction=prediction["label"],
                    model_confidence=prediction["confidence"],
                    evaded=evaded,
                )
                all_results.append(attack_result)

        # Calculate metrics
        total_evasions = sum(1 for r in all_results if r.evaded)
        evasion_rate = total_evasions / len(all_results) if all_results else 0.0

        duration = (datetime.now(UTC) - start_time).total_seconds()

        result = AttackBatchResult(
            timestamp=start_time,
            total_originals=len(corpus_df),
            total_variants=len(all_results),
            total_evasions=total_evasions,
            evasion_rate=evasion_rate,
            attack_results=all_results,
            strategies_used=list(strategies_used),
            duration_seconds=duration,
        )

        logger.info(
            f"Attack batch complete: {total_evasions}/{len(all_results)} evasions "
            f"({evasion_rate:.1%}) in {duration:.1f}s"
        )

        return result

    async def run_batch_async(
        self,
        corpus_df: pd.DataFrame | None = None,
        sample_size: int | None = None,
    ) -> AttackBatchResult:
        """Run a batch of attacks asynchronously.

        Args:
            corpus_df: DataFrame with texts to attack.
            sample_size: Number of samples to attack.

        Returns:
            AttackBatchResult with all attack results.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.run_batch(corpus_df, sample_size),
        )

    def run_single(self, text: str, label: int = 1) -> list[AttackResult]:
        """Run attack on a single text.

        Args:
            text: Text to attack.
            label: True label (default: 1 for toxic).

        Returns:
            List of AttackResults for all variants.
        """
        orchestrator = self._get_orchestrator()

        # Generate variants
        orch_result = orchestrator.generate(text, num_variants=self.config.num_variants)

        if not orch_result.results:
            return []

        # Evaluate variants
        variant_texts = [r.evasion for r in orch_result.results]
        predictions = self.classifier.predict(variant_texts)

        results = []
        for evasion_result, prediction in zip(orch_result.results, predictions):
            evaded = label == 1 and prediction["label"] == 0

            results.append(
                AttackResult(
                    original_text=text,
                    original_label=label,
                    variant_text=evasion_result.evasion,
                    strategy_name=evasion_result.strategy,
                    model_prediction=prediction["label"],
                    model_confidence=prediction["confidence"],
                    evaded=evaded,
                )
            )

        return results


class MockClassifier:
    """Mock classifier for testing purposes."""

    def __init__(self, evasion_rate: float = 0.1) -> None:
        """Initialize mock classifier.

        Args:
            evasion_rate: Probability of classifying toxic as non-toxic.
        """
        self.evasion_rate = evasion_rate
        import random
        self._random = random.Random(42)

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Mock prediction."""
        results = []
        for text in texts:
            # Simulate evasion based on rate
            if self._random.random() < self.evasion_rate:
                results.append({
                    "label": 0,
                    "label_name": "non-toxic",
                    "confidence": 0.7 + self._random.random() * 0.2,
                })
            else:
                results.append({
                    "label": 1,
                    "label_name": "toxic",
                    "confidence": 0.8 + self._random.random() * 0.15,
                })
        return results
