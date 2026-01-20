"""Korean Attack Runner for Adversarial Pipeline.

한국어 공격 실행기 - 한국어 텍스트에 대한 적대적 공격 수행
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ml_service.attacker.korean_strategies import (
    KOREAN_ATTACK_STRATEGIES,
    apply_korean_attack,
    apply_random_korean_attacks,
)
from ml_service.pipeline.korean_config import KoreanAttackConfig

logger = logging.getLogger(__name__)


@dataclass
class KoreanAttackResult:
    """Single attack result."""

    original_text: str
    variant_text: str
    strategy_name: str
    original_label: int
    model_prediction: int
    model_confidence: float
    is_evasion: bool  # 모델이 잘못 분류했으면 True


@dataclass
class KoreanBatchResult:
    """Batch attack results."""

    attack_results: list[KoreanAttackResult] = field(default_factory=list)
    total_attacks: int = 0
    total_evasions: int = 0
    elapsed_seconds: float = 0.0

    @property
    def evasion_rate(self) -> float:
        """Calculate evasion rate."""
        if self.total_attacks == 0:
            return 0.0
        return self.total_evasions / self.total_attacks

    def get_failed_samples(self) -> list[KoreanAttackResult]:
        """Get samples that evaded detection."""
        return [r for r in self.attack_results if r.is_evasion]

    def get_strategy_stats(self) -> dict[str, dict[str, int]]:
        """Get statistics per strategy."""
        stats: dict[str, dict[str, int]] = {}

        for result in self.attack_results:
            if result.strategy_name not in stats:
                stats[result.strategy_name] = {"total": 0, "evasions": 0}

            stats[result.strategy_name]["total"] += 1
            if result.is_evasion:
                stats[result.strategy_name]["evasions"] += 1

        return stats


class KoreanAttackRunner:
    """Korean attack runner for adversarial pipeline."""

    def __init__(
        self,
        config: KoreanAttackConfig,
        classifier: Any,
    ):
        """Initialize attack runner.

        Args:
            config: Attack configuration
            classifier: Classifier with predict() method
        """
        self.config = config
        self.classifier = classifier
        self._corpus: pd.DataFrame | None = None

    def load_corpus(self, corpus_path: Path | None = None) -> None:
        """Load toxic corpus for attacks."""
        path = corpus_path or Path("data/korean/korean_hate_speech_balanced.csv")

        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: {path}")

        self._corpus = pd.read_csv(path)
        # Filter to only toxic samples for attacks
        self._corpus = self._corpus[self._corpus["label"] == 1]
        logger.info(f"Loaded {len(self._corpus)} toxic samples from corpus")

    def _get_sample_texts(self, n: int) -> list[tuple[str, int]]:
        """Get sample texts from corpus."""
        if self._corpus is None:
            self.load_corpus()

        samples = self._corpus.sample(n=min(n, len(self._corpus)))
        return [(row["text"], row["label"]) for _, row in samples.iterrows()]

    def _generate_variants(
        self,
        text: str,
        num_variants: int,
    ) -> list[tuple[str, str]]:
        """Generate attack variants for a single text.

        Returns:
            List of (strategy_name, variant_text) tuples
        """
        variants = []

        # 각 전략별로 변형 생성
        strategies = [s for s in KOREAN_ATTACK_STRATEGIES
                      if s.name in self.config.strategies]

        for strategy in strategies:
            try:
                variant = strategy.transform(text)
                if variant != text:  # 변형이 실제로 발생한 경우만
                    variants.append((strategy.name, variant))
            except Exception as e:
                logger.warning(f"Strategy {strategy.name} failed: {e}")

        # 추가 랜덤 변형
        if len(variants) < num_variants:
            random_variants = apply_random_korean_attacks(
                text,
                num_variants=num_variants - len(variants)
            )
            variants.extend(random_variants)

        return variants[:num_variants]

    def run_batch(
        self,
        num_samples: int | None = None,
        texts: list[tuple[str, int]] | None = None,
    ) -> KoreanBatchResult:
        """Run batch attack.

        Args:
            num_samples: Number of samples to attack (from corpus)
            texts: Optional list of (text, label) tuples to attack

        Returns:
            KoreanBatchResult with attack results
        """
        start_time = time.perf_counter()

        # Get texts to attack
        if texts is None:
            num_samples = num_samples or self.config.batch_size
            texts = self._get_sample_texts(num_samples)

        logger.info(f"Running attacks on {len(texts)} samples")

        all_results: list[KoreanAttackResult] = []
        total_evasions = 0

        for original_text, original_label in texts:
            # Generate variants
            variants = self._generate_variants(
                original_text,
                num_variants=self.config.num_variants,
            )

            if not variants:
                continue

            # Get predictions for all variants
            variant_texts = [v[1] for v in variants]
            predictions = self.classifier.predict(variant_texts)

            for (strategy_name, variant_text), pred in zip(variants, predictions):
                is_evasion = pred["label"] != original_label

                result = KoreanAttackResult(
                    original_text=original_text,
                    variant_text=variant_text,
                    strategy_name=strategy_name,
                    original_label=original_label,
                    model_prediction=pred["label"],
                    model_confidence=pred["confidence"],
                    is_evasion=is_evasion,
                )
                all_results.append(result)

                if is_evasion:
                    total_evasions += 1

        elapsed = time.perf_counter() - start_time

        result = KoreanBatchResult(
            attack_results=all_results,
            total_attacks=len(all_results),
            total_evasions=total_evasions,
            elapsed_seconds=elapsed,
        )

        logger.info(
            f"Attack batch complete: {total_evasions}/{len(all_results)} "
            f"evasions ({result.evasion_rate:.1%}) in {elapsed:.1f}s"
        )

        # Log per-strategy stats
        stats = result.get_strategy_stats()
        logger.info("Per-strategy results:")
        for strategy, s in sorted(stats.items(), key=lambda x: -x[1].get("evasions", 0)):
            evasion_rate = s["evasions"] / s["total"] if s["total"] > 0 else 0
            logger.info(f"  {strategy:20s}: {s['evasions']}/{s['total']} ({evasion_rate:.1%})")

        return result


def run_korean_attack_test(classifier, num_samples: int = 20) -> KoreanBatchResult:
    """Quick test function for Korean attacks.

    Args:
        classifier: Classifier with predict() method
        num_samples: Number of samples to test

    Returns:
        KoreanBatchResult
    """
    config = KoreanAttackConfig(
        num_variants=10,
        batch_size=num_samples,
    )
    runner = KoreanAttackRunner(config, classifier)
    return runner.run_batch()
