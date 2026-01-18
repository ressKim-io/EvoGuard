"""Leetspeak-based evasion strategy."""

import random

from attacker.mappings.leetspeak_map import LEETSPEAK_MAPS, LeetspeakLevel
from attacker.strategies.base import AttackStrategy, EvasionResult

# Re-export for convenience
__all__ = ["LeetspeakLevel", "LeetspeakStrategy"]


class LeetspeakStrategy(AttackStrategy):
    """Leetspeak substitution strategy for content evasion.

    Converts text to leetspeak with configurable intensity levels:
    - BASIC: Simple substitutions (a->4, e->3, i->1, o->0)
    - MODERATE: More substitutions, some multi-char
    - ADVANCED: Aggressive multi-char substitutions
    - EXTREME: Maximum obfuscation with special characters
    """

    def __init__(
        self,
        level: LeetspeakLevel = LeetspeakLevel.MODERATE,
        probability: float = 0.7,
        preserve_case: bool = False,
    ) -> None:
        """Initialize strategy.

        Args:
            level: Leetspeak intensity level
            probability: Probability of transforming each eligible character
            preserve_case: If True, attempt to preserve original case
        """
        self._level = level
        self._probability = min(1.0, max(0.0, probability))
        self._preserve_case = preserve_case
        self._mapping = LEETSPEAK_MAPS[level]

    @property
    def name(self) -> str:
        return f"leetspeak_{self._level.value}"

    def generate(self, text: str, num_variants: int = 1) -> list[EvasionResult]:
        """Generate leetspeak evasion variants."""
        results = []

        for _ in range(num_variants):
            evasion = self._apply_leetspeak(text)
            confidence = self._calculate_confidence(text, evasion)
            results.append(
                EvasionResult(
                    original=text,
                    evasion=evasion,
                    strategy=self.name,
                    confidence=confidence,
                )
            )

        return results

    def _apply_leetspeak(self, text: str) -> str:
        """Apply leetspeak transformation to text."""
        result = []

        for char in text:
            lower_char = char.lower()

            if random.random() < self._probability and lower_char in self._mapping:
                replacements = self._mapping[lower_char]
                replacement = random.choice(replacements)

                if self._preserve_case and char.isupper():
                    replacement = replacement.upper()

                result.append(replacement)
            else:
                result.append(char)

        return "".join(result)

    def _calculate_confidence(self, original: str, evasion: str) -> float:
        """Calculate confidence based on transformation level and amount."""
        if original == evasion:
            return 0.1

        # Base confidence by level
        level_confidence = {
            LeetspeakLevel.BASIC: 0.5,
            LeetspeakLevel.MODERATE: 0.65,
            LeetspeakLevel.ADVANCED: 0.75,
            LeetspeakLevel.EXTREME: 0.85,
        }

        base = level_confidence.get(self._level, 0.6)

        # Adjust based on transformation ratio
        transformed = sum(1 for o, e in zip(original, evasion, strict=False) if o != e)
        ratio = transformed / len(original) if original else 0

        return min(0.95, base + (ratio * 0.1))


class AdaptiveLeetspeakStrategy(AttackStrategy):
    """Adaptive leetspeak that increases intensity based on text characteristics."""

    def __init__(self, base_probability: float = 0.6) -> None:
        """Initialize strategy.

        Args:
            base_probability: Base probability for transformations
        """
        self._base_probability = base_probability

    @property
    def name(self) -> str:
        return "leetspeak_adaptive"

    def generate(self, text: str, num_variants: int = 1) -> list[EvasionResult]:
        """Generate adaptive leetspeak variants."""
        results = []

        for i in range(num_variants):
            # Increase level with each variant
            level = self._select_level(text, i, num_variants)
            strategy = LeetspeakStrategy(
                level=level, probability=self._base_probability
            )
            result = strategy.generate(text, 1)[0]

            # Wrap with adaptive strategy name
            results.append(
                EvasionResult(
                    original=result.original,
                    evasion=result.evasion,
                    strategy=self.name,
                    confidence=result.confidence,
                )
            )

        return results

    def _select_level(
        self, text: str, variant_idx: int, total_variants: int
    ) -> LeetspeakLevel:
        """Select appropriate level based on text and variant index."""
        levels = list(LeetspeakLevel)

        # For multiple variants, distribute across levels
        if total_variants > 1:
            level_idx = min(variant_idx, len(levels) - 1)
            return levels[level_idx]

        # For single variant, choose based on text length
        if len(text) < 5:
            return LeetspeakLevel.BASIC
        elif len(text) < 15:
            return LeetspeakLevel.MODERATE
        elif len(text) < 30:
            return LeetspeakLevel.ADVANCED
        else:
            return LeetspeakLevel.EXTREME
