"""Homoglyph-based evasion strategy."""
# ruff: noqa: RUF002

import random

from attacker.mappings.homoglyph_map import HOMOGLYPH_MAP
from attacker.strategies.base import AttackStrategy, EvasionResult


class HomoglyphStrategy(AttackStrategy):
    """Homoglyph substitution strategy for content evasion.

    Replaces characters with visually similar characters from different
    Unicode blocks (Cyrillic, Greek, Cherokee, etc.).

    Examples:
        - hello -> hеllo (e replaced with Cyrillic е)
        - ALPHA -> ΑLPHA (A replaced with Greek Α)
        - o -> 0 (letter o replaced with digit 0)
    """

    def __init__(
        self,
        probability: float = 0.5,
        min_substitutions: int = 1,
        max_substitutions: int | None = None,
    ) -> None:
        """Initialize strategy.

        Args:
            probability: Probability of substituting each eligible character
            min_substitutions: Minimum number of substitutions to make
            max_substitutions: Maximum substitutions (None = unlimited)
        """
        self._probability = min(1.0, max(0.0, probability))
        self._min_substitutions = max(1, min_substitutions)
        self._max_substitutions = max_substitutions

    @property
    def name(self) -> str:
        return "homoglyph"

    def generate(self, text: str, num_variants: int = 1) -> list[EvasionResult]:
        """Generate homoglyph evasion variants."""
        results = []

        for _ in range(num_variants):
            evasion = self._apply_substitution(text)
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

    def _apply_substitution(self, text: str) -> str:
        """Apply homoglyph substitution to text."""
        chars = list(text)
        eligible_indices = [
            i for i, char in enumerate(chars) if char in HOMOGLYPH_MAP
        ]

        if not eligible_indices:
            return text

        # Determine number of substitutions
        num_subs = self._determine_substitution_count(len(eligible_indices))

        # Select indices to substitute
        indices_to_sub = random.sample(
            eligible_indices, min(num_subs, len(eligible_indices))
        )

        # Apply substitutions
        for idx in indices_to_sub:
            char = chars[idx]
            homoglyphs = HOMOGLYPH_MAP[char]
            chars[idx] = random.choice(homoglyphs)

        return "".join(chars)

    def _determine_substitution_count(self, eligible_count: int) -> int:
        """Determine how many substitutions to make."""
        if eligible_count == 0:
            return 0

        # Calculate based on probability
        expected = int(eligible_count * self._probability)
        count = max(self._min_substitutions, expected)

        if self._max_substitutions is not None:
            count = min(count, self._max_substitutions)

        return count

    def _calculate_confidence(self, original: str, evasion: str) -> float:
        """Calculate confidence based on substitution ratio."""
        if original == evasion:
            return 0.1

        # Count actual substitutions
        substitutions = sum(
            1 for o, e in zip(original, evasion, strict=False) if o != e
        )

        # Confidence increases with more substitutions
        ratio = substitutions / len(original) if original else 0
        base_confidence = 0.7

        # Scale confidence: more subs = higher confidence, up to a point
        return min(0.95, base_confidence + (ratio * 0.25))


class TargetedHomoglyphStrategy(AttackStrategy):
    """Targeted homoglyph strategy that focuses on specific characters."""

    def __init__(self, target_chars: set[str] | None = None) -> None:
        """Initialize strategy.

        Args:
            target_chars: Set of characters to target for substitution.
                         If None, targets common filter-triggering chars.
        """
        self._target_chars = (
            target_chars if target_chars is not None else {"a", "e", "i", "o", "u", "s", "c"}
        )

    @property
    def name(self) -> str:
        return "homoglyph_targeted"

    def generate(self, text: str, num_variants: int = 1) -> list[EvasionResult]:
        """Generate targeted homoglyph variants."""
        results = []

        for _ in range(num_variants):
            evasion = self._apply_targeted_substitution(text)
            confidence = 0.75 if evasion != text else 0.1
            results.append(
                EvasionResult(
                    original=text,
                    evasion=evasion,
                    strategy=self.name,
                    confidence=confidence,
                )
            )

        return results

    def _apply_targeted_substitution(self, text: str) -> str:
        """Apply substitution only to target characters."""
        chars = list(text)

        for i, char in enumerate(chars):
            lower_char = char.lower()
            if lower_char in self._target_chars and char in HOMOGLYPH_MAP:
                homoglyphs = HOMOGLYPH_MAP[char]
                chars[i] = random.choice(homoglyphs)

        return "".join(chars)
