"""Unicode-based evasion strategies."""
# ruff: noqa: RUF002

import random
from enum import Enum

from attacker.mappings.unicode_map import (
    COMBINING_DIACRITICS,
    FULLWIDTH_MAP,
    ZERO_WIDTH_CHARS,
    decompose_korean_syllable,
)
from attacker.strategies.base import AttackStrategy, EvasionResult


class UnicodeEvasionType(Enum):
    """Unicode evasion technique types."""

    JAMO_DECOMPOSE = "jamo_decompose"  # Korean syllable decomposition
    SPACE_INSERT = "space_insert"  # Insert spaces between chars
    ZERO_WIDTH = "zero_width"  # Insert zero-width characters
    FULLWIDTH = "fullwidth"  # Convert to fullwidth characters
    COMBINING_MARKS = "combining_marks"  # Add combining diacritical marks


class UnicodeEvasionStrategy(AttackStrategy):
    """Unicode manipulation strategy for content evasion.

    Implements 5 techniques:
    1. Jamo decomposition: 바보 -> ㅂㅏㅂㅗ
    2. Space insertion: 바보 -> 바 보
    3. Zero-width insertion: invisible unicode characters
    4. Fullwidth conversion: bad -> ｂａｄ
    5. Combining marks: bad -> b̈äd̈
    """

    def __init__(
        self,
        evasion_type: UnicodeEvasionType = UnicodeEvasionType.JAMO_DECOMPOSE,
        probability: float = 1.0,
    ) -> None:
        """Initialize strategy.

        Args:
            evasion_type: Type of unicode evasion to apply
            probability: Probability of transforming each character (0.0 - 1.0)
        """
        self._evasion_type = evasion_type
        self._probability = min(1.0, max(0.0, probability))

    @property
    def name(self) -> str:
        return f"unicode_{self._evasion_type.value}"

    def generate(self, text: str, num_variants: int = 1) -> list[EvasionResult]:
        """Generate unicode evasion variants."""
        results = []

        for _ in range(num_variants):
            evasion = self._apply_evasion(text)
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

    def _apply_evasion(self, text: str) -> str:
        """Apply the configured evasion technique."""
        match self._evasion_type:
            case UnicodeEvasionType.JAMO_DECOMPOSE:
                return self._jamo_decompose(text)
            case UnicodeEvasionType.SPACE_INSERT:
                return self._space_insert(text)
            case UnicodeEvasionType.ZERO_WIDTH:
                return self._zero_width_insert(text)
            case UnicodeEvasionType.FULLWIDTH:
                return self._fullwidth_convert(text)
            case UnicodeEvasionType.COMBINING_MARKS:
                return self._combining_marks(text)

    def _jamo_decompose(self, text: str) -> str:
        """Decompose Korean syllables into Jamo components."""
        result = []
        for char in text:
            if random.random() < self._probability:
                decomposed = decompose_korean_syllable(char)
                result.append(decomposed if decomposed else char)
            else:
                result.append(char)
        return "".join(result)

    def _space_insert(self, text: str) -> str:
        """Insert spaces between characters."""
        result = []
        for i, char in enumerate(text):
            result.append(char)
            if (
                i < len(text) - 1
                and random.random() < self._probability
                and not char.isspace()
                and not text[i + 1].isspace()
            ):
                result.append(" ")
        return "".join(result)

    def _zero_width_insert(self, text: str) -> str:
        """Insert zero-width characters."""
        result = []
        for i, char in enumerate(text):
            result.append(char)
            if i < len(text) - 1 and random.random() < self._probability:
                zwc = random.choice(ZERO_WIDTH_CHARS)
                result.append(zwc)
        return "".join(result)

    def _fullwidth_convert(self, text: str) -> str:
        """Convert ASCII to fullwidth characters."""
        result = []
        for char in text:
            if random.random() < self._probability and char in FULLWIDTH_MAP:
                result.append(FULLWIDTH_MAP[char])
            else:
                result.append(char)
        return "".join(result)

    def _combining_marks(self, text: str) -> str:
        """Add combining diacritical marks to characters."""
        result = []
        for char in text:
            result.append(char)
            if random.random() < self._probability and char.isalpha():
                mark = random.choice(COMBINING_DIACRITICS)
                result.append(mark)
        return "".join(result)

    def _calculate_confidence(self, original: str, evasion: str) -> float:
        """Calculate confidence score based on transformation amount."""
        if original == evasion:
            return 0.1

        # Base confidence by evasion type
        base_confidence = {
            UnicodeEvasionType.JAMO_DECOMPOSE: 0.7,
            UnicodeEvasionType.SPACE_INSERT: 0.5,
            UnicodeEvasionType.ZERO_WIDTH: 0.8,
            UnicodeEvasionType.FULLWIDTH: 0.6,
            UnicodeEvasionType.COMBINING_MARKS: 0.65,
        }

        return base_confidence.get(self._evasion_type, 0.5)
