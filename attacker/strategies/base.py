"""Base classes for attack strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EvasionResult:
    """Result of an evasion attempt.

    Attributes:
        original: Original input text
        evasion: Transformed text attempting to evade detection
        strategy: Name of the strategy used
        confidence: Confidence score (0.0 - 1.0) of evasion success likelihood
    """

    original: str
    evasion: str
    strategy: str
    confidence: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")


class AttackStrategy(ABC):
    """Abstract base class for attack strategies.

    All evasion strategies must implement the generate method.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the strategy name."""
        ...

    @abstractmethod
    def generate(self, text: str, num_variants: int = 1) -> list[EvasionResult]:
        """Generate evasion variants for the given text.

        Args:
            text: Input text to transform
            num_variants: Number of variants to generate

        Returns:
            List of EvasionResult objects
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
