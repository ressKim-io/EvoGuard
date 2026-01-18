"""Classifier interface and implementations."""

import random
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class ClassifierResult:
    """Result of a classification."""

    is_toxic: bool
    confidence: float
    label: str


class BaseClassifier(ABC):
    """Abstract base class for text classifiers."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Return the model version."""
        ...

    @abstractmethod
    def predict(self, text: str) -> ClassifierResult:
        """Classify a single text.

        Args:
            text: The text to classify.

        Returns:
            ClassifierResult with is_toxic, confidence, and label.
        """
        ...

    def predict_batch(self, texts: list[str]) -> list[ClassifierResult]:
        """Classify multiple texts.

        Args:
            texts: List of texts to classify.

        Returns:
            List of ClassifierResult objects.
        """
        return [self.predict(text) for text in texts]


class MockClassifier(BaseClassifier):
    """Mock classifier for testing and development.

    Uses simple keyword matching to simulate toxic content detection.
    """

    TOXIC_PATTERNS: ClassVar[list[str]] = [
        r"\b(hate|kill|die|stupid|idiot|moron|dumb)\b",
        r"\b(f[*u]ck|sh[*i]t|damn|ass|bitch)\b",
        r"\b(racist|sexist|homophobic)\b",
    ]

    def __init__(self, threshold: float = 0.5) -> None:
        """Initialize the mock classifier.

        Args:
            threshold: Confidence threshold for classification.
        """
        self._threshold = threshold
        self._patterns = [re.compile(p, re.IGNORECASE) for p in self.TOXIC_PATTERNS]

    @property
    def version(self) -> str:
        """Return the model version."""
        return "mock-v1.0.0"

    def predict(self, text: str) -> ClassifierResult:
        """Classify a single text using keyword matching.

        Args:
            text: The text to classify.

        Returns:
            ClassifierResult with is_toxic, confidence, and label.
        """
        # Count pattern matches
        match_count = sum(1 for p in self._patterns if p.search(text))

        # Calculate confidence based on matches and add some randomness
        base_confidence = min(0.3 + (match_count * 0.25), 0.95)
        noise = random.uniform(-0.1, 0.1)
        confidence = max(0.0, min(1.0, base_confidence + noise))

        # Determine if toxic based on threshold
        is_toxic = confidence >= self._threshold

        return ClassifierResult(
            is_toxic=is_toxic,
            confidence=round(confidence, 4),
            label="toxic" if is_toxic else "non-toxic",
        )
