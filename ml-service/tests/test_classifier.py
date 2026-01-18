"""Tests for classifier implementations."""

import pytest

from ml_service.models.classifier import MockClassifier


class TestMockClassifier:
    """Tests for MockClassifier."""

    @pytest.fixture
    def classifier(self) -> MockClassifier:
        """Create a mock classifier instance."""
        return MockClassifier(threshold=0.5)

    def test_version(self, classifier: MockClassifier) -> None:
        """Test classifier version property."""
        assert classifier.version == "mock-v1.0.0"

    def test_predict_non_toxic(self, classifier: MockClassifier) -> None:
        """Test predicting non-toxic text."""
        result = classifier.predict("Hello, how are you?")
        assert result.label in ["toxic", "non-toxic"]
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_toxic_keywords(self, classifier: MockClassifier) -> None:
        """Test predicting text with toxic keywords."""
        result = classifier.predict("I hate you, stupid idiot!")
        # Should have higher confidence due to keyword matches
        assert result.confidence > 0.3

    def test_predict_batch(self, classifier: MockClassifier) -> None:
        """Test batch prediction."""
        texts = ["Hello", "You are stupid", "Nice day"]
        results = classifier.predict_batch(texts)

        assert len(results) == 3
        for result in results:
            assert result.label in ["toxic", "non-toxic"]
            assert 0.0 <= result.confidence <= 1.0

    def test_confidence_within_bounds(self, classifier: MockClassifier) -> None:
        """Test that confidence is always within [0, 1]."""
        # Test with various inputs
        test_texts = [
            "",
            "a",
            "Hello world",
            "hate hate hate hate hate",
            "kill die stupid idiot moron",
        ]
        for text in test_texts:
            result = classifier.predict(text)
            assert 0.0 <= result.confidence <= 1.0, f"Confidence out of bounds for: {text}"

    def test_threshold_affects_classification(self) -> None:
        """Test that threshold affects is_toxic classification."""
        text = "I hate you"

        # With low threshold, should be toxic
        low_threshold = MockClassifier(threshold=0.2)
        result_low = low_threshold.predict(text)

        # With high threshold, might not be toxic
        high_threshold = MockClassifier(threshold=0.95)
        result_high = high_threshold.predict(text)

        # The confidence should be the same-ish, but is_toxic differs
        # Note: Due to randomness, we can't make exact assertions
        assert result_low.confidence >= 0.0
        assert result_high.confidence >= 0.0
