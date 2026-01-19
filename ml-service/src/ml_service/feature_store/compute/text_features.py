"""Text feature transformer for extracting text characteristics."""

from typing import Any

from ml_service.feature_store.compute.base import FeatureTransformer


class TextFeatureTransformer(FeatureTransformer):
    """Extract statistical features from text data.

    Features extracted:
        - text_length: Total character count
        - word_count: Number of words (whitespace-separated)
        - unicode_ratio: Ratio of non-ASCII characters
        - special_char_ratio: Ratio of special characters
        - repeated_char_ratio: Ratio of consecutively repeated characters (3+)

    Example:
        >>> transformer = TextFeatureTransformer()
        >>> features = transformer.transform("Hello World!")
        >>> print(features)
        {'text_length': 12, 'word_count': 2, 'unicode_ratio': 0.0, ...}
    """

    @property
    def feature_names(self) -> list[str]:
        """List of features produced by this transformer."""
        return [
            "text_length",
            "word_count",
            "unicode_ratio",
            "special_char_ratio",
            "repeated_char_ratio",
        ]

    @property
    def entity_type(self) -> str:
        """Entity type for text features."""
        return "text"

    @property
    def version(self) -> str:
        """Transformer version."""
        return "1.0.0"

    def transform(self, data: Any) -> dict[str, Any]:
        """Extract features from text.

        Args:
            data: Input text string. If not a string, will be converted.

        Returns:
            Dictionary with computed feature values.
        """
        # Handle non-string input
        text = str(data) if data is not None else ""

        if not text:
            return dict.fromkeys(self.feature_names, 0)

        length = len(text)
        words = text.split()

        # Unicode ratio: non-ASCII characters / total
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        unicode_ratio = non_ascii_count / length if length > 0 else 0.0

        # Special character ratio: non-alphanumeric, non-whitespace / total
        special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())
        special_ratio = special_count / length if length > 0 else 0.0

        # Repeated character ratio: characters in runs of 3+ / total
        repeated_count = self._count_repeated_chars(text)
        repeated_ratio = repeated_count / length if length > 0 else 0.0

        return {
            "text_length": length,
            "word_count": len(words),
            "unicode_ratio": round(unicode_ratio, 4),
            "special_char_ratio": round(special_ratio, 4),
            "repeated_char_ratio": round(repeated_ratio, 4),
        }

    def _count_repeated_chars(self, text: str) -> int:
        """Count characters that are part of runs of 3 or more.

        Args:
            text: Input text to analyze.

        Returns:
            Number of characters in consecutive runs of 3+.
        """
        if not text:
            return 0

        count = 0
        i = 0

        while i < len(text):
            # Find the end of the current run
            j = i
            while j < len(text) and text[j] == text[i]:
                j += 1

            # If run length is 3 or more, count all characters
            run_length = j - i
            if run_length >= 3:
                count += run_length

            i = j

        return count

    def get_feature_schema(self) -> dict[str, str]:
        """Get schema with feature types."""
        return {
            "text_length": "int",
            "word_count": "int",
            "unicode_ratio": "float",
            "special_char_ratio": "float",
            "repeated_char_ratio": "float",
        }

    def transform_batch(self, data_list: list[Any]) -> list[dict[str, Any]]:
        """Transform a batch of texts.

        Optimized batch processing - same as default but can be
        extended for vectorized operations if needed.

        Args:
            data_list: List of text strings.

        Returns:
            List of feature dictionaries.
        """
        return [self.transform(text) for text in data_list]
