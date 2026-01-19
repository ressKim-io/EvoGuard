"""Tests for TextFeatureTransformer."""

import pytest

from ml_service.feature_store.compute.text_features import TextFeatureTransformer


@pytest.fixture
def transformer() -> TextFeatureTransformer:
    """Create a TextFeatureTransformer instance."""
    return TextFeatureTransformer()


class TestTextFeatureTransformer:
    """Tests for TextFeatureTransformer."""

    def test_feature_names(self, transformer: TextFeatureTransformer) -> None:
        """Test that feature_names returns expected list."""
        expected = [
            "text_length",
            "word_count",
            "unicode_ratio",
            "special_char_ratio",
            "repeated_char_ratio",
        ]
        assert transformer.feature_names == expected

    def test_entity_type(self, transformer: TextFeatureTransformer) -> None:
        """Test that entity_type is 'text'."""
        assert transformer.entity_type == "text"

    def test_version(self, transformer: TextFeatureTransformer) -> None:
        """Test that version is defined."""
        assert transformer.version == "1.0.0"

    def test_transform_simple_text(self, transformer: TextFeatureTransformer) -> None:
        """Test transform with simple ASCII text."""
        result = transformer.transform("Hello World")

        assert result["text_length"] == 11
        assert result["word_count"] == 2
        assert result["unicode_ratio"] == 0.0
        assert result["special_char_ratio"] == 0.0
        assert result["repeated_char_ratio"] == 0.0

    def test_transform_empty_text(self, transformer: TextFeatureTransformer) -> None:
        """Test transform with empty string."""
        result = transformer.transform("")

        assert result["text_length"] == 0
        assert result["word_count"] == 0
        assert result["unicode_ratio"] == 0
        assert result["special_char_ratio"] == 0
        assert result["repeated_char_ratio"] == 0

    def test_transform_none(self, transformer: TextFeatureTransformer) -> None:
        """Test transform with None input."""
        result = transformer.transform(None)

        assert result["text_length"] == 0
        assert result["word_count"] == 0

    def test_transform_unicode_text(self, transformer: TextFeatureTransformer) -> None:
        """Test transform with unicode characters."""
        # 10 chars total, 5 are unicode (한글로)
        result = transformer.transform("Hello 한글로")

        assert result["text_length"] == 9
        assert result["word_count"] == 2
        # 3 unicode chars out of 9 = 0.3333
        assert result["unicode_ratio"] == pytest.approx(0.3333, abs=0.01)

    def test_transform_special_chars(self, transformer: TextFeatureTransformer) -> None:
        """Test transform with special characters."""
        result = transformer.transform("Hello!@#$%")

        assert result["text_length"] == 10
        # 5 special chars (!@#$%) out of 10
        assert result["special_char_ratio"] == 0.5

    def test_transform_repeated_chars(self, transformer: TextFeatureTransformer) -> None:
        """Test transform with repeated characters."""
        # "Hellooo" has 3 repeated 'o's
        result = transformer.transform("Hellooo")

        assert result["text_length"] == 7
        # 3 repeated chars out of 7
        assert result["repeated_char_ratio"] == pytest.approx(3 / 7, abs=0.01)

    def test_transform_long_repeat(self, transformer: TextFeatureTransformer) -> None:
        """Test transform with long repeated sequence."""
        # 10 repeated 'a's
        result = transformer.transform("aaaaaaaaaa")

        assert result["text_length"] == 10
        assert result["repeated_char_ratio"] == 1.0

    def test_transform_short_repeat_not_counted(
        self, transformer: TextFeatureTransformer
    ) -> None:
        """Test that repeats of 2 or less are not counted."""
        # "aa" is only 2 repeated, not counted
        result = transformer.transform("aabb")

        assert result["text_length"] == 4
        assert result["repeated_char_ratio"] == 0.0

    def test_transform_batch(self, transformer: TextFeatureTransformer) -> None:
        """Test batch transformation."""
        texts = ["Hello", "World", "Test"]
        results = transformer.transform_batch(texts)

        assert len(results) == 3
        assert results[0]["text_length"] == 5
        assert results[1]["text_length"] == 5
        assert results[2]["text_length"] == 4

    def test_get_feature_schema(self, transformer: TextFeatureTransformer) -> None:
        """Test get_feature_schema returns correct types."""
        schema = transformer.get_feature_schema()

        assert schema["text_length"] == "int"
        assert schema["word_count"] == "int"
        assert schema["unicode_ratio"] == "float"
        assert schema["special_char_ratio"] == "float"
        assert schema["repeated_char_ratio"] == "float"

    def test_validate_output_success(self, transformer: TextFeatureTransformer) -> None:
        """Test validate_output with valid features."""
        features = transformer.transform("Hello")
        assert transformer.validate_output(features) is True

    def test_validate_output_missing(self, transformer: TextFeatureTransformer) -> None:
        """Test validate_output with missing features."""
        features = {"text_length": 5}  # Missing other features

        with pytest.raises(ValueError, match="Missing features"):
            transformer.validate_output(features)

    def test_repr(self, transformer: TextFeatureTransformer) -> None:
        """Test string representation."""
        result = repr(transformer)
        assert "TextFeatureTransformer" in result
        assert "entity_type=text" in result
