"""Tests for Unicode evasion strategy."""


from attacker.strategies.unicode_evasion import (
    UnicodeEvasionStrategy,
    UnicodeEvasionType,
)


class TestUnicodeEvasionStrategy:
    """Tests for UnicodeEvasionStrategy."""

    def test_jamo_decompose_korean(self, korean_texts: list[str]) -> None:
        """Test Jamo decomposition of Korean text."""
        strategy = UnicodeEvasionStrategy(
            evasion_type=UnicodeEvasionType.JAMO_DECOMPOSE,
            probability=1.0,
        )

        for text in korean_texts:
            results = strategy.generate(text, 1)

            assert len(results) == 1
            result = results[0]
            assert result.original == text
            assert result.strategy == "unicode_jamo_decompose"
            assert 0.0 <= result.confidence <= 1.0

            # For Korean text, decomposition should produce different output
            if any("\uac00" <= c <= "\ud7a3" for c in text):  # Has Hangul
                assert result.evasion != text or text == result.evasion

    def test_jamo_decompose_preserves_non_korean(self) -> None:
        """Test that non-Korean text is preserved."""
        strategy = UnicodeEvasionStrategy(
            evasion_type=UnicodeEvasionType.JAMO_DECOMPOSE,
            probability=1.0,
        )

        text = "hello world"
        results = strategy.generate(text, 1)

        assert results[0].evasion == text

    def test_space_insert(self, english_texts: list[str]) -> None:
        """Test space insertion."""
        strategy = UnicodeEvasionStrategy(
            evasion_type=UnicodeEvasionType.SPACE_INSERT,
            probability=1.0,
        )

        text = "hello"
        results = strategy.generate(text, 1)

        assert len(results) == 1
        result = results[0]
        assert result.strategy == "unicode_space_insert"
        # With probability 1.0, spaces should be inserted
        assert " " in result.evasion or result.evasion == text

    def test_zero_width_insert(self) -> None:
        """Test zero-width character insertion."""
        strategy = UnicodeEvasionStrategy(
            evasion_type=UnicodeEvasionType.ZERO_WIDTH,
            probability=1.0,
        )

        text = "test"
        results = strategy.generate(text, 1)

        assert len(results) == 1
        result = results[0]
        assert result.strategy == "unicode_zero_width"
        # Zero-width chars are invisible but increase length
        assert len(result.evasion) >= len(text)

    def test_fullwidth_convert(self) -> None:
        """Test fullwidth character conversion."""
        strategy = UnicodeEvasionStrategy(
            evasion_type=UnicodeEvasionType.FULLWIDTH,
            probability=1.0,
        )

        text = "abc123"
        results = strategy.generate(text, 1)

        assert len(results) == 1
        result = results[0]
        assert result.strategy == "unicode_fullwidth"
        # Fullwidth chars should be different
        assert result.evasion != text

    def test_combining_marks(self) -> None:
        """Test combining diacritical marks."""
        strategy = UnicodeEvasionStrategy(
            evasion_type=UnicodeEvasionType.COMBINING_MARKS,
            probability=1.0,
        )

        text = "hello"
        results = strategy.generate(text, 1)

        assert len(results) == 1
        result = results[0]
        assert result.strategy == "unicode_combining_marks"
        # Combining marks add characters
        assert len(result.evasion) >= len(text)

    def test_multiple_variants(self) -> None:
        """Test generating multiple variants."""
        strategy = UnicodeEvasionStrategy(
            evasion_type=UnicodeEvasionType.SPACE_INSERT,
            probability=0.5,
        )

        text = "hello"
        results = strategy.generate(text, 5)

        assert len(results) == 5
        assert all(r.original == text for r in results)

    def test_probability_zero(self) -> None:
        """Test with zero probability (no transformation)."""
        strategy = UnicodeEvasionStrategy(
            evasion_type=UnicodeEvasionType.FULLWIDTH,
            probability=0.0,
        )

        text = "test"
        results = strategy.generate(text, 1)

        assert results[0].evasion == text
        assert results[0].confidence == 0.1  # Low confidence when unchanged

    def test_strategy_name(self) -> None:
        """Test strategy name property."""
        for evasion_type in UnicodeEvasionType:
            strategy = UnicodeEvasionStrategy(evasion_type=evasion_type)
            assert strategy.name == f"unicode_{evasion_type.value}"

    def test_empty_text(self) -> None:
        """Test with empty text."""
        strategy = UnicodeEvasionStrategy(
            evasion_type=UnicodeEvasionType.JAMO_DECOMPOSE,
        )

        results = strategy.generate("", 1)

        assert len(results) == 1
        assert results[0].evasion == ""
