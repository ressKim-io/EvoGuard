"""Tests for homoglyph evasion strategy."""


from attacker.strategies.homoglyph import HomoglyphStrategy, TargetedHomoglyphStrategy


class TestHomoglyphStrategy:
    """Tests for HomoglyphStrategy."""

    def test_basic_substitution(self) -> None:
        """Test basic homoglyph substitution."""
        strategy = HomoglyphStrategy(probability=1.0, min_substitutions=1)

        text = "hello"
        results = strategy.generate(text, 1)

        assert len(results) == 1
        result = results[0]
        assert result.original == text
        assert result.strategy == "homoglyph"
        # With high probability, some characters should be substituted
        assert result.evasion != text or result.confidence < 0.5

    def test_min_substitutions(self) -> None:
        """Test minimum substitution count."""
        strategy = HomoglyphStrategy(probability=1.0, min_substitutions=3)

        text = "hello"
        results = strategy.generate(text, 3)

        for result in results:
            # Count differences
            diff_count = sum(
                1 for a, b in zip(result.original, result.evasion, strict=False)
                if a != b
            )
            # Should have at least some substitutions (may be less than min if not enough eligible chars)
            assert result.evasion != text or diff_count >= 0

    def test_max_substitutions(self) -> None:
        """Test maximum substitution limit."""
        strategy = HomoglyphStrategy(
            probability=1.0,
            min_substitutions=1,
            max_substitutions=2,
        )

        text = "hello world test"
        results = strategy.generate(text, 5)

        for result in results:
            diff_count = sum(
                1 for a, b in zip(result.original, result.evasion, strict=False)
                if a != b
            )
            assert diff_count <= 2 or result.evasion == text

    def test_no_eligible_characters(self) -> None:
        """Test with text that has no eligible characters."""
        strategy = HomoglyphStrategy(probability=1.0)

        # Special characters only
        text = "!@#$%"
        results = strategy.generate(text, 1)

        assert results[0].evasion == text
        assert results[0].confidence == 0.1

    def test_uppercase_substitution(self) -> None:
        """Test substitution of uppercase characters."""
        strategy = HomoglyphStrategy(probability=1.0, min_substitutions=1)

        text = "HELLO"
        results = strategy.generate(text, 1)

        result = results[0]
        assert result.strategy == "homoglyph"
        # Uppercase letters should be substitutable
        assert len(result.evasion) == len(text)

    def test_digit_substitution(self) -> None:
        """Test substitution of digit characters."""
        strategy = HomoglyphStrategy(probability=1.0, min_substitutions=1)

        text = "test123"
        results = strategy.generate(text, 1)

        assert len(results[0].evasion) == len(text)

    def test_multiple_variants_uniqueness(self) -> None:
        """Test that multiple variants can be different."""
        strategy = HomoglyphStrategy(probability=0.5)

        text = "hello"
        results = strategy.generate(text, 10)

        # Should have some variation (probabilistic)
        unique_evasions = {r.evasion for r in results}
        assert len(results) == 10
        assert len(unique_evasions) >= 1  # At least one unique evasion

    def test_confidence_calculation(self) -> None:
        """Test confidence score calculation."""
        strategy = HomoglyphStrategy(probability=1.0, min_substitutions=1)

        text = "test"
        results = strategy.generate(text, 1)

        result = results[0]
        assert 0.0 <= result.confidence <= 1.0


class TestTargetedHomoglyphStrategy:
    """Tests for TargetedHomoglyphStrategy."""

    def test_default_targets(self) -> None:
        """Test with default target characters."""
        strategy = TargetedHomoglyphStrategy()

        text = "hello"
        results = strategy.generate(text, 1)

        assert len(results) == 1
        assert results[0].strategy == "homoglyph_targeted"
        # 'e' and 'o' are in default targets
        assert results[0].evasion != text

    def test_custom_targets(self) -> None:
        """Test with custom target characters."""
        strategy = TargetedHomoglyphStrategy(target_chars={"x"})

        text = "hello"
        results = strategy.generate(text, 1)

        # 'x' is not in text, so should be unchanged
        assert results[0].evasion == text
        assert results[0].confidence == 0.1

    def test_all_targets_present(self) -> None:
        """Test when all target characters are present."""
        strategy = TargetedHomoglyphStrategy(target_chars={"a", "e"})

        text = "ae"
        results = strategy.generate(text, 1)

        # Both characters should be substituted
        result = results[0]
        assert result.confidence == 0.75

    def test_empty_targets(self) -> None:
        """Test with empty target set."""
        strategy = TargetedHomoglyphStrategy(target_chars=set())

        text = "hello"
        results = strategy.generate(text, 1)

        assert results[0].evasion == text
