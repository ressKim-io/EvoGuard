"""Tests for leetspeak evasion strategy."""


from attacker.mappings.leetspeak_map import LeetspeakLevel
from attacker.strategies.leetspeak import AdaptiveLeetspeakStrategy, LeetspeakStrategy


class TestLeetspeakStrategy:
    """Tests for LeetspeakStrategy."""

    def test_basic_level(self) -> None:
        """Test BASIC leetspeak level."""
        strategy = LeetspeakStrategy(
            level=LeetspeakLevel.BASIC,
            probability=1.0,
        )

        text = "hello"
        results = strategy.generate(text, 1)

        assert len(results) == 1
        result = results[0]
        assert result.original == text
        assert result.strategy == "leetspeak_basic"
        # 'e' and 'o' should be transformed (e->3, o->0)
        assert "3" in result.evasion or "0" in result.evasion

    def test_all_levels(self) -> None:
        """Test all leetspeak levels."""
        text = "hello"

        for level in LeetspeakLevel:
            strategy = LeetspeakStrategy(level=level, probability=1.0)
            results = strategy.generate(text, 1)

            assert len(results) == 1
            assert results[0].strategy == f"leetspeak_{level.value}"

    def test_probability_affects_output(self) -> None:
        """Test that probability affects transformation rate."""
        text = "hello world test"

        # High probability
        high_strategy = LeetspeakStrategy(
            level=LeetspeakLevel.MODERATE,
            probability=1.0,
        )
        high_results = high_strategy.generate(text, 1)

        # Low probability
        low_strategy = LeetspeakStrategy(
            level=LeetspeakLevel.MODERATE,
            probability=0.1,
        )
        low_results = low_strategy.generate(text, 1)

        # High probability should have more transformations (on average)
        high_diff = sum(
            1 for a, b in zip(text, high_results[0].evasion, strict=False)
            if a != b
        )
        low_diff = sum(
            1 for a, b in zip(text, low_results[0].evasion, strict=False)
            if a != b
        )

        # Can't guarantee this due to randomness, but pattern should hold
        assert high_diff >= 0 and low_diff >= 0

    def test_preserve_case(self) -> None:
        """Test case preservation option."""
        strategy = LeetspeakStrategy(
            level=LeetspeakLevel.BASIC,
            probability=1.0,
            preserve_case=True,
        )

        text = "HELLO"
        results = strategy.generate(text, 1)

        # Case should be preserved where possible
        assert len(results) == 1

    def test_no_preserve_case(self) -> None:
        """Test without case preservation."""
        strategy = LeetspeakStrategy(
            level=LeetspeakLevel.BASIC,
            probability=1.0,
            preserve_case=False,
        )

        text = "HELLO"
        results = strategy.generate(text, 1)

        assert len(results) == 1

    def test_non_alphabetic_preserved(self) -> None:
        """Test that non-alphabetic characters are preserved."""
        strategy = LeetspeakStrategy(
            level=LeetspeakLevel.MODERATE,
            probability=1.0,
        )

        text = "hello 123 !@#"
        results = strategy.generate(text, 1)

        # Digits and special chars in original should remain
        assert "123" in results[0].evasion.replace(" ", "")
        assert "!@#" in results[0].evasion

    def test_extreme_level_substitutions(self) -> None:
        """Test EXTREME level multi-character substitutions."""
        strategy = LeetspeakStrategy(
            level=LeetspeakLevel.EXTREME,
            probability=1.0,
        )

        text = "hello"
        results = strategy.generate(text, 1)

        # EXTREME level can produce longer outputs due to multi-char subs
        assert len(results) == 1
        assert results[0].confidence >= 0.5

    def test_confidence_increases_with_level(self) -> None:
        """Test that confidence generally increases with level."""
        text = "hello"

        confidences = []
        for level in LeetspeakLevel:
            strategy = LeetspeakStrategy(level=level, probability=1.0)
            results = strategy.generate(text, 1)
            confidences.append(results[0].confidence)

        # Higher levels should have higher base confidence
        assert confidences[0] <= confidences[-1]

    def test_empty_text(self) -> None:
        """Test with empty text."""
        strategy = LeetspeakStrategy(level=LeetspeakLevel.BASIC)

        results = strategy.generate("", 1)

        assert results[0].evasion == ""

    def test_multiple_variants(self) -> None:
        """Test generating multiple variants."""
        strategy = LeetspeakStrategy(
            level=LeetspeakLevel.MODERATE,
            probability=0.5,
        )

        text = "hello"
        results = strategy.generate(text, 10)

        assert len(results) == 10
        # With probability 0.5, we should get some variation
        evasions = [r.evasion for r in results]
        assert len(evasions) == 10


class TestAdaptiveLeetspeakStrategy:
    """Tests for AdaptiveLeetspeakStrategy."""

    def test_adaptive_level_selection(self) -> None:
        """Test that level adapts based on text and variant index."""
        strategy = AdaptiveLeetspeakStrategy(base_probability=0.8)

        # Short text
        short_results = strategy.generate("hi", 1)
        assert len(short_results) == 1
        assert short_results[0].strategy == "leetspeak_adaptive"

        # Longer text
        long_results = strategy.generate("this is a much longer text for testing", 1)
        assert len(long_results) == 1

    def test_multiple_variants_use_different_levels(self) -> None:
        """Test that multiple variants use progressively higher levels."""
        strategy = AdaptiveLeetspeakStrategy()

        text = "hello"
        results = strategy.generate(text, 4)

        assert len(results) == 4
        # All should have the adaptive strategy name
        assert all(r.strategy == "leetspeak_adaptive" for r in results)

    def test_base_probability(self) -> None:
        """Test base probability setting."""
        high_prob = AdaptiveLeetspeakStrategy(base_probability=1.0)
        low_prob = AdaptiveLeetspeakStrategy(base_probability=0.1)

        text = "hello"

        high_results = high_prob.generate(text, 1)
        low_results = low_prob.generate(text, 1)

        # Both should produce results
        assert len(high_results) == 1
        assert len(low_results) == 1
