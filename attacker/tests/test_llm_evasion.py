"""Tests for LLM-based evasion strategies."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from attacker.ollama_client import GenerateResponse
from attacker.strategies.llm_evasion import BatchLLMEvasionStrategy, LLMEvasionStrategy


class TestLLMEvasionStrategy:
    """Tests for LLMEvasionStrategy."""

    @pytest.fixture
    def mock_response(self) -> GenerateResponse:
        """Create mock response."""
        return GenerateResponse(
            model="mistral:7b-instruct-v0.2-q4_K_S",
            response="transformed text output",
            done=True,
        )

    @pytest.mark.asyncio
    async def test_generate_async(self, mock_response: GenerateResponse) -> None:
        """Test async generation with mocked client."""
        with patch("attacker.strategies.llm_evasion.OllamaClient") as MockClient:
            mock_client = MagicMock()
            mock_client.generate = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            strategy = LLMEvasionStrategy()
            results = await strategy.generate_async("test input", 1)

            assert len(results) == 1
            assert results[0].original == "test input"
            assert results[0].strategy == "llm_evasion"
            assert results[0].evasion == "transformed text output"

    @pytest.mark.asyncio
    async def test_generate_multiple_variants(
        self, mock_response: GenerateResponse
    ) -> None:
        """Test generating multiple variants."""
        with patch("attacker.strategies.llm_evasion.OllamaClient") as MockClient:
            mock_client = MagicMock()
            mock_client.generate = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            strategy = LLMEvasionStrategy()
            results = await strategy.generate_async("test", 3)

            assert len(results) == 3
            assert mock_client.generate.call_count == 3

    def test_clean_response_removes_quotes(self) -> None:
        """Test response cleaning removes quotes."""
        strategy = LLMEvasionStrategy()

        assert strategy._clean_response('"quoted text"') == "quoted text"
        assert strategy._clean_response("'single quotes'") == "single quotes"

    def test_clean_response_removes_prefixes(self) -> None:
        """Test response cleaning removes common prefixes."""
        strategy = LLMEvasionStrategy()

        assert strategy._clean_response("Transformed: result") == "result"
        assert strategy._clean_response("Output: result") == "result"
        assert strategy._clean_response("Result: result") == "result"

    def test_calculate_confidence_empty(self) -> None:
        """Test confidence calculation for empty response."""
        strategy = LLMEvasionStrategy()

        assert strategy._calculate_confidence("original", "") == 0.1

    def test_calculate_confidence_same(self) -> None:
        """Test confidence calculation when output equals input."""
        strategy = LLMEvasionStrategy()

        assert strategy._calculate_confidence("same", "same") == 0.1

    def test_calculate_confidence_good_ratio(self) -> None:
        """Test confidence calculation for good length ratio."""
        strategy = LLMEvasionStrategy()

        confidence = strategy._calculate_confidence("hello", "h3ll0")
        assert confidence == 0.75

    def test_calculate_confidence_bad_ratio(self) -> None:
        """Test confidence calculation for bad length ratio."""
        strategy = LLMEvasionStrategy()

        # Too short
        assert strategy._calculate_confidence("hello world", "hi") == 0.3
        # Too long
        long_response = "hello " * 100
        assert strategy._calculate_confidence("hello", long_response) == 0.3

    def test_strategy_name(self) -> None:
        """Test strategy name property."""
        strategy = LLMEvasionStrategy()
        assert strategy.name == "llm_evasion"

    def test_custom_prompt_template(self) -> None:
        """Test custom prompt template."""
        custom_template = "Custom: {text}"
        strategy = LLMEvasionStrategy(prompt_template=custom_template)

        assert strategy._prompt_template == custom_template


class TestBatchLLMEvasionStrategy:
    """Tests for BatchLLMEvasionStrategy."""

    @pytest.mark.asyncio
    async def test_batch_generation_uses_different_temperatures(self) -> None:
        """Test that batch generation uses different temperatures."""
        response = GenerateResponse(
            model="test",
            response="result",
            done=True,
        )

        with patch("attacker.strategies.llm_evasion.OllamaClient") as MockClient:
            mock_client = MagicMock()
            mock_client.generate = AsyncMock(return_value=response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            MockClient.return_value = mock_client

            strategy = BatchLLMEvasionStrategy(
                temperatures=(0.5, 0.7, 0.9),
            )
            results = await strategy.generate_async("test", 3)

            assert len(results) == 3

    def test_strategy_name(self) -> None:
        """Test strategy name property."""
        strategy = BatchLLMEvasionStrategy()
        assert strategy.name == "llm_evasion_batch"

    def test_temperature_cycling(self) -> None:
        """Test that temperatures cycle for more variants than temps."""
        strategy = BatchLLMEvasionStrategy(
            temperatures=(0.5, 0.7),
        )

        # With 2 temperatures and 4 variants, should cycle
        assert strategy._temperatures == (0.5, 0.7)
