"""LLM-based evasion strategy using Ollama."""

import asyncio
import logging
from pathlib import Path

from attacker.ollama_client import OllamaClient, OllamaError, OllamaSettings
from attacker.strategies.base import AttackStrategy, EvasionResult

logger = logging.getLogger(__name__)

# Load prompt template
_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "evasion.txt"
_PROMPT_TEMPLATE = _PROMPT_PATH.read_text() if _PROMPT_PATH.exists() else """
Transform the following text while preserving its meaning:
Original: "{text}"
Transformed:
"""


class LLMEvasionStrategy(AttackStrategy):
    """LLM-based evasion using Ollama.

    Uses a local LLM (Mistral 7B) to generate creative text variations
    that maintain semantic meaning while potentially evading pattern-based filters.
    """

    def __init__(
        self,
        settings: OllamaSettings | None = None,
        temperature: float = 0.9,
        prompt_template: str | None = None,
    ) -> None:
        """Initialize strategy.

        Args:
            settings: Ollama client settings
            temperature: Sampling temperature (higher = more creative)
            prompt_template: Custom prompt template (use {text} placeholder)
        """
        self._settings = settings or OllamaSettings()
        self._temperature = temperature
        self._prompt_template = prompt_template or _PROMPT_TEMPLATE

    @property
    def name(self) -> str:
        return "llm_evasion"

    def generate(self, text: str, num_variants: int = 1) -> list[EvasionResult]:
        """Generate LLM-based evasion variants (sync wrapper)."""
        return asyncio.run(self.generate_async(text, num_variants))

    async def generate_async(
        self, text: str, num_variants: int = 1
    ) -> list[EvasionResult]:
        """Generate LLM-based evasion variants asynchronously."""
        results = []

        async with OllamaClient(self._settings) as client:
            for _ in range(num_variants):
                try:
                    result = await self._generate_single(client, text)
                    results.append(result)
                except OllamaError as e:
                    logger.error(f"LLM generation failed: {e}")
                    # Return original text with low confidence on error
                    results.append(
                        EvasionResult(
                            original=text,
                            evasion=text,
                            strategy=self.name,
                            confidence=0.0,
                        )
                    )

        return results

    async def _generate_single(
        self, client: OllamaClient, text: str
    ) -> EvasionResult:
        """Generate a single evasion variant."""
        prompt = self._prompt_template.format(text=text)

        response = await client.generate(
            prompt=prompt,
            temperature=self._temperature,
            max_tokens=256,
        )

        evasion = self._clean_response(response.response)

        # Calculate confidence based on response quality
        confidence = self._calculate_confidence(text, evasion)

        return EvasionResult(
            original=text,
            evasion=evasion,
            strategy=self.name,
            confidence=confidence,
        )

    def _clean_response(self, response: str) -> str:
        """Clean and extract the transformed text from LLM response."""
        # Remove common prefixes/suffixes
        cleaned = response.strip()

        # Remove quotes if present
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        if cleaned.startswith("'") and cleaned.endswith("'"):
            cleaned = cleaned[1:-1]

        # Remove any "Transformed:" prefix
        prefixes = ["Transformed:", "Output:", "Result:", "Variation:"]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix) :].strip()

        return cleaned

    def _calculate_confidence(self, original: str, evasion: str) -> float:
        """Calculate confidence based on transformation quality."""
        if not evasion or evasion == original:
            return 0.1

        # Check if response is too short or too long
        len_ratio = len(evasion) / len(original) if original else 0

        if len_ratio < 0.3 or len_ratio > 3.0:
            return 0.3

        # LLM-generated content typically has moderate-high confidence
        # Actual confidence should be validated by the defender model
        return 0.75


class BatchLLMEvasionStrategy(AttackStrategy):
    """Batch LLM evasion for generating multiple variants efficiently."""

    def __init__(
        self,
        settings: OllamaSettings | None = None,
        temperatures: tuple[float, ...] = (0.7, 0.9, 1.1),
    ) -> None:
        """Initialize strategy.

        Args:
            settings: Ollama client settings
            temperatures: Temperature values for different variants
        """
        self._settings = settings or OllamaSettings()
        self._temperatures = temperatures

    @property
    def name(self) -> str:
        return "llm_evasion_batch"

    def generate(self, text: str, num_variants: int = 1) -> list[EvasionResult]:
        """Generate multiple variants with different temperatures."""
        return asyncio.run(self.generate_async(text, num_variants))

    async def generate_async(
        self, text: str, num_variants: int = 1
    ) -> list[EvasionResult]:
        """Generate variants asynchronously."""
        results = []

        for i in range(num_variants):
            temp_idx = i % len(self._temperatures)
            temperature = self._temperatures[temp_idx]

            strategy = LLMEvasionStrategy(
                settings=self._settings,
                temperature=temperature,
            )

            variant_results = await strategy.generate_async(text, 1)
            results.extend(variant_results)

        return results
