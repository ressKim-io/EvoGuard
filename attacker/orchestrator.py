"""Strategy orchestrator for coordinating multiple evasion strategies."""

import asyncio
import logging
import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from attacker.mappings.leetspeak_map import LeetspeakLevel
from attacker.ollama_client import OllamaSettings
from attacker.strategies.adversarial_llm import AdversarialLLMStrategy
from attacker.strategies.base import AttackStrategy, EvasionResult
from attacker.strategies.homoglyph import HomoglyphStrategy
from attacker.strategies.leetspeak import LeetspeakStrategy
from attacker.strategies.llm_evasion import LLMEvasionStrategy
from attacker.strategies.unicode_evasion import (
    UnicodeEvasionStrategy,
    UnicodeEvasionType,
)

logger = logging.getLogger(__name__)


class SelectionMode(Enum):
    """Strategy selection mode."""

    ROUND_ROBIN = "round_robin"  # Cycle through strategies
    WEIGHTED = "weighted"  # Select based on weights
    RANDOM = "random"  # Random selection
    ALL = "all"  # Use all strategies


@dataclass
class StrategyConfig:
    """Configuration for a single strategy."""

    strategy: AttackStrategy
    weight: float = 1.0
    enabled: bool = True


@dataclass
class OrchestratorConfig:
    """Orchestrator configuration."""

    selection_mode: SelectionMode = SelectionMode.WEIGHTED
    include_llm: bool = True
    ollama_settings: OllamaSettings | None = None
    default_variants_per_strategy: int = 1


@dataclass
class OrchestratorResult:
    """Result from orchestrator execution."""

    results: list[EvasionResult]
    strategies_used: list[str]
    total_variants: int

    @property
    def best_result(self) -> EvasionResult | None:
        """Return result with highest confidence."""
        if not self.results:
            return None
        return max(self.results, key=lambda r: r.confidence)

    @property
    def unique_evasions(self) -> set[str]:
        """Return set of unique evasion texts."""
        return {r.evasion for r in self.results}


class AttackOrchestrator:
    """Orchestrates multiple evasion strategies.

    Supports different selection modes:
    - ROUND_ROBIN: Cycles through strategies in order
    - WEIGHTED: Selects strategies based on configured weights
    - RANDOM: Random strategy selection
    - ALL: Uses all enabled strategies
    """

    def __init__(self, config: OrchestratorConfig | None = None) -> None:
        """Initialize orchestrator.

        Args:
            config: Orchestrator configuration
        """
        self._config = config or OrchestratorConfig()
        self._strategies: list[StrategyConfig] = []
        self._round_robin_idx = 0

        # Initialize default strategies
        self._init_default_strategies()

    def _init_default_strategies(self) -> None:
        """Initialize default set of strategies."""
        # Rule-based strategies
        for evasion_type in UnicodeEvasionType:
            self._strategies.append(
                StrategyConfig(
                    strategy=UnicodeEvasionStrategy(evasion_type=evasion_type),
                    weight=1.0,
                )
            )

        self._strategies.append(
            StrategyConfig(strategy=HomoglyphStrategy(), weight=1.5)
        )

        for level in LeetspeakLevel:
            weight = 1.0 + (list(LeetspeakLevel).index(level) * 0.2)
            self._strategies.append(
                StrategyConfig(
                    strategy=LeetspeakStrategy(level=level),
                    weight=weight,
                )
            )

        # LLM-based strategies (if enabled)
        if self._config.include_llm:
            self._strategies.append(
                StrategyConfig(
                    strategy=LLMEvasionStrategy(
                        settings=self._config.ollama_settings
                    ),
                    weight=2.0,
                )
            )
            self._strategies.append(
                StrategyConfig(
                    strategy=AdversarialLLMStrategy(
                        settings=self._config.ollama_settings
                    ),
                    weight=2.5,
                )
            )

    @property
    def strategies(self) -> list[StrategyConfig]:
        """Return list of configured strategies."""
        return self._strategies

    @property
    def enabled_strategies(self) -> list[StrategyConfig]:
        """Return list of enabled strategies."""
        return [s for s in self._strategies if s.enabled]

    def add_strategy(
        self,
        strategy: AttackStrategy,
        weight: float = 1.0,
        enabled: bool = True,
    ) -> None:
        """Add a custom strategy.

        Args:
            strategy: Strategy instance
            weight: Selection weight
            enabled: Whether strategy is enabled
        """
        self._strategies.append(
            StrategyConfig(strategy=strategy, weight=weight, enabled=enabled)
        )

    def remove_strategy(self, name: str) -> bool:
        """Remove strategy by name.

        Args:
            name: Strategy name

        Returns:
            True if strategy was removed
        """
        for i, config in enumerate(self._strategies):
            if config.strategy.name == name:
                self._strategies.pop(i)
                return True
        return False

    def set_strategy_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a strategy.

        Args:
            name: Strategy name
            enabled: Whether to enable

        Returns:
            True if strategy was found and updated
        """
        for config in self._strategies:
            if config.strategy.name == name:
                config.enabled = enabled
                return True
        return False

    def generate(
        self,
        text: str,
        num_variants: int = 5,
        strategies: Sequence[str] | None = None,
    ) -> OrchestratorResult:
        """Generate evasion variants using configured strategies.

        Args:
            text: Input text to transform
            num_variants: Total number of variants to generate
            strategies: Optional list of strategy names to use

        Returns:
            OrchestratorResult with all generated variants
        """
        # Filter strategies if specific ones requested
        if strategies:
            available = [
                s for s in self.enabled_strategies
                if s.strategy.name in strategies
            ]
        else:
            available = self.enabled_strategies

        if not available:
            logger.warning("No strategies available")
            return OrchestratorResult(
                results=[],
                strategies_used=[],
                total_variants=0,
            )

        # Select strategies based on mode
        selected = self._select_strategies(available, num_variants)

        # Generate variants
        results = []
        strategies_used = set()

        for strategy_config, variant_count in selected:
            try:
                variants = strategy_config.strategy.generate(text, variant_count)
                results.extend(variants)
                strategies_used.add(strategy_config.strategy.name)
            except Exception as e:
                logger.error(f"Strategy {strategy_config.strategy.name} failed: {e}")

        return OrchestratorResult(
            results=results,
            strategies_used=list(strategies_used),
            total_variants=len(results),
        )

    async def generate_async(
        self,
        text: str,
        num_variants: int = 5,
        strategies: Sequence[str] | None = None,
    ) -> OrchestratorResult:
        """Generate evasion variants asynchronously.

        Args:
            text: Input text
            num_variants: Number of variants
            strategies: Optional strategy names

        Returns:
            OrchestratorResult
        """
        # For now, run synchronously in executor
        # Future: implement native async for LLM strategies
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.generate(text, num_variants, strategies),
        )

    def _select_strategies(
        self,
        available: list[StrategyConfig],
        num_variants: int,
    ) -> list[tuple[StrategyConfig, int]]:
        """Select strategies based on configured mode.

        Returns:
            List of (strategy_config, num_variants) tuples
        """
        match self._config.selection_mode:
            case SelectionMode.ALL:
                return self._select_all(available, num_variants)
            case SelectionMode.ROUND_ROBIN:
                return self._select_round_robin(available, num_variants)
            case SelectionMode.WEIGHTED:
                return self._select_weighted(available, num_variants)
            case SelectionMode.RANDOM:
                return self._select_random(available, num_variants)

    def _select_all(
        self,
        available: list[StrategyConfig],
        num_variants: int,
    ) -> list[tuple[StrategyConfig, int]]:
        """Use all strategies, distributing variants evenly."""
        variants_per = max(1, num_variants // len(available))
        return [(s, variants_per) for s in available]

    def _select_round_robin(
        self,
        available: list[StrategyConfig],
        num_variants: int,
    ) -> list[tuple[StrategyConfig, int]]:
        """Cycle through strategies."""
        selected: list[tuple[StrategyConfig, int]] = []
        for i in range(num_variants):
            idx = (self._round_robin_idx + i) % len(available)
            strategy = available[idx]

            # Check if already in selected
            found = False
            for j, (s, count) in enumerate(selected):
                if s.strategy.name == strategy.strategy.name:
                    selected[j] = (s, count + 1)
                    found = True
                    break

            if not found:
                selected.append((strategy, 1))

        self._round_robin_idx = (self._round_robin_idx + num_variants) % len(available)
        return selected

    def _select_weighted(
        self,
        available: list[StrategyConfig],
        num_variants: int,
    ) -> list[tuple[StrategyConfig, int]]:
        """Select based on weights."""
        total_weight = sum(s.weight for s in available)

        selected = []
        for strategy in available:
            # Calculate variants based on weight proportion
            proportion = strategy.weight / total_weight
            count = max(1, int(num_variants * proportion))
            selected.append((strategy, count))

        return selected

    def _select_random(
        self,
        available: list[StrategyConfig],
        num_variants: int,
    ) -> list[tuple[StrategyConfig, int]]:
        """Random selection with replacement."""
        weights = [s.weight for s in available]
        selections = random.choices(available, weights=weights, k=num_variants)

        # Count selections
        counts: dict[str, tuple[StrategyConfig, int]] = {}
        for strategy in selections:
            name = strategy.strategy.name
            if name in counts:
                s, c = counts[name]
                counts[name] = (s, c + 1)
            else:
                counts[name] = (strategy, 1)

        return list(counts.values())


# Convenience function for quick usage
def create_default_orchestrator(
    include_llm: bool = True,
    selection_mode: SelectionMode = SelectionMode.WEIGHTED,
) -> AttackOrchestrator:
    """Create an orchestrator with default configuration.

    Args:
        include_llm: Whether to include LLM-based strategies
        selection_mode: Strategy selection mode

    Returns:
        Configured AttackOrchestrator
    """
    config = OrchestratorConfig(
        selection_mode=selection_mode,
        include_llm=include_llm,
    )
    return AttackOrchestrator(config)
