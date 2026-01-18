"""Tests for attack orchestrator."""



from attacker.orchestrator import (
    AttackOrchestrator,
    OrchestratorConfig,
    OrchestratorResult,
    SelectionMode,
    create_default_orchestrator,
)
from attacker.strategies.base import AttackStrategy, EvasionResult


class MockStrategy(AttackStrategy):
    """Mock strategy for testing."""

    def __init__(self, name: str = "mock") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def generate(self, text: str, num_variants: int = 1) -> list[EvasionResult]:
        return [
            EvasionResult(
                original=text,
                evasion=f"{self._name}_{text}_{i}",
                strategy=self._name,
                confidence=0.5,
            )
            for i in range(num_variants)
        ]


class TestAttackOrchestrator:
    """Tests for AttackOrchestrator."""

    def test_init_default_strategies(self) -> None:
        """Test that default strategies are initialized."""
        config = OrchestratorConfig(include_llm=False)
        orchestrator = AttackOrchestrator(config)

        # Should have rule-based strategies
        assert len(orchestrator.strategies) > 0
        strategy_names = [s.strategy.name for s in orchestrator.strategies]

        # Should include unicode, homoglyph, leetspeak
        assert any("unicode" in name for name in strategy_names)
        assert any("homoglyph" in name for name in strategy_names)
        assert any("leetspeak" in name for name in strategy_names)

    def test_init_with_llm(self) -> None:
        """Test initialization with LLM strategies."""
        config = OrchestratorConfig(include_llm=True)
        orchestrator = AttackOrchestrator(config)

        strategy_names = [s.strategy.name for s in orchestrator.strategies]
        assert any("llm" in name for name in strategy_names)

    def test_add_custom_strategy(self) -> None:
        """Test adding a custom strategy."""
        config = OrchestratorConfig(include_llm=False)
        orchestrator = AttackOrchestrator(config)

        initial_count = len(orchestrator.strategies)

        mock = MockStrategy("custom_test")
        orchestrator.add_strategy(mock, weight=2.0)

        assert len(orchestrator.strategies) == initial_count + 1
        assert orchestrator.strategies[-1].strategy.name == "custom_test"
        assert orchestrator.strategies[-1].weight == 2.0

    def test_remove_strategy(self) -> None:
        """Test removing a strategy."""
        config = OrchestratorConfig(include_llm=False)
        orchestrator = AttackOrchestrator(config)

        # Add a mock strategy
        mock = MockStrategy("to_remove")
        orchestrator.add_strategy(mock)

        initial_count = len(orchestrator.strategies)

        result = orchestrator.remove_strategy("to_remove")

        assert result is True
        assert len(orchestrator.strategies) == initial_count - 1

    def test_remove_nonexistent_strategy(self) -> None:
        """Test removing a nonexistent strategy."""
        config = OrchestratorConfig(include_llm=False)
        orchestrator = AttackOrchestrator(config)

        result = orchestrator.remove_strategy("nonexistent")

        assert result is False

    def test_enable_disable_strategy(self) -> None:
        """Test enabling/disabling strategies."""
        config = OrchestratorConfig(include_llm=False)
        orchestrator = AttackOrchestrator(config)

        # Get first strategy name
        first_name = orchestrator.strategies[0].strategy.name

        # Disable it
        result = orchestrator.set_strategy_enabled(first_name, False)

        assert result is True
        assert orchestrator.strategies[0].enabled is False
        assert orchestrator.strategies[0] not in orchestrator.enabled_strategies

        # Re-enable it
        orchestrator.set_strategy_enabled(first_name, True)
        assert orchestrator.strategies[0].enabled is True

    def test_generate_with_all_mode(self) -> None:
        """Test generation with ALL selection mode."""
        config = OrchestratorConfig(
            selection_mode=SelectionMode.ALL,
            include_llm=False,
        )
        orchestrator = AttackOrchestrator(config)

        result = orchestrator.generate("test", num_variants=10)

        assert isinstance(result, OrchestratorResult)
        assert result.total_variants > 0
        assert len(result.strategies_used) > 0

    def test_generate_with_weighted_mode(self) -> None:
        """Test generation with WEIGHTED selection mode."""
        config = OrchestratorConfig(
            selection_mode=SelectionMode.WEIGHTED,
            include_llm=False,
        )
        orchestrator = AttackOrchestrator(config)

        result = orchestrator.generate("test", num_variants=5)

        assert result.total_variants > 0

    def test_generate_with_round_robin_mode(self) -> None:
        """Test generation with ROUND_ROBIN selection mode."""
        config = OrchestratorConfig(
            selection_mode=SelectionMode.ROUND_ROBIN,
            include_llm=False,
        )
        orchestrator = AttackOrchestrator(config)

        result1 = orchestrator.generate("test", num_variants=3)
        result2 = orchestrator.generate("test", num_variants=3)

        assert result1.total_variants > 0
        assert result2.total_variants > 0

    def test_generate_with_random_mode(self) -> None:
        """Test generation with RANDOM selection mode."""
        config = OrchestratorConfig(
            selection_mode=SelectionMode.RANDOM,
            include_llm=False,
        )
        orchestrator = AttackOrchestrator(config)

        result = orchestrator.generate("test", num_variants=5)

        assert result.total_variants > 0

    def test_generate_with_specific_strategies(self) -> None:
        """Test generation with specific strategy selection."""
        config = OrchestratorConfig(include_llm=False)
        orchestrator = AttackOrchestrator(config)

        result = orchestrator.generate(
            "test",
            num_variants=3,
            strategies=["homoglyph"],
        )

        assert all("homoglyph" in s for s in result.strategies_used)

    def test_generate_no_strategies(self) -> None:
        """Test generation when no strategies are available."""
        config = OrchestratorConfig(include_llm=False)
        orchestrator = AttackOrchestrator(config)

        # Disable all strategies
        for strategy_config in orchestrator.strategies:
            strategy_config.enabled = False

        result = orchestrator.generate("test", num_variants=5)

        assert result.total_variants == 0
        assert len(result.strategies_used) == 0

    def test_best_result(self) -> None:
        """Test best_result property of OrchestratorResult."""
        config = OrchestratorConfig(include_llm=False)
        orchestrator = AttackOrchestrator(config)

        result = orchestrator.generate("test", num_variants=10)

        if result.results:
            best = result.best_result
            assert best is not None
            assert all(r.confidence <= best.confidence for r in result.results)

    def test_unique_evasions(self) -> None:
        """Test unique_evasions property of OrchestratorResult."""
        config = OrchestratorConfig(include_llm=False)
        orchestrator = AttackOrchestrator(config)

        result = orchestrator.generate("test", num_variants=10)

        unique = result.unique_evasions
        assert isinstance(unique, set)
        assert len(unique) <= len(result.results)


class TestCreateDefaultOrchestrator:
    """Tests for create_default_orchestrator function."""

    def test_create_without_llm(self) -> None:
        """Test creating orchestrator without LLM."""
        orchestrator = create_default_orchestrator(include_llm=False)

        strategy_names = [s.strategy.name for s in orchestrator.strategies]
        assert not any("llm" in name for name in strategy_names)

    def test_create_with_llm(self) -> None:
        """Test creating orchestrator with LLM."""
        orchestrator = create_default_orchestrator(include_llm=True)

        strategy_names = [s.strategy.name for s in orchestrator.strategies]
        assert any("llm" in name for name in strategy_names)

    def test_create_with_selection_mode(self) -> None:
        """Test creating orchestrator with specific selection mode."""
        for mode in SelectionMode:
            orchestrator = create_default_orchestrator(
                include_llm=False,
                selection_mode=mode,
            )
            assert orchestrator._config.selection_mode == mode
