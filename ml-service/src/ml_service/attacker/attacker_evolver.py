"""Attacker Evolver for Automatic Attack Evolution.

공격자가 약할 때 (evasion < 5%) 자동으로 공격 전략을 진화시킵니다.

Components:
- StrategyMutator: 기존 전략 변형/조합으로 새 전략 생성 (evolver/strategy_mutator.py)
- SlangEvolver: 슬랭 사전 동적 확장 (evolver/slang_evolver.py)
- AttackerEvolver: 메인 클래스 (통합)
"""

from __future__ import annotations

import json
import logging
import random
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Callable

from ml_service.attacker.korean_strategies import KOREAN_ATTACK_STRATEGIES, KoreanAttackStrategy
from ml_service.attacker.evolver import (
    AttackerEvolverConfig,
    EvolutionEvent,
    StrategyMutator,
    SlangEvolver,
)

if TYPE_CHECKING:
    from ml_service.attacker.learning_attacker import LearningAttacker

logger = logging.getLogger(__name__)


class AttackerEvolver:
    """Main class for attacker evolution.

    공격자가 약할 때 자동으로 진화합니다.

    Evolution Modes:
    - aggressive (evasion < 3%): 새 전략 5개, 슬랭 확장, 탐색 +1.0
    - normal (evasion 3-5%): 새 전략 2개, 탐색 +0.3
    - maintenance (evasion 5-8%): 통계 감쇠만
    - none (evasion > 8%): 진화 안함 (방어자 학습)
    """

    def __init__(
        self,
        config: AttackerEvolverConfig | None = None,
        classifier: Any = None,
    ) -> None:
        """Initialize AttackerEvolver.

        Args:
            config: 설정
            classifier: 분류기 (슬랭 검증용)
        """
        self.config = config or AttackerEvolverConfig()
        self.classifier = classifier

        # 컴포넌트
        self._mutator = StrategyMutator()
        self._slang_evolver = SlangEvolver(classifier=classifier)

        # 연결된 LearningAttacker
        self._learning_attacker: LearningAttacker | None = None

        # 진화 기록
        self._evolution_history: list[EvolutionEvent] = []

        # 상태
        self._total_evolutions = 0
        self._last_evolution_mode: str | None = None

        self._load_state()

    def _load_state(self) -> None:
        """Load evolver state."""
        if self.config.state_path.exists():
            try:
                with open(self.config.state_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._total_evolutions = data.get("total_evolutions", 0)
                self._last_evolution_mode = data.get("last_mode")
                self._evolution_history = [
                    EvolutionEvent(**e) for e in data.get("history", [])[-50:]
                ]
            except Exception:
                pass

    def _save_state(self) -> None:
        """Save evolver state."""
        self.config.state_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "total_evolutions": self._total_evolutions,
            "last_mode": self._last_evolution_mode,
            "history": [e.to_dict() for e in self._evolution_history[-50:]],
            "updated_at": datetime.now(UTC).isoformat(),
        }

        with open(self.config.state_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def set_learning_attacker(self, attacker: "LearningAttacker") -> None:
        """Set the associated LearningAttacker.

        Args:
            attacker: LearningAttacker instance
        """
        self._learning_attacker = attacker

    def set_classifier(self, classifier: Any) -> None:
        """Set classifier for validation.

        Args:
            classifier: 분류기
        """
        self.classifier = classifier
        self._slang_evolver.classifier = classifier

    def get_evolution_mode(self, evasion_rate: float) -> str:
        """Determine evolution mode based on evasion rate.

        Args:
            evasion_rate: 현재 evasion rate

        Returns:
            Evolution mode: "aggressive", "normal", "maintenance", or "none"
        """
        if evasion_rate < self.config.aggressive_threshold:
            return "aggressive"
        elif evasion_rate < self.config.normal_threshold:
            return "normal"
        elif evasion_rate < self.config.maintenance_threshold:
            return "maintenance"
        else:
            return "none"

    def evolve(
        self,
        evasion_rate: float,
        successful_evasions: list[dict] | None = None,
        force_mode: str | None = None,
    ) -> dict[str, Any]:
        """Evolve attacker based on current performance.

        Args:
            evasion_rate: 현재 evasion rate
            successful_evasions: 성공한 우회 공격 리스트
            force_mode: 강제 모드 지정

        Returns:
            Evolution summary
        """
        mode = force_mode or self.get_evolution_mode(evasion_rate)

        summary = {
            "mode": mode,
            "evasion_rate": evasion_rate,
            "changes": [],
            "new_strategies": 0,
            "new_slang": 0,
        }

        if mode == "none":
            logger.info(f"[Evolver] No evolution needed (evasion={evasion_rate:.1%})")
            return summary

        # 모드별 진화 실행
        if mode == "aggressive":
            result = self._aggressive_evolve(successful_evasions)
        elif mode == "normal":
            result = self._normal_evolve(successful_evasions)
        else:  # maintenance
            result = self._maintenance_evolve()

        summary.update(result)

        # 기록
        event = EvolutionEvent(
            timestamp=datetime.now(UTC).isoformat(),
            evasion_rate=evasion_rate,
            mode=mode,
            changes=summary["changes"],
            new_strategies=summary["new_strategies"],
            new_slang=summary["new_slang"],
        )
        self._evolution_history.append(event)
        self._total_evolutions += 1
        self._last_evolution_mode = mode

        self._save_state()

        logger.info(
            f"[Evolver] Evolved ({mode}): "
            f"{summary['new_strategies']} strategies, "
            f"{summary['new_slang']} slang, "
            f"{len(summary['changes'])} changes"
        )

        return summary

    def _aggressive_evolve(
        self,
        successful_evasions: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Aggressive evolution (evasion < 3%).

        - 새 전략 5개 생성
        - 슬랭 사전 대폭 확장
        - 탐색 가중치 +1.0
        """
        changes = []
        new_strategies = 0
        new_slang = 0

        # 1. 새 전략 생성
        base_strategies = [s.name for s in KOREAN_ATTACK_STRATEGIES]
        mutation_types = ["phonetic", "separator", "combined"]

        for _ in range(self.config.aggressive_new_strategies):
            base = random.choice(base_strategies)
            mutation = random.choice(mutation_types)

            strategy = self._mutator.create_mutated_strategy(base, mutation)
            if strategy:
                new_strategies += 1
                changes.append(f"Created strategy: {strategy.name}")

        # 2. 딥 조합 전략 생성
        if len(base_strategies) >= 3:
            combo_strategies = random.sample(base_strategies, 3)
            deep_strategy = self._mutator.create_deep_combination(combo_strategies)
            if deep_strategy:
                new_strategies += 1
                changes.append(f"Created deep combo: {deep_strategy.name}")

        # 3. 슬랭 사전 확장
        if self.config.expand_slang_on_evolve:
            # 발음 변형 생성
            variants = self._slang_evolver.generate_phonetic_variants(
                max_variants=self.config.max_slang_per_evolve
            )
            added = self._slang_evolver.validate_and_add(variants)
            new_slang += added
            if added > 0:
                changes.append(f"Added {added} phonetic slang variants")

            # 성공한 우회에서 발견
            if successful_evasions:
                discovered = self._slang_evolver.discover_from_failures(successful_evasions)
                added = self._slang_evolver.validate_and_add(discovered)
                new_slang += added
                if added > 0:
                    changes.append(f"Discovered {added} slang from evasions")

        # 4. 탐색 가중치 증가
        if self._learning_attacker:
            old_weight = self._learning_attacker._selector.exploration_weight
            new_weight = min(5.0, old_weight + self.config.aggressive_exploration_boost)
            self._learning_attacker._selector.exploration_weight = new_weight
            changes.append(f"Exploration weight: {old_weight:.1f} → {new_weight:.1f}")

            # 변형 수 증가
            if self.config.increase_variants_on_evolve:
                old_variants = self._learning_attacker.config.num_variants
                new_variants = min(self.config.max_variants, old_variants + 10)
                self._learning_attacker.config.num_variants = new_variants
                changes.append(f"Variants: {old_variants} → {new_variants}")

        return {
            "changes": changes,
            "new_strategies": new_strategies,
            "new_slang": new_slang,
        }

    def _normal_evolve(
        self,
        successful_evasions: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Normal evolution (evasion 3-5%).

        - 새 전략 1-2개 생성
        - 탐색 가중치 +0.3
        """
        changes = []
        new_strategies = 0
        new_slang = 0

        # 1. 새 전략 생성
        base_strategies = [s.name for s in KOREAN_ATTACK_STRATEGIES]

        for _ in range(self.config.normal_new_strategies):
            base = random.choice(base_strategies)
            strategy = self._mutator.create_mutated_strategy(base, "phonetic")
            if strategy:
                new_strategies += 1
                changes.append(f"Created strategy: {strategy.name}")

        # 2. 성공한 우회에서 슬랭 발견
        if successful_evasions and self.config.expand_slang_on_evolve:
            discovered = self._slang_evolver.discover_from_failures(successful_evasions)
            added = self._slang_evolver.validate_and_add(discovered[:10])
            new_slang += added
            if added > 0:
                changes.append(f"Discovered {added} slang")

        # 3. 탐색 가중치 증가
        if self._learning_attacker:
            old_weight = self._learning_attacker._selector.exploration_weight
            new_weight = min(4.0, old_weight + self.config.normal_exploration_boost)
            self._learning_attacker._selector.exploration_weight = new_weight
            changes.append(f"Exploration weight: {old_weight:.1f} → {new_weight:.1f}")

        return {
            "changes": changes,
            "new_strategies": new_strategies,
            "new_slang": new_slang,
        }

    def _maintenance_evolve(self) -> dict[str, Any]:
        """Maintenance evolution (evasion 5-8%).

        - 통계 감쇠만 수행
        """
        changes = []

        # 오래된 통계 감쇠
        if self._learning_attacker:
            self._learning_attacker._selector.decay_old_stats(decay_factor=0.95)
            changes.append("Decayed old statistics")

        return {
            "changes": changes,
            "new_strategies": 0,
            "new_slang": 0,
        }

    def get_new_strategies(self) -> list[KoreanAttackStrategy]:
        """Get all newly created strategies.

        Returns:
            List of new strategies
        """
        return self._mutator.get_mutated_strategies()

    def get_strategy_transforms(self) -> dict[str, Callable[[str], str]]:
        """Get all strategy transforms including mutations.

        Returns:
            Strategy name to transform function map
        """
        return self._mutator.get_strategy_map()

    def get_statistics(self) -> dict[str, Any]:
        """Get evolver statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_evolutions": self._total_evolutions,
            "last_mode": self._last_evolution_mode,
            "new_strategies": len(self._mutator.get_mutated_strategies()),
            "evolved_slang": self._slang_evolver.get_evolved_count(),
            "history_count": len(self._evolution_history),
        }

    def __repr__(self) -> str:
        return (
            f"AttackerEvolver("
            f"evolutions={self._total_evolutions}, "
            f"strategies={len(self._mutator.get_mutated_strategies())}, "
            f"slang={self._slang_evolver.get_evolved_count()})"
        )


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Attacker Evolver Demo")
    print("=" * 60)

    evolver = AttackerEvolver()
    print(f"\nInitialized: {evolver}")

    # 진화 모드 테스트
    test_rates = [0.01, 0.04, 0.06, 0.10, 0.20]
    print("\n[Evolution Modes]")
    for rate in test_rates:
        mode = evolver.get_evolution_mode(rate)
        print(f"  evasion={rate:.0%} → mode={mode}")

    # Aggressive 진화 테스트
    print("\n[Aggressive Evolution Test]")
    result = evolver.evolve(
        evasion_rate=0.02,
        successful_evasions=[
            {"variant_text": "ㅅㅂ", "original_text": "시발"},
            {"variant_text": "ㅂㅅ", "original_text": "병신"},
        ],
    )
    print(f"  Mode: {result['mode']}")
    print(f"  New strategies: {result['new_strategies']}")
    print(f"  New slang: {result['new_slang']}")
    print(f"  Changes:")
    for change in result["changes"]:
        print(f"    - {change}")

    # 생성된 전략
    strategies = evolver.get_new_strategies()
    print(f"\n[Generated Strategies] {len(strategies)}")
    for s in strategies[:5]:
        print(f"  {s.name}: {s.description}")

    # 통계
    stats = evolver.get_statistics()
    print(f"\n[Statistics]")
    print(f"  Total evolutions: {stats['total_evolutions']}")
    print(f"  New strategies: {stats['new_strategies']}")
    print(f"  Evolved slang: {stats['evolved_slang']}")

    print(f"\n{evolver}")
