"""Attacker Evolver for Automatic Attack Evolution.

공격자가 약할 때 (evasion < 5%) 자동으로 공격 전략을 진화시킵니다.

Components:
- StrategyMutator: 기존 전략 변형/조합으로 새 전략 생성
- SlangEvolver: 슬랭 사전 동적 확장
- AttackerEvolver: 메인 클래스 (통합)
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ml_service.attacker.korean_strategies import (
    CHOSEONG,
    JUNGSEONG,
    JONGSEONG,
    KOREAN_ATTACK_STRATEGIES,
    KoreanAttackStrategy,
    decompose_syllable,
    compose_syllable,
    is_hangul_syllable,
)
from ml_service.attacker.slang_dictionary import (
    add_new_slang,
    get_all_slang,
    get_slang_by_category,
    generate_variants,
    SUBSTITUTION_MAP,
    INSERTION_PATTERNS,
)

if TYPE_CHECKING:
    from ml_service.attacker.learning_attacker import LearningAttacker

logger = logging.getLogger(__name__)


@dataclass
class AttackerEvolverConfig:
    """Configuration for Attacker Evolver."""

    # 진화 모드 임계값
    aggressive_threshold: float = 0.03  # evasion < 3%: aggressive
    normal_threshold: float = 0.05  # evasion < 5%: normal
    maintenance_threshold: float = 0.08  # evasion < 8%: maintenance

    # 진화 강도
    aggressive_new_strategies: int = 5
    normal_new_strategies: int = 2
    maintenance_new_strategies: int = 0

    # 탐색 가중치 조정
    aggressive_exploration_boost: float = 1.0
    normal_exploration_boost: float = 0.3
    maintenance_exploration_boost: float = 0.0

    # 슬랭 확장
    expand_slang_on_evolve: bool = True
    max_slang_per_evolve: int = 20

    # 변형 수 조정
    increase_variants_on_evolve: bool = True
    max_variants: int = 50

    # 저장 경로
    state_path: Path = field(default_factory=lambda: Path("data/korean/evolver_state.json"))


@dataclass
class EvolutionEvent:
    """Record of an evolution event."""

    timestamp: str
    evasion_rate: float
    mode: str
    changes: list[str]
    new_strategies: int = 0
    new_slang: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "evasion_rate": self.evasion_rate,
            "mode": self.mode,
            "changes": self.changes,
            "new_strategies": self.new_strategies,
            "new_slang": self.new_slang,
        }


class StrategyMutator:
    """Mutate and combine existing attack strategies.

    기존 전략을 변형하고 조합하여 새로운 전략을 생성합니다.
    """

    # 발음 변형 맵
    PHONETIC_MAP = {
        "시": ["씨", "쉬", "si", "shi"],
        "발": ["빨", "팔", "bal", "8"],
        "병": ["뼝", "벵", "byung"],
        "신": ["씬", "신", "sin"],
        "놈": ["넘", "늠", "nom"],
        "새": ["쌔", "쎄", "sae"],
        "끼": ["키", "kki"],
        "죽": ["쥭", "juk"],
        "년": ["뇬", "녀", "nyeon"],
        "지": ["쥐", "ji"],
        "랄": ["랑", "ral"],
    }

    # 구분자 패턴
    SEPARATORS = ["", " ", ".", "-", "_", "~", "`", "'", "·", "ㅋ", "ㅎ"]

    def __init__(self) -> None:
        self._strategy_map: dict[str, Callable[[str], str]] = {
            s.name: s.transform for s in KOREAN_ATTACK_STRATEGIES
        }
        self._mutated_strategies: list[KoreanAttackStrategy] = []

    def mutate_phonetic(self, text: str) -> str:
        """Apply phonetic mutation to text.

        Args:
            text: 입력 텍스트

        Returns:
            발음 변형된 텍스트
        """
        result = text
        for original, variants in self.PHONETIC_MAP.items():
            if original in result:
                variant = random.choice(variants)
                result = result.replace(original, variant, 1)
        return result

    def apply_separator_variation(self, text: str) -> str:
        """Insert separators between characters.

        Args:
            text: 입력 텍스트

        Returns:
            구분자 삽입된 텍스트
        """
        if len(text) < 2:
            return text

        sep = random.choice(self.SEPARATORS)
        if not sep:
            return text

        # 랜덤 위치에 삽입
        pos = random.randint(1, len(text) - 1)
        return text[:pos] + sep + text[pos:]

    def create_mutated_strategy(
        self,
        base_strategy_name: str,
        mutation_type: str = "phonetic",
    ) -> KoreanAttackStrategy | None:
        """Create a mutated version of a base strategy.

        Args:
            base_strategy_name: 기본 전략 이름
            mutation_type: 변형 타입 ("phonetic", "separator", "combined")

        Returns:
            새로운 전략 또는 None
        """
        if base_strategy_name not in self._strategy_map:
            return None

        base_transform = self._strategy_map[base_strategy_name]

        if mutation_type == "phonetic":
            def mutated_transform(text: str) -> str:
                return self.mutate_phonetic(base_transform(text))
            name = f"{base_strategy_name}_phonetic"
            desc = f"Phonetic mutation of {base_strategy_name}"

        elif mutation_type == "separator":
            def mutated_transform(text: str) -> str:
                return self.apply_separator_variation(base_transform(text))
            name = f"{base_strategy_name}_sep"
            desc = f"Separator variation of {base_strategy_name}"

        elif mutation_type == "combined":
            def mutated_transform(text: str) -> str:
                result = base_transform(text)
                result = self.mutate_phonetic(result)
                result = self.apply_separator_variation(result)
                return result
            name = f"{base_strategy_name}_combo"
            desc = f"Combined mutation of {base_strategy_name}"

        else:
            return None

        strategy = KoreanAttackStrategy(
            name=name,
            description=desc,
            transform=mutated_transform,
            example_input="시발",
            example_output=mutated_transform("시발"),
        )

        self._mutated_strategies.append(strategy)
        self._strategy_map[name] = mutated_transform

        return strategy

    def create_deep_combination(
        self,
        strategy_names: list[str],
        name: str | None = None,
    ) -> KoreanAttackStrategy | None:
        """Create a deep combination of 2-3 strategies.

        Args:
            strategy_names: 조합할 전략 이름들 (2-3개)
            name: 새 전략 이름

        Returns:
            새로운 전략 또는 None
        """
        if len(strategy_names) < 2 or len(strategy_names) > 3:
            return None

        transforms = []
        for sname in strategy_names:
            if sname not in self._strategy_map:
                return None
            transforms.append(self._strategy_map[sname])

        if name is None:
            name = "deep_" + "_".join(s[:4] for s in strategy_names)

        def combined_transform(text: str) -> str:
            result = text
            for t in transforms:
                result = t(result)
            # 추가 변형
            result = self.mutate_phonetic(result)
            return result

        strategy = KoreanAttackStrategy(
            name=name,
            description=f"Deep combination: {' + '.join(strategy_names)}",
            transform=combined_transform,
            example_input="시발",
            example_output=combined_transform("시발"),
        )

        self._mutated_strategies.append(strategy)
        self._strategy_map[name] = combined_transform

        return strategy

    def get_mutated_strategies(self) -> list[KoreanAttackStrategy]:
        """Get all mutated strategies."""
        return self._mutated_strategies.copy()

    def get_strategy_map(self) -> dict[str, Callable[[str], str]]:
        """Get full strategy map including mutations."""
        return self._strategy_map.copy()


class SlangEvolver:
    """Dynamically expand slang dictionary.

    슬랭 사전을 동적으로 확장합니다.
    """

    def __init__(
        self,
        classifier: Any = None,
        storage_path: Path | None = None,
    ) -> None:
        """Initialize SlangEvolver.

        Args:
            classifier: 분류기 (테스트용)
            storage_path: 저장 경로
        """
        self.classifier = classifier
        self.storage_path = storage_path or Path("data/korean/evolved_slang.json")

        self._evolved_slang: list[dict] = []
        self._pending_slang: list[str] = []

        self._load_state()

    def _load_state(self) -> None:
        """Load evolved slang state."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._evolved_slang = data.get("evolved", [])
                self._pending_slang = data.get("pending", [])
            except Exception:
                pass

    def _save_state(self) -> None:
        """Save evolved slang state."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "evolved": self._evolved_slang,
            "pending": self._pending_slang,
            "updated_at": datetime.now(UTC).isoformat(),
        }

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def generate_phonetic_variants(
        self,
        base_words: list[str] | None = None,
        max_variants: int = 20,
    ) -> list[str]:
        """Generate phonetic variants of base words.

        Args:
            base_words: 기본 단어 리스트 (없으면 슬랭 사전에서)
            max_variants: 최대 변형 수

        Returns:
            생성된 변형 리스트
        """
        if base_words is None:
            base_words = get_all_slang()[:20]

        variants = []

        for word in base_words:
            # 발음 변형
            for char, subs in SUBSTITUTION_MAP.items():
                if char in word:
                    for sub in subs:
                        variant = word.replace(char, sub)
                        if variant not in variants and variant != word:
                            variants.append(variant)

            # 구분자 삽입
            for pattern in INSERTION_PATTERNS[:5]:
                try:
                    variant = pattern(word)
                    if variant not in variants:
                        variants.append(variant)
                except Exception:
                    pass

        # 랜덤 샘플링
        if len(variants) > max_variants:
            variants = random.sample(variants, max_variants)

        return variants

    def discover_from_failures(
        self,
        successful_evasions: list[dict],
    ) -> list[str]:
        """Discover new slang from successful evasions.

        Args:
            successful_evasions: 성공한 우회 공격 리스트

        Returns:
            발견된 새 슬랭 리스트
        """
        existing = set(get_all_slang())
        existing.update(s["text"] for s in self._evolved_slang)

        discovered = []

        for evasion in successful_evasions:
            variant = evasion.get("variant_text", "")

            # 조건: 짧고, 새로운 것
            if 2 <= len(variant) <= 8 and variant not in existing:
                discovered.append(variant)
                existing.add(variant)

        return discovered

    def validate_and_add(
        self,
        candidates: list[str],
        category: str = "behavior",
    ) -> int:
        """Validate candidates and add to slang dictionary.

        Args:
            candidates: 후보 슬랭 리스트
            category: 카테고리

        Returns:
            추가된 슬랭 수
        """
        added = 0

        for candidate in candidates:
            # 분류기로 테스트 (있으면)
            if self.classifier:
                try:
                    pred = self.classifier.predict([candidate])[0]
                    # 정상으로 예측되면 유효한 우회 슬랭
                    if pred["label"] == 0:
                        if add_new_slang(candidate, category):
                            self._evolved_slang.append({
                                "text": candidate,
                                "category": category,
                                "added_at": datetime.now(UTC).isoformat(),
                            })
                            added += 1
                except Exception:
                    pass
            else:
                # 분류기 없으면 바로 추가
                if add_new_slang(candidate, category):
                    self._evolved_slang.append({
                        "text": candidate,
                        "category": category,
                        "added_at": datetime.now(UTC).isoformat(),
                    })
                    added += 1

        if added > 0:
            self._save_state()

        return added

    def get_evolved_count(self) -> int:
        """Get number of evolved slang."""
        return len(self._evolved_slang)


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
