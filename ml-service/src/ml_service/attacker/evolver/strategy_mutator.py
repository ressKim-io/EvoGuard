"""Strategy Mutator for Attacker Evolver.

기존 전략을 변형하고 조합하여 새로운 전략을 생성합니다.
"""

from __future__ import annotations

import random
from typing import Callable

from ml_service.attacker.korean_strategies import (
    KOREAN_ATTACK_STRATEGIES,
    KoreanAttackStrategy,
)


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


__all__ = ['StrategyMutator']
