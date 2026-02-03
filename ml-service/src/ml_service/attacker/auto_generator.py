"""Auto Attack Generator for Learning Attacker.

성공한 공격 패턴에서 새로운 슬랭과 전략 조합을 자동 생성합니다.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from ml_service.attacker.korean_strategies import KOREAN_ATTACK_STRATEGIES
from ml_service.attacker.hangul_utils import (
    CHOSEONG,
    JUNGSEONG,
    JONGSEONG,
    decompose_syllable,
    compose_syllable,
    is_hangul_syllable,
)
from ml_service.attacker.slang_dictionary import (
    add_new_slang,
    get_all_slang,
    FAMILY_SLANG,
    DEMOGRAPHIC_SLANG,
    HARM_SLANG,
)

if TYPE_CHECKING:
    from ml_service.attacker.failure_analyzer import AttackFeatures


@dataclass
class DiscoveredSlang:
    """Newly discovered slang expression."""

    text: str
    source: str  # 발견 출처 (예: "evasion_pattern", "combination")
    original: str | None = None  # 원본 (있는 경우)
    confidence: float = 0.5  # 유효성 신뢰도 (0-1)
    discovered_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "source": self.source,
            "original": self.original,
            "confidence": self.confidence,
            "discovered_at": self.discovered_at,
        }


@dataclass
class CombinedStrategy:
    """A combined attack strategy."""

    name: str
    base_strategies: list[str]
    description: str
    transform: Callable[[str], str]
    success_rate: float = 0.0
    usage_count: int = 0


class AutoAttackGenerator:
    """자동 공격 생성기.

    성공한 공격 패턴을 분석하여:
    1. 새로운 슬랭 표현 발견
    2. 효과적인 전략 조합 생성
    3. 슬랭 사전 자동 업데이트
    """

    def __init__(
        self,
        slang_dict_path: Path | None = None,
        min_confidence: float = 0.3,
    ) -> None:
        """Initialize generator.

        Args:
            slang_dict_path: 슬랭 사전 저장 경로
            min_confidence: 최소 신뢰도 (새 슬랭 추가 기준)
        """
        self.slang_dict_path = slang_dict_path or Path("data/korean/discovered_slang.json")
        self.min_confidence = min_confidence

        # 발견된 슬랭
        self._discovered_slang: list[DiscoveredSlang] = []
        self._validated_slang: set[str] = set()

        # 생성된 전략 조합
        self._combined_strategies: list[CombinedStrategy] = []

        # 전략 맵 (이름 → 변환 함수)
        self._strategy_map: dict[str, Callable[[str], str]] = {
            s.name: s.transform for s in KOREAN_ATTACK_STRATEGIES
        }

        # 로드
        self._load_discovered_slang()

    def _load_discovered_slang(self) -> None:
        """Load previously discovered slang."""
        if self.slang_dict_path.exists():
            try:
                with open(self.slang_dict_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for item in data.get("discovered", []):
                        self._discovered_slang.append(
                            DiscoveredSlang(
                                text=item["text"],
                                source=item.get("source", "unknown"),
                                original=item.get("original"),
                                confidence=item.get("confidence", 0.5),
                                discovered_at=item.get("discovered_at", ""),
                            )
                        )
                    self._validated_slang = set(data.get("validated", []))
            except Exception:
                pass

    def _save_discovered_slang(self) -> None:
        """Save discovered slang to file."""
        self.slang_dict_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "discovered": [s.to_dict() for s in self._discovered_slang],
            "validated": list(self._validated_slang),
            "updated_at": datetime.now(UTC).isoformat(),
        }

        with open(self.slang_dict_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def discover_new_slang(
        self,
        successful_attacks: list[dict],
        min_length: int = 2,
        max_length: int = 10,
    ) -> list[DiscoveredSlang]:
        """Discover new slang from successful attack variants.

        Args:
            successful_attacks: List of successful attack results with
                - variant_text: 변형된 텍스트
                - original_text: 원본 텍스트
                - strategy_name: 사용된 전략
            min_length: 최소 길이
            max_length: 최대 길이

        Returns:
            List of newly discovered slang
        """
        existing_slang = set(get_all_slang())
        existing_slang.update(self._validated_slang)
        existing_slang.update(s.text for s in self._discovered_slang)

        new_slang = []

        for attack in successful_attacks:
            variant = attack.get("variant_text", "")
            original = attack.get("original_text", "")
            strategy = attack.get("strategy_name", "unknown")

            # 길이 필터
            if not (min_length <= len(variant) <= max_length):
                continue

            # 이미 알려진 슬랭인지 확인
            if variant in existing_slang:
                continue

            # 원본과 너무 비슷하면 제외
            if variant == original:
                continue

            # 의미 있는 변형인지 확인
            confidence = self._calculate_slang_confidence(variant, original, strategy)

            if confidence >= self.min_confidence:
                slang = DiscoveredSlang(
                    text=variant,
                    source=f"evasion_{strategy}",
                    original=original,
                    confidence=confidence,
                )
                new_slang.append(slang)
                self._discovered_slang.append(slang)
                existing_slang.add(variant)

        if new_slang:
            self._save_discovered_slang()

        return new_slang

    def _calculate_slang_confidence(
        self,
        variant: str,
        original: str,
        strategy: str,
    ) -> float:
        """Calculate confidence that a variant is a valid new slang.

        Args:
            variant: 변형된 텍스트
            original: 원본 텍스트
            strategy: 사용된 전략

        Returns:
            Confidence score (0-1)
        """
        confidence = 0.3  # 기본 점수

        # 1. 길이 적절성 (2-6자가 이상적)
        if 2 <= len(variant) <= 6:
            confidence += 0.2
        elif len(variant) <= 10:
            confidence += 0.1

        # 2. 한글 포함 비율
        hangul_ratio = sum(1 for c in variant if is_hangul_syllable(c)) / max(len(variant), 1)
        confidence += hangul_ratio * 0.2

        # 3. 초성만 있는 경우 높은 점수 (은어 가능성)
        chosung_count = sum(1 for c in variant if c in CHOSEONG)
        if chosung_count >= 2 and len(variant) <= 4:
            confidence += 0.2

        # 4. 전략 신뢰도
        high_confidence_strategies = ["chosung", "slang", "number_sub", "compat_jamo"]
        if strategy in high_confidence_strategies:
            confidence += 0.1

        # 5. 특수문자가 너무 많으면 감점
        special_chars = sum(1 for c in variant if not c.isalnum() and not c.isspace())
        if special_chars > len(variant) * 0.3:
            confidence -= 0.2

        return min(max(confidence, 0.0), 1.0)

    def create_combined_strategy(
        self,
        strategy_names: list[str],
        name: str | None = None,
    ) -> CombinedStrategy | None:
        """Create a new combined strategy from multiple base strategies.

        Args:
            strategy_names: 조합할 전략 이름들
            name: 새 전략 이름 (없으면 자동 생성)

        Returns:
            CombinedStrategy or None if failed
        """
        # 전략 존재 확인
        transforms = []
        for sname in strategy_names:
            if sname not in self._strategy_map:
                return None
            transforms.append(self._strategy_map[sname])

        # 이름 생성
        if name is None:
            name = "_".join(strategy_names[:3])
            if len(strategy_names) > 3:
                name += f"_plus{len(strategy_names) - 3}"

        # 조합 함수 생성
        def combined_transform(text: str) -> str:
            result = text
            for transform in transforms:
                result = transform(result)
            return result

        strategy = CombinedStrategy(
            name=name,
            base_strategies=strategy_names,
            description=f"Combined: {' + '.join(strategy_names)}",
            transform=combined_transform,
        )

        self._combined_strategies.append(strategy)
        return strategy

    def get_combined_strategies(self) -> list[CombinedStrategy]:
        """Get all created combined strategies."""
        return self._combined_strategies.copy()

    def generate_variants_with_discovered_slang(
        self,
        text: str,
        num_variants: int = 5,
    ) -> list[tuple[str, str]]:
        """Generate variants using discovered slang patterns.

        Args:
            text: 원본 텍스트
            num_variants: 생성할 변형 수

        Returns:
            List of (variant, source) tuples
        """
        variants = []

        # 발견된 슬랭 패턴 적용
        for slang in self._discovered_slang:
            if slang.original and slang.original in text:
                variant = text.replace(slang.original, slang.text, 1)
                if variant != text:
                    variants.append((variant, f"discovered_slang_{slang.source}"))

        # 검증된 슬랭으로 변환 시도
        for validated in self._validated_slang:
            # 비슷한 원본 찾기 (간단한 휴리스틱)
            for slang in self._discovered_slang:
                if slang.text == validated and slang.original:
                    if slang.original in text:
                        variant = text.replace(slang.original, validated, 1)
                        if variant != text:
                            variants.append((variant, "validated_slang"))

        # 랜덤 샘플링
        if len(variants) > num_variants:
            variants = random.sample(variants, num_variants)

        return variants

    def update_slang_dictionary(
        self,
        min_confidence: float | None = None,
    ) -> int:
        """Update the main slang dictionary with validated discoveries.

        Args:
            min_confidence: 최소 신뢰도 (없으면 self.min_confidence 사용)

        Returns:
            Number of new slang added
        """
        threshold = min_confidence or self.min_confidence
        added = 0

        for slang in self._discovered_slang:
            if slang.confidence >= threshold and slang.text not in self._validated_slang:
                # 카테고리 추론
                category = self._infer_category(slang.text, slang.original)

                # 런타임 사전에 추가
                if add_new_slang(slang.text, category):
                    self._validated_slang.add(slang.text)
                    added += 1

        if added > 0:
            self._save_discovered_slang()

        return added

    def _infer_category(self, text: str, original: str | None) -> str:
        """Infer the category of discovered slang.

        Args:
            text: 슬랭 텍스트
            original: 원본 텍스트

        Returns:
            Category name
        """
        # 기존 카테고리에서 힌트 찾기
        if original:
            # 가족 관련
            for family_word in ["엄마", "애비", "부모", "니미", "느금"]:
                if family_word in original.lower():
                    return "family"

            # 자해/위협 관련
            for harm_word in ["죽", "뒤져", "꺼져", "재기"]:
                if harm_word in original.lower():
                    return "harm"

            # 세대/성별 관련
            for demo_word in ["틀딱", "한남", "한녀", "맘충", "페미"]:
                if demo_word in original.lower():
                    return "demographic"

        # 기본값
        return "behavior"

    def auto_generate_strategies(
        self,
        weak_spots: list[dict],
        top_n: int = 5,
    ) -> list[CombinedStrategy]:
        """Auto-generate new strategies based on weak spots analysis.

        Args:
            weak_spots: 식별된 약점 목록 (FailurePatternAnalyzer에서)
            top_n: 생성할 전략 수

        Returns:
            List of newly created combined strategies
        """
        new_strategies = []

        # 약점에서 효과적인 전략 추출
        effective_strategies = []
        for ws in weak_spots:
            pattern_type = ws.get("pattern_type", "")
            if pattern_type.startswith("strategy_"):
                strategy_name = pattern_type.replace("strategy_", "")
                effective_strategies.append(
                    (strategy_name, ws.get("success_rate", 0))
                )

        # 성공률 순 정렬
        effective_strategies.sort(key=lambda x: x[1], reverse=True)

        # 상위 전략들 조합
        top_strategies = [s[0] for s in effective_strategies[:6]]

        for i, s1 in enumerate(top_strategies):
            for s2 in top_strategies[i + 1:]:
                if len(new_strategies) >= top_n:
                    break

                combo = self.create_combined_strategy([s1, s2])
                if combo:
                    new_strategies.append(combo)

            if len(new_strategies) >= top_n:
                break

        # 3개 전략 조합도 시도
        if len(top_strategies) >= 3 and len(new_strategies) < top_n:
            for i in range(min(3, len(top_strategies) - 2)):
                if len(new_strategies) >= top_n:
                    break
                combo = self.create_combined_strategy(top_strategies[i:i + 3])
                if combo:
                    new_strategies.append(combo)

        return new_strategies

    def get_discovered_slang_stats(self) -> dict[str, Any]:
        """Get statistics about discovered slang.

        Returns:
            Statistics dictionary
        """
        by_source = Counter(s.source for s in self._discovered_slang)
        by_confidence = {
            "high": sum(1 for s in self._discovered_slang if s.confidence >= 0.7),
            "medium": sum(1 for s in self._discovered_slang if 0.4 <= s.confidence < 0.7),
            "low": sum(1 for s in self._discovered_slang if s.confidence < 0.4),
        }

        return {
            "total_discovered": len(self._discovered_slang),
            "total_validated": len(self._validated_slang),
            "by_source": dict(by_source),
            "by_confidence": by_confidence,
            "combined_strategies": len(self._combined_strategies),
        }

    def reset(self) -> None:
        """Reset all discovered data."""
        self._discovered_slang.clear()
        self._validated_slang.clear()
        self._combined_strategies.clear()


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Auto Attack Generator Demo")
    print("=" * 60)

    generator = AutoAttackGenerator()

    # 성공한 공격 시뮬레이션
    successful_attacks = [
        {"variant_text": "ㅅㅂ", "original_text": "시발", "strategy_name": "chosung"},
        {"variant_text": "시8", "original_text": "시발", "strategy_name": "number_sub"},
        {"variant_text": "ㅂㅅ", "original_text": "병신", "strategy_name": "chosung"},
        {"variant_text": "느금마", "original_text": "니엄마", "strategy_name": "slang"},
        {"variant_text": "ㅁㅊ", "original_text": "미친", "strategy_name": "chosung"},
        {"variant_text": "ㅈㄹ", "original_text": "지랄", "strategy_name": "chosung"},
    ]

    # 새 슬랭 발견
    new_slang = generator.discover_new_slang(successful_attacks)
    print(f"\n[Discovered Slang] {len(new_slang)} items")
    for s in new_slang:
        print(f"  '{s.text}' <- '{s.original}' (confidence: {s.confidence:.2f})")

    # 전략 조합 생성
    combo1 = generator.create_combined_strategy(["chosung", "space_insertion"])
    combo2 = generator.create_combined_strategy(["number_sub", "zero_width"])
    print(f"\n[Combined Strategies]")
    for combo in generator.get_combined_strategies():
        print(f"  {combo.name}: {combo.description}")

    # 발견된 슬랭으로 변형 생성
    test_text = "시발놈아"
    variants = generator.generate_variants_with_discovered_slang(test_text, num_variants=5)
    print(f"\n[Variants for '{test_text}']")
    for variant, source in variants:
        print(f"  '{variant}' (source: {source})")

    # 통계
    stats = generator.get_discovered_slang_stats()
    print(f"\n[Statistics]")
    print(f"  Discovered: {stats['total_discovered']}")
    print(f"  Validated: {stats['total_validated']}")
    print(f"  Combined strategies: {stats['combined_strategies']}")
