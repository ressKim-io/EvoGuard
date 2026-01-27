"""Failure Pattern Analyzer for Learning Attacker.

성공한 공격(탐지 우회)의 공통 특성을 분석하고
방어 모델의 약점을 식별합니다.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from ml_service.attacker.korean_strategies import (
    CHOSEONG,
    JUNGSEONG,
    JONGSEONG,
    decompose_syllable,
    is_hangul_syllable,
)


@dataclass
class AttackFeatures:
    """Features extracted from an attack attempt."""

    original_text: str
    variant_text: str
    strategy_name: str
    is_evasion: bool  # True if evaded detection

    # Text features
    length_change_ratio: float = 0.0
    jamo_decompose_ratio: float = 0.0
    space_count_change: int = 0
    special_char_count: int = 0
    emoji_count: int = 0
    digit_count: int = 0
    english_char_count: int = 0
    zero_width_count: int = 0

    # Pattern features
    has_chosung_only: bool = False
    has_number_substitution: bool = False
    has_repeated_chars: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "original_text": self.original_text,
            "variant_text": self.variant_text,
            "strategy_name": self.strategy_name,
            "is_evasion": self.is_evasion,
            "length_change_ratio": self.length_change_ratio,
            "jamo_decompose_ratio": self.jamo_decompose_ratio,
            "space_count_change": self.space_count_change,
            "special_char_count": self.special_char_count,
            "emoji_count": self.emoji_count,
            "digit_count": self.digit_count,
            "english_char_count": self.english_char_count,
            "zero_width_count": self.zero_width_count,
            "has_chosung_only": self.has_chosung_only,
            "has_number_substitution": self.has_number_substitution,
            "has_repeated_chars": self.has_repeated_chars,
        }


@dataclass
class WeakSpot:
    """Identified weak spot in the defense model."""

    pattern_type: str  # 패턴 유형 (예: "chosung", "space_insertion")
    description: str
    success_rate: float  # 해당 패턴의 탐지 우회 성공률
    sample_count: int  # 샘플 수
    examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "success_rate": self.success_rate,
            "sample_count": self.sample_count,
            "examples": self.examples[:5],  # 최대 5개만
        }


class FailurePatternAnalyzer:
    """실패 패턴 분석기.

    성공한 공격(탐지 우회)의 공통 특성을 분석하여
    방어 모델의 약점을 식별합니다.
    """

    # 제로 너비 문자 목록
    ZERO_WIDTH_CHARS = {"\u200b", "\u200c", "\u200d", "\ufeff"}

    # 이모지 패턴
    EMOJI_PATTERN = re.compile(
        r"[\U0001F600-\U0001F64F"  # emoticons
        r"\U0001F300-\U0001F5FF"  # symbols & pictographs
        r"\U0001F680-\U0001F6FF"  # transport & map
        r"\U0001F700-\U0001F77F"  # alchemical
        r"\U0001F780-\U0001F7FF"  # Geometric Shapes
        r"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        r"\U0001F900-\U0001F9FF"  # Supplemental Symbols
        r"\U0001FA00-\U0001FA6F"  # Chess Symbols
        r"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        r"\U00002702-\U000027B0"  # Dingbats
        r"\U0001f926-\U0001f937"
        r"\U00010000-\U0010ffff"
        r"\u200d\u2640-\u2642\u2600-\u2B55\u23cf\u23e9\u231a\ufe0f\u3030]+"
    )

    def __init__(self) -> None:
        """Initialize analyzer."""
        self._features_history: list[AttackFeatures] = []
        self._weak_spots: list[WeakSpot] = []

    def extract_features(
        self,
        original: str,
        variant: str,
        strategy_name: str = "unknown",
        is_evasion: bool = False,
    ) -> AttackFeatures:
        """Extract features from an attack attempt.

        Args:
            original: 원본 텍스트
            variant: 변형된 텍스트
            strategy_name: 사용된 전략 이름
            is_evasion: 탐지 우회 성공 여부

        Returns:
            AttackFeatures object
        """
        features = AttackFeatures(
            original_text=original,
            variant_text=variant,
            strategy_name=strategy_name,
            is_evasion=is_evasion,
        )

        # Length change ratio
        if len(original) > 0:
            features.length_change_ratio = len(variant) / len(original)

        # Space count change
        orig_spaces = original.count(" ")
        var_spaces = variant.count(" ")
        features.space_count_change = var_spaces - orig_spaces

        # Count special characters
        features.special_char_count = sum(
            1 for c in variant
            if not c.isalnum() and not c.isspace() and c not in self.ZERO_WIDTH_CHARS
        )

        # Emoji count
        emoji_matches = self.EMOJI_PATTERN.findall(variant)
        features.emoji_count = len(emoji_matches)

        # Digit count
        features.digit_count = sum(1 for c in variant if c.isdigit())

        # English character count
        features.english_char_count = sum(1 for c in variant if c.isascii() and c.isalpha())

        # Zero-width character count
        features.zero_width_count = sum(1 for c in variant if c in self.ZERO_WIDTH_CHARS)

        # Jamo decomposition ratio (자모 분리 비율)
        jamo_count = sum(
            1 for c in variant
            if c in CHOSEONG or c in JUNGSEONG or c in JONGSEONG
        )
        if len(variant) > 0:
            features.jamo_decompose_ratio = jamo_count / len(variant)

        # Check for chosung-only pattern (초성만 있는 경우)
        chosung_only = [c for c in variant if c in CHOSEONG]
        features.has_chosung_only = len(chosung_only) >= 2

        # Check for number substitution (숫자 치환)
        features.has_number_substitution = (
            features.digit_count > 0 and
            any(c.isdigit() for c in variant) and
            not any(c.isdigit() for c in original)
        )

        # Check for repeated characters
        for i in range(len(variant) - 1):
            if variant[i] == variant[i + 1] and variant[i] not in " \n":
                features.has_repeated_chars = True
                break

        return features

    def analyze_batch(
        self,
        results: list[dict],
    ) -> dict[str, Any]:
        """Analyze a batch of attack results.

        Args:
            results: List of attack results with keys:
                - original_text
                - variant_text
                - strategy_name
                - is_evasion (or success)

        Returns:
            Analysis summary
        """
        if not results:
            return {"error": "No results to analyze"}

        # Extract features
        features_list = []
        for r in results:
            is_evasion = r.get("is_evasion", r.get("success", False))
            features = self.extract_features(
                original=r.get("original_text", ""),
                variant=r.get("variant_text", ""),
                strategy_name=r.get("strategy_name", "unknown"),
                is_evasion=is_evasion,
            )
            features_list.append(features)
            self._features_history.append(features)

        # Separate successful evasions and failures
        evasions = [f for f in features_list if f.is_evasion]
        detections = [f for f in features_list if not f.is_evasion]

        # Strategy effectiveness
        strategy_stats = defaultdict(lambda: {"evasions": 0, "total": 0})
        for f in features_list:
            strategy_stats[f.strategy_name]["total"] += 1
            if f.is_evasion:
                strategy_stats[f.strategy_name]["evasions"] += 1

        strategy_effectiveness = {
            name: stats["evasions"] / stats["total"] if stats["total"] > 0 else 0
            for name, stats in strategy_stats.items()
        }

        # Common patterns in successful evasions
        evasion_patterns = {
            "avg_length_change": (
                sum(f.length_change_ratio for f in evasions) / len(evasions)
                if evasions else 0
            ),
            "avg_space_change": (
                sum(f.space_count_change for f in evasions) / len(evasions)
                if evasions else 0
            ),
            "chosung_only_rate": (
                sum(1 for f in evasions if f.has_chosung_only) / len(evasions)
                if evasions else 0
            ),
            "number_sub_rate": (
                sum(1 for f in evasions if f.has_number_substitution) / len(evasions)
                if evasions else 0
            ),
            "zero_width_rate": (
                sum(1 for f in evasions if f.zero_width_count > 0) / len(evasions)
                if evasions else 0
            ),
            "emoji_rate": (
                sum(1 for f in evasions if f.emoji_count > 0) / len(evasions)
                if evasions else 0
            ),
        }

        return {
            "total_samples": len(features_list),
            "evasion_count": len(evasions),
            "detection_count": len(detections),
            "evasion_rate": len(evasions) / len(features_list) if features_list else 0,
            "strategy_effectiveness": strategy_effectiveness,
            "evasion_patterns": evasion_patterns,
        }

    def get_weak_spots(
        self,
        min_samples: int = 5,
        min_success_rate: float = 0.3,
    ) -> list[WeakSpot]:
        """Identify weak spots in the defense model.

        Args:
            min_samples: 최소 샘플 수
            min_success_rate: 최소 성공률

        Returns:
            List of identified weak spots
        """
        if len(self._features_history) < min_samples:
            return []

        weak_spots = []

        # 1. 전략별 약점
        strategy_stats = defaultdict(lambda: {"evasions": 0, "total": 0, "examples": []})
        for f in self._features_history:
            strategy_stats[f.strategy_name]["total"] += 1
            if f.is_evasion:
                strategy_stats[f.strategy_name]["evasions"] += 1
                strategy_stats[f.strategy_name]["examples"].append(f.variant_text)

        for name, stats in strategy_stats.items():
            if stats["total"] >= min_samples:
                rate = stats["evasions"] / stats["total"]
                if rate >= min_success_rate:
                    weak_spots.append(WeakSpot(
                        pattern_type=f"strategy_{name}",
                        description=f"Strategy '{name}' has high evasion rate",
                        success_rate=rate,
                        sample_count=stats["total"],
                        examples=stats["examples"][:5],
                    ))

        # 2. 특정 패턴 약점 분석

        # 초성만 사용하는 패턴
        chosung_features = [f for f in self._features_history if f.has_chosung_only]
        if len(chosung_features) >= min_samples:
            rate = sum(1 for f in chosung_features if f.is_evasion) / len(chosung_features)
            if rate >= min_success_rate:
                weak_spots.append(WeakSpot(
                    pattern_type="chosung_only",
                    description="Chosung-only patterns evade detection",
                    success_rate=rate,
                    sample_count=len(chosung_features),
                    examples=[f.variant_text for f in chosung_features if f.is_evasion][:5],
                ))

        # 숫자 치환 패턴
        number_features = [f for f in self._features_history if f.has_number_substitution]
        if len(number_features) >= min_samples:
            rate = sum(1 for f in number_features if f.is_evasion) / len(number_features)
            if rate >= min_success_rate:
                weak_spots.append(WeakSpot(
                    pattern_type="number_substitution",
                    description="Number substitution patterns evade detection",
                    success_rate=rate,
                    sample_count=len(number_features),
                    examples=[f.variant_text for f in number_features if f.is_evasion][:5],
                ))

        # 제로 너비 문자 사용
        zwc_features = [f for f in self._features_history if f.zero_width_count > 0]
        if len(zwc_features) >= min_samples:
            rate = sum(1 for f in zwc_features if f.is_evasion) / len(zwc_features)
            if rate >= min_success_rate:
                weak_spots.append(WeakSpot(
                    pattern_type="zero_width_chars",
                    description="Zero-width characters evade detection",
                    success_rate=rate,
                    sample_count=len(zwc_features),
                    examples=[f.variant_text for f in zwc_features if f.is_evasion][:5],
                ))

        # 공백 삽입 패턴
        space_features = [f for f in self._features_history if f.space_count_change > 0]
        if len(space_features) >= min_samples:
            rate = sum(1 for f in space_features if f.is_evasion) / len(space_features)
            if rate >= min_success_rate:
                weak_spots.append(WeakSpot(
                    pattern_type="space_insertion",
                    description="Space insertion patterns evade detection",
                    success_rate=rate,
                    sample_count=len(space_features),
                    examples=[f.variant_text for f in space_features if f.is_evasion][:5],
                ))

        # 길이 변화가 큰 패턴
        length_features = [f for f in self._features_history if f.length_change_ratio > 1.5]
        if len(length_features) >= min_samples:
            rate = sum(1 for f in length_features if f.is_evasion) / len(length_features)
            if rate >= min_success_rate:
                weak_spots.append(WeakSpot(
                    pattern_type="length_expansion",
                    description="Length expansion (>1.5x) patterns evade detection",
                    success_rate=rate,
                    sample_count=len(length_features),
                    examples=[f.variant_text for f in length_features if f.is_evasion][:5],
                ))

        # 성공률 순으로 정렬
        weak_spots.sort(key=lambda x: x.success_rate, reverse=True)

        self._weak_spots = weak_spots
        return weak_spots

    def suggest_strategy_combinations(
        self,
        top_n: int = 3,
    ) -> list[list[str]]:
        """Suggest effective strategy combinations based on analysis.

        Args:
            top_n: 제안할 조합 수

        Returns:
            List of strategy name combinations
        """
        if not self._weak_spots:
            self.get_weak_spots()

        # 성공률 높은 패턴에서 전략 추출
        effective_strategies = []
        for ws in self._weak_spots:
            if ws.pattern_type.startswith("strategy_"):
                strategy_name = ws.pattern_type.replace("strategy_", "")
                effective_strategies.append((strategy_name, ws.success_rate))

        # 성공률 순 정렬
        effective_strategies.sort(key=lambda x: x[1], reverse=True)

        # 조합 생성: 상위 전략 2-3개씩 조합
        combinations = []
        strategy_names = [s[0] for s in effective_strategies[:6]]

        for i in range(len(strategy_names)):
            for j in range(i + 1, len(strategy_names)):
                combinations.append([strategy_names[i], strategy_names[j]])
                if len(combinations) >= top_n:
                    break
            if len(combinations) >= top_n:
                break

        return combinations

    def get_summary(self) -> dict[str, Any]:
        """Get analysis summary.

        Returns:
            Summary dictionary
        """
        if not self._features_history:
            return {"error": "No data analyzed yet"}

        total = len(self._features_history)
        evasions = sum(1 for f in self._features_history if f.is_evasion)

        return {
            "total_samples": total,
            "total_evasions": evasions,
            "overall_evasion_rate": evasions / total if total > 0 else 0,
            "weak_spots_count": len(self._weak_spots),
            "top_weak_spots": [ws.to_dict() for ws in self._weak_spots[:5]],
        }

    def clear_history(self) -> None:
        """Clear analysis history."""
        self._features_history.clear()
        self._weak_spots.clear()


if __name__ == "__main__":
    # Demo
    print("=" * 60)
    print("Failure Pattern Analyzer Demo")
    print("=" * 60)

    analyzer = FailurePatternAnalyzer()

    # 샘플 데이터
    test_results = [
        {"original_text": "시발놈", "variant_text": "ㅅㅂ놈", "strategy_name": "chosung", "is_evasion": True},
        {"original_text": "시발놈", "variant_text": "시8놈", "strategy_name": "number_sub", "is_evasion": True},
        {"original_text": "시발놈", "variant_text": "시 발 놈", "strategy_name": "space_insertion", "is_evasion": True},
        {"original_text": "병신", "variant_text": "ㅂㅅ", "strategy_name": "chosung", "is_evasion": True},
        {"original_text": "병신", "variant_text": "병​신", "strategy_name": "zero_width", "is_evasion": True},
        {"original_text": "시발", "variant_text": "씨발", "strategy_name": "similar_char", "is_evasion": False},
        {"original_text": "시발", "variant_text": "시발", "strategy_name": "none", "is_evasion": False},
        {"original_text": "새끼", "variant_text": "ㅅㄲ", "strategy_name": "chosung", "is_evasion": True},
        {"original_text": "미친", "variant_text": "미1친", "strategy_name": "number_sub", "is_evasion": False},
        {"original_text": "지랄", "variant_text": "ㅈㄹ", "strategy_name": "chosung", "is_evasion": True},
    ]

    # 분석
    analysis = analyzer.analyze_batch(test_results)
    print("\n[Batch Analysis]")
    print(f"  Total samples: {analysis['total_samples']}")
    print(f"  Evasion rate: {analysis['evasion_rate']:.1%}")
    print("\n  Strategy effectiveness:")
    for name, rate in sorted(analysis["strategy_effectiveness"].items(), key=lambda x: x[1], reverse=True):
        print(f"    {name:20s}: {rate:.1%}")

    # 약점 식별
    weak_spots = analyzer.get_weak_spots(min_samples=2, min_success_rate=0.3)
    print("\n[Weak Spots]")
    for ws in weak_spots:
        print(f"  {ws.pattern_type}: {ws.success_rate:.1%} ({ws.sample_count} samples)")
        if ws.examples:
            print(f"    Examples: {ws.examples[:3]}")

    # 전략 조합 제안
    combinations = analyzer.suggest_strategy_combinations(top_n=3)
    print("\n[Suggested Strategy Combinations]")
    for combo in combinations:
        print(f"  {' + '.join(combo)}")
