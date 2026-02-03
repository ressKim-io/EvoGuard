"""Korean Attack Strategy Registry and Utilities.

한국어 텍스트 공격 전략 레지스트리.
모든 공격 함수는 attacks 패키지에서 관리됩니다.

전략 분류:
- 기본 전략 (10개): chosung, jamo_decompose, number_sub, ...
- 고급 전략 (11개): reverse, slang, dialect, ...
- Phase 7 전략 (3개): community_slang, phonetic_transform, emoji_combo
- KOTOX 전략 (6개): iconic_consonant, yamin, cjk_semantic, ...

총 30개 공격 전략 제공
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

# Import all attack functions from attacks package
from .attacks import (
    # Basic attacks
    chosung_attack,
    jamo_decompose_attack,
    number_substitution_attack,
    english_phonetic_attack,
    space_insertion_attack,
    similar_char_attack,
    emoji_insertion_attack,
    zero_width_attack,
    consonant_elongation_attack,
    mixed_attack,
    # Advanced attacks
    reverse_attack,
    slang_attack,
    dialect_attack,
    compat_jamo_attack,
    partial_mask_attack,
    unicode_variant_attack,
    context_injection_attack,
    leet_korean_attack,
    syllable_swap_attack,
    typo_attack,
    heavy_mixed_attack,
    # Community attacks
    community_slang_attack,
    phonetic_transform_attack,
    emoji_combo_attack,
    # KOTOX attacks
    iconic_consonant_attack,
    yamin_attack,
    cjk_semantic_attack,
    syllable_anagram_attack,
    symbol_comprehensive_attack,
    kotox_mixed_attack,
)


# =============================================================================
# Strategy Registry
# =============================================================================

@dataclass
class KoreanAttackStrategy:
    """Korean attack strategy definition."""
    name: str
    description: str
    transform: Callable[[str], str]
    example_input: str
    example_output: str


KOREAN_ATTACK_STRATEGIES: list[KoreanAttackStrategy] = [
    # === 기본 전략 ===
    KoreanAttackStrategy(
        name="chosung",
        description="초성 변환: 글자를 초성으로 변환",
        transform=chosung_attack,
        example_input="시발놈",
        example_output="ㅅㅂ놈",
    ),
    KoreanAttackStrategy(
        name="jamo_decompose",
        description="자모 분리: 음절을 자모로 분리",
        transform=jamo_decompose_attack,
        example_input="바보",
        example_output="ㅂㅏㅂㅗ",
    ),
    KoreanAttackStrategy(
        name="number_sub",
        description="숫자 치환: 발음이 비슷한 숫자로 치환",
        transform=number_substitution_attack,
        example_input="시발",
        example_output="시8",
    ),
    KoreanAttackStrategy(
        name="english_phonetic",
        description="영어 발음: 두벌식 키보드 영어로 변환",
        transform=english_phonetic_attack,
        example_input="시발",
        example_output="tlqkf",
    ),
    KoreanAttackStrategy(
        name="space_insertion",
        description="공백 삽입: 글자 사이에 공백/특수문자 삽입",
        transform=space_insertion_attack,
        example_input="병신",
        example_output="병 신",
    ),
    KoreanAttackStrategy(
        name="similar_char",
        description="유사 문자: 비슷한 문자로 치환",
        transform=similar_char_attack,
        example_input="시발",
        example_output="씨발",
    ),
    KoreanAttackStrategy(
        name="emoji_insertion",
        description="이모지 삽입: 글자 사이에 이모지 삽입",
        transform=emoji_insertion_attack,
        example_input="시발",
        example_output="시🔥발",
    ),
    KoreanAttackStrategy(
        name="zero_width",
        description="제로 너비 문자: 보이지 않는 문자 삽입",
        transform=zero_width_attack,
        example_input="시발",
        example_output="시​발",
    ),
    KoreanAttackStrategy(
        name="consonant_elongation",
        description="자음 늘리기: 초성/종성 반복",
        transform=consonant_elongation_attack,
        example_input="시발",
        example_output="ㅅ시발",
    ),
    KoreanAttackStrategy(
        name="mixed",
        description="복합 공격: 여러 전략 조합",
        transform=mixed_attack,
        example_input="시발놈아",
        example_output="ㅅ8 놈아",
    ),
    # === 고급 전략 ===
    KoreanAttackStrategy(
        name="reverse",
        description="역순 공격: 글자 순서 뒤집기",
        transform=reverse_attack,
        example_input="시발",
        example_output="발시",
    ),
    KoreanAttackStrategy(
        name="slang",
        description="신조어/줄임말: 인터넷 은어로 변환",
        transform=slang_attack,
        example_input="시발",
        example_output="시부랄",
    ),
    KoreanAttackStrategy(
        name="dialect",
        description="방언 변형: 비표준어/사투리로 변환",
        transform=dialect_attack,
        example_input="병신",
        example_output="뱅신",
    ),
    KoreanAttackStrategy(
        name="compat_jamo",
        description="호환용 자모: 다른 유니코드 자모 사용",
        transform=compat_jamo_attack,
        example_input="ㅅㅂ",
        example_output="ᄉᄇ",
    ),
    KoreanAttackStrategy(
        name="partial_mask",
        description="부분 마스킹: 일부 글자를 X, * 등으로 대체",
        transform=partial_mask_attack,
        example_input="시발",
        example_output="시X",
    ),
    KoreanAttackStrategy(
        name="unicode_variant",
        description="유니코드 변형: 자모 분리 혼용",
        transform=unicode_variant_attack,
        example_input="시발",
        example_output="시ㅂㅏㄹ",
    ),
    KoreanAttackStrategy(
        name="context_injection",
        description="맥락 주입: 앞뒤에 무관한 텍스트 삽입",
        transform=context_injection_attack,
        example_input="시발",
        example_output="아 진짜 시발 ㅋㅋ",
    ),
    KoreanAttackStrategy(
        name="leet_korean",
        description="한글 리트: 숫자와 기호로 치환",
        transform=leet_korean_attack,
        example_input="시발",
        example_output="시8ㅏㄹ",
    ),
    KoreanAttackStrategy(
        name="syllable_swap",
        description="음절 교환: 인접한 음절 위치 교환",
        transform=syllable_swap_attack,
        example_input="병신",
        example_output="신병",
    ),
    KoreanAttackStrategy(
        name="typo",
        description="의도적 오타: 키보드 인접 키로 치환",
        transform=typo_attack,
        example_input="시발",
        example_output="씨발",
    ),
    KoreanAttackStrategy(
        name="heavy_mixed",
        description="강력한 복합 공격: 여러 고급 전략 조합",
        transform=heavy_mixed_attack,
        example_input="시발놈아",
        example_output="ㅅ8 놈​아",
    ),
    # === Phase 7 전략 (2026) ===
    KoreanAttackStrategy(
        name="community_slang",
        description="커뮤니티 특화 은어: DC, 루리웹, 에펨코리아 등의 은어로 변환",
        transform=community_slang_attack,
        example_input="병신 진짜 미친",
        example_output="븅신 ㄹㅇ 믻친",
    ),
    KoreanAttackStrategy(
        name="phonetic_transform",
        description="음성 변환: 발음 기반으로 텍스트 변형",
        transform=phonetic_transform_attack,
        example_input="시발 병신",
        example_output="씨발 뼝신",
    ),
    KoreanAttackStrategy(
        name="emoji_combo",
        description="이모지 조합 강화: 의미 전달 이모지 조합으로 변환",
        transform=emoji_combo_attack,
        example_input="병신 꺼져",
        example_output="🧠❌ 👋 🤬",
    ),
    # === KOTOX 기반 전략 (2025) ===
    KoreanAttackStrategy(
        name="iconic_consonant",
        description="도상적 자모 대체: 자음/모음을 유사 문자로 치환 (KOTOX)",
        transform=iconic_consonant_attack,
        example_input="시발",
        example_output="人ㅣ발",
    ),
    KoreanAttackStrategy(
        name="yamin",
        description="야민정음: 시각적 유사 음절로 치환 (KOTOX)",
        transform=yamin_attack,
        example_input="귀엽다",
        example_output="커엽다",
    ),
    KoreanAttackStrategy(
        name="cjk_semantic",
        description="한자 의미 대체: 발음 같은 한자로 치환 (KOTOX)",
        transform=cjk_semantic_attack,
        example_input="수상해",
        example_output="水상해",
    ),
    KoreanAttackStrategy(
        name="syllable_anagram",
        description="음절 배열 교란: 중간 음절 순서 섞기 (KOTOX)",
        transform=syllable_anagram_attack,
        example_input="오랜만에",
        example_output="오만랜에",
    ),
    KoreanAttackStrategy(
        name="symbol_comprehensive",
        description="종합 기호 추가: 다양한 특수 기호 삽입 (KOTOX)",
        transform=symbol_comprehensive_attack,
        example_input="시발",
        example_output="《시°발》",
    ),
    KoreanAttackStrategy(
        name="kotox_mixed",
        description="KOTOX 복합 공격: KOTOX 기반 여러 전략 조합",
        transform=kotox_mixed_attack,
        example_input="시발놈아",
        example_output="《人ㅣ°발》놈亜",
    ),
]


# =============================================================================
# Utility Functions
# =============================================================================

def get_korean_strategies() -> list[KoreanAttackStrategy]:
    """Get all Korean attack strategies."""
    return KOREAN_ATTACK_STRATEGIES


def apply_korean_attack(text: str, strategy_name: str) -> str:
    """Apply a specific Korean attack strategy."""
    for strategy in KOREAN_ATTACK_STRATEGIES:
        if strategy.name == strategy_name:
            return strategy.transform(text)

    raise ValueError(f"Unknown strategy: {strategy_name}")


def apply_random_korean_attacks(text: str, num_variants: int = 5) -> list[tuple[str, str]]:
    """Apply random Korean attack strategies.

    Returns:
        List of (strategy_name, transformed_text) tuples
    """
    results = []
    strategies = get_korean_strategies()

    for _ in range(num_variants):
        strategy = random.choice(strategies)
        transformed = strategy.transform(text)
        if transformed != text:
            results.append((strategy.name, transformed))

    return results


# =============================================================================
# Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("한국어 공격 전략 데모")
    print("=" * 60)

    test_texts = ["시발놈아", "병신같은놈", "꺼져 미친놈", "죽어라 쓰레기"]

    for text in test_texts:
        print(f"\n원본: {text}")
        print("-" * 40)

        for strategy in KOREAN_ATTACK_STRATEGIES:
            random.seed(42)
            result = strategy.transform(text)
            print(f"  {strategy.name:20s}: {result}")
