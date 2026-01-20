"""Korean-specific attack strategies for adversarial text generation.

한국어 특화 공격 전략:
1. 초성 변환 (Chosung) - 시발 → ㅅㅂ
2. 자모 분리 (Jamo decomposition) - 바보 → ㅂㅏㅂㅗ
3. 숫자 치환 (Number substitution) - 시발 → 시8
4. 영어 발음 (English phonetic) - 시발 → tlqkf
5. 공백 삽입 (Space insertion) - 병신 → 병 신
6. 유사 문자 (Similar character) - 시발 → 씌발
7. 자음 반복 (Consonant repetition) - 시발 → 시ㅂㅏㄹ
8. 이모지 삽입 (Emoji insertion) - 시발 → 시🔥발
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Callable

# =============================================================================
# Korean Unicode Constants
# =============================================================================

# 한글 음절 범위: 가(0xAC00) ~ 힣(0xD7A3)
HANGUL_START = 0xAC00
HANGUL_END = 0xD7A3

# 초성 (Initial consonants) - 19개
CHOSEONG = [
    'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]

# 중성 (Medial vowels) - 21개
JUNGSEONG = [
    'ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
    'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ'
]

# 종성 (Final consonants) - 28개 (첫번째는 종성 없음)
JONGSEONG = [
    '', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ',
    'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
    'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ'
]


# =============================================================================
# Helper Functions
# =============================================================================

def is_hangul_syllable(char: str) -> bool:
    """Check if character is a complete Hangul syllable (가-힣)."""
    if len(char) != 1:
        return False
    code = ord(char)
    return HANGUL_START <= code <= HANGUL_END


def decompose_syllable(char: str) -> tuple[str, str, str] | None:
    """Decompose a Hangul syllable into (초성, 중성, 종성).

    Example: 한 → (ㅎ, ㅏ, ㄴ)
    """
    if not is_hangul_syllable(char):
        return None

    code = ord(char) - HANGUL_START
    cho_idx = code // (21 * 28)
    jung_idx = (code % (21 * 28)) // 28
    jong_idx = code % 28

    return (CHOSEONG[cho_idx], JUNGSEONG[jung_idx], JONGSEONG[jong_idx])


def compose_syllable(cho: str, jung: str, jong: str = '') -> str:
    """Compose Hangul syllable from (초성, 중성, 종성).

    Example: (ㅎ, ㅏ, ㄴ) → 한
    """
    try:
        cho_idx = CHOSEONG.index(cho)
        jung_idx = JUNGSEONG.index(jung)
        jong_idx = JONGSEONG.index(jong) if jong else 0
    except ValueError:
        return cho + jung + jong

    code = HANGUL_START + (cho_idx * 21 * 28) + (jung_idx * 28) + jong_idx
    return chr(code)


def extract_choseong(text: str) -> str:
    """Extract only 초성 from text.

    Example: 시발 → ㅅㅂ
    """
    result = []
    for char in text:
        decomposed = decompose_syllable(char)
        if decomposed:
            result.append(decomposed[0])
        else:
            result.append(char)
    return ''.join(result)


# =============================================================================
# Attack Strategy Implementations
# =============================================================================

def chosung_attack(text: str) -> str:
    """초성 변환: 일부 글자를 초성으로 변환.

    Example: 시발놈아 → ㅅㅂ놈아, 시ㅂ놈아
    """
    result = list(text)

    for i, char in enumerate(text):
        if is_hangul_syllable(char) and random.random() < 0.5:
            decomposed = decompose_syllable(char)
            if decomposed:
                result[i] = decomposed[0]

    return ''.join(result)


def jamo_decompose_attack(text: str) -> str:
    """자모 분리: 음절을 자모로 분리.

    Example: 바보 → ㅂㅏㅂㅗ
    """
    result = []

    for char in text:
        decomposed = decompose_syllable(char)
        if decomposed and random.random() < 0.4:
            cho, jung, jong = decomposed
            result.append(cho + jung + jong)
        else:
            result.append(char)

    return ''.join(result)


# 숫자 치환 맵
NUMBER_SUBSTITUTION = {
    '발': '8',
    '빨': '8',
    '팔': '8',
    '일': '1',
    '이': '2',
    '삼': '3',
    '사': '4',
    '오': '5',
    '육': '6',
    '칠': '7',
    '구': '9',
    '공': '0',
    '영': '0',
}

def number_substitution_attack(text: str) -> str:
    """숫자 치환: 발음이 비슷한 숫자로 치환.

    Example: 시발 → 시8, 십팔 → 18
    """
    result = text

    for korean, number in NUMBER_SUBSTITUTION.items():
        if korean in result and random.random() < 0.6:
            result = result.replace(korean, number, 1)

    return result


# 영어 발음 맵 (두벌식 키보드 기준)
KOREAN_TO_ENGLISH = {
    'ㅂ': 'q', 'ㅈ': 'w', 'ㄷ': 'e', 'ㄱ': 'r', 'ㅅ': 't',
    'ㅛ': 'y', 'ㅕ': 'u', 'ㅑ': 'i', 'ㅐ': 'o', 'ㅔ': 'p',
    'ㅁ': 'a', 'ㄴ': 's', 'ㅇ': 'd', 'ㄹ': 'f', 'ㅎ': 'g',
    'ㅗ': 'h', 'ㅓ': 'j', 'ㅏ': 'k', 'ㅣ': 'l',
    'ㅋ': 'z', 'ㅌ': 'x', 'ㅊ': 'c', 'ㅍ': 'v', 'ㅠ': 'b',
    'ㅜ': 'n', 'ㅡ': 'm',
    'ㅃ': 'Q', 'ㅉ': 'W', 'ㄸ': 'E', 'ㄲ': 'R', 'ㅆ': 'T',
}

def english_phonetic_attack(text: str) -> str:
    """영어 발음 변환: 한글을 영어 키보드 입력으로 변환.

    Example: 시발 → tlqkf
    """
    result = []

    for char in text:
        decomposed = decompose_syllable(char)
        if decomposed and random.random() < 0.7:
            cho, jung, jong = decomposed
            eng_cho = KOREAN_TO_ENGLISH.get(cho, cho)
            eng_jung = KOREAN_TO_ENGLISH.get(jung, jung)
            eng_jong = KOREAN_TO_ENGLISH.get(jong, jong) if jong else ''
            result.append(eng_cho + eng_jung + eng_jong)
        else:
            result.append(char)

    return ''.join(result)


def space_insertion_attack(text: str) -> str:
    """공백 삽입: 글자 사이에 공백 삽입.

    Example: 병신 → 병 신, 병ㅡ신
    """
    result = []
    spacers = [' ', 'ㅡ', '.', '_', '']

    for i, char in enumerate(text):
        result.append(char)
        if i < len(text) - 1 and is_hangul_syllable(char) and random.random() < 0.3:
            result.append(random.choice(spacers))

    return ''.join(result)


# 유사 문자 맵
SIMILAR_CHARS = {
    'ㅅ': ['ㅆ', 'ㅈ', 's'],
    'ㅂ': ['ㅃ', 'ㅍ', 'b'],
    'ㄱ': ['ㄲ', 'ㅋ', 'g'],
    'ㄷ': ['ㄸ', 'ㅌ', 'd'],
    'ㅈ': ['ㅉ', 'ㅊ', 'j'],
    'ㅏ': ['ㅑ', 'ㅐ', 'a'],
    'ㅓ': ['ㅕ', 'ㅔ'],
    'ㅗ': ['ㅛ', 'ㅚ', 'o'],
    'ㅜ': ['ㅠ', 'ㅟ', 'u'],
    '시': ['씨', '쉬', '싀'],
    '발': ['빨', '벌', '밟'],
    '놈': ['넘', '눔'],
    '새': ['쌔', '섀'],
    '끼': ['키', '띠'],
}

def similar_char_attack(text: str) -> str:
    """유사 문자 치환: 비슷한 문자로 치환.

    Example: 시발 → 씨발, 쉬발
    """
    result = text

    for original, similars in SIMILAR_CHARS.items():
        if original in result and random.random() < 0.5:
            result = result.replace(original, random.choice(similars), 1)

    return result


def emoji_insertion_attack(text: str) -> str:
    """이모지 삽입: 글자 사이에 이모지 삽입.

    Example: 시발 → 시🔥발, 병💀신
    """
    emojis = ['🔥', '💀', '😡', '🤬', '💢', '⚡', '🖕', '👊', '😤', '🤮']
    result = []

    for i, char in enumerate(text):
        result.append(char)
        if i < len(text) - 1 and is_hangul_syllable(char) and random.random() < 0.2:
            result.append(random.choice(emojis))

    return ''.join(result)


def zero_width_attack(text: str) -> str:
    """제로 너비 문자 삽입: 보이지 않는 문자 삽입.

    Example: 시발 → 시​발 (사이에 zero-width space)
    """
    zwc = [
        '\u200b',  # Zero-width space
        '\u200c',  # Zero-width non-joiner
        '\u200d',  # Zero-width joiner
        '\ufeff',  # Zero-width no-break space
    ]

    result = []
    for i, char in enumerate(text):
        result.append(char)
        if i < len(text) - 1 and random.random() < 0.3:
            result.append(random.choice(zwc))

    return ''.join(result)


def consonant_elongation_attack(text: str) -> str:
    """자음 늘리기: 초성이나 종성을 반복.

    Example: 시발 → 시ㅂㅏㄹ, 씨이발
    """
    result = []

    for char in text:
        decomposed = decompose_syllable(char)
        if decomposed and random.random() < 0.3:
            cho, jung, jong = decomposed
            if random.random() < 0.5:
                result.append(cho + char)
            else:
                result.append(char + jung)
        else:
            result.append(char)

    return ''.join(result)


def mixed_attack(text: str) -> str:
    """복합 공격: 여러 전략을 랜덤하게 조합."""
    strategies = [
        chosung_attack,
        jamo_decompose_attack,
        number_substitution_attack,
        space_insertion_attack,
        similar_char_attack,
        emoji_insertion_attack,
    ]

    result = text
    selected = random.sample(strategies, k=random.randint(2, 3))
    for strategy in selected:
        result = strategy(result)

    return result


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
]


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
