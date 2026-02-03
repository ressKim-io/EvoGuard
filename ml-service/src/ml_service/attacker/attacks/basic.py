"""Basic Korean attack strategies for adversarial text generation.

기본 공격 전략:
1. 초성 변환 (Chosung) - 시발 → ㅅㅂ
2. 자모 분리 (Jamo decomposition) - 바보 → ㅂㅏㅂㅗ
3. 숫자 치환 (Number substitution) - 시발 → 시8
4. 영어 발음 (English phonetic) - 시발 → tlqkf
5. 공백 삽입 (Space insertion) - 병신 → 병 신
6. 유사 문자 (Similar character) - 시발 → 씌발
7. 이모지 삽입 (Emoji insertion) - 시발 → 시🔥발
8. 제로 너비 문자 (Zero-width) - 보이지 않는 문자 삽입
9. 자음 늘리기 (Consonant elongation)
10. 복합 공격 (Mixed attack)
"""

from __future__ import annotations

import random

from ..hangul_utils import (
    is_hangul_syllable,
    decompose_syllable,
    CHOSEONG,
    JUNGSEONG,
    JONGSEONG,
)


# =============================================================================
# 숫자 치환 맵
# =============================================================================

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


def number_substitution_attack(text: str) -> str:
    """숫자 치환: 발음이 비슷한 숫자로 치환.

    Example: 시발 → 시8, 십팔 → 18
    """
    result = text

    for korean, number in NUMBER_SUBSTITUTION.items():
        if korean in result and random.random() < 0.6:
            result = result.replace(korean, number, 1)

    return result


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


# Export all attack functions
__all__ = [
    'chosung_attack',
    'jamo_decompose_attack',
    'number_substitution_attack',
    'english_phonetic_attack',
    'space_insertion_attack',
    'similar_char_attack',
    'emoji_insertion_attack',
    'zero_width_attack',
    'consonant_elongation_attack',
    'mixed_attack',
    'NUMBER_SUBSTITUTION',
    'KOREAN_TO_ENGLISH',
    'SIMILAR_CHARS',
]
