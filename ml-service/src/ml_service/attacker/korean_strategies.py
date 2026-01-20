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

KOTOX 기반 추가 전략 (2025):
9. 도상적 자모 대체 (Iconic consonant) - ㄱ → 勹, ㅂ → 廿
10. 야민정음 (Yamin) - 귀 → 커, 명 → 띵
11. 한자 의미 대체 (CJK semantic) - 수 → 水, 남 → 男
12. 음절 배열 교란 (Syllable anagram) - 오랜만에 → 오만랜에
13. 종합 기호 추가 (Symbol comprehensive) - 시발 → 시°♡발《》
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
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
# 고급 한국어 공격 전략 (Advanced Korean Attack Strategies)
# =============================================================================

def reverse_attack(text: str) -> str:
    """역순 공격: 단어 또는 글자 순서 뒤집기.

    Example: 시발 → 발시, 병신 → 신병
    """
    words = text.split()
    result = []

    for word in words:
        if len(word) >= 2 and random.random() < 0.6:
            # 한글 단어만 뒤집기
            hangul_chars = [c for c in word if is_hangul_syllable(c)]
            if len(hangul_chars) >= 2:
                result.append(word[::-1])
            else:
                result.append(word)
        else:
            result.append(word)

    return ' '.join(result)


# 신조어/줄임말 매핑
SLANG_MAP = {
    '시발': ['ㅅㅂ', '시부랄', '시팔', 'ㅅㅃ', '시1발', 'tlqkf'],
    '병신': ['ㅂㅅ', '병1신', '벵신', 'qudtls', 'ㅄ'],
    '새끼': ['ㅅㄲ', '색히', '샠기', 'torks'],
    '개새끼': ['ㄱㅅㄲ', '개색히', '개샠기'],
    '미친': ['ㅁㅊ', '미1친', '믻힌'],
    '씨발': ['ㅆㅂ', '씨1발', '씨팔', 'tlqkf'],
    '지랄': ['ㅈㄹ', '지1랄', 'wlfkf'],
    '꺼져': ['ㄲㅈ', '꺼1져', '껴저'],
    '닥쳐': ['ㄷㅊ', '닥1쳐', '닥처'],
    '죽어': ['ㅈㅇ', '죽1어', '줘거'],
}

def slang_attack(text: str) -> str:
    """신조어/줄임말 변환.

    Example: 시발 → ㅅㅂ, 병신 → ㅂㅅ
    """
    result = text

    for original, slangs in SLANG_MAP.items():
        if original in result and random.random() < 0.7:
            result = result.replace(original, random.choice(slangs), 1)

    return result


# 방언 변형 매핑
DIALECT_MAP = {
    '뭐': ['머', '뭣', '뭔'],
    '왜': ['와', '웨'],
    '아니': ['아이', '아니야'],
    '병신': ['뱅신', '벙신', '빙신'],
    '바보': ['빠보', '바부'],
    '멍청이': ['멍충이', '멍텅구리'],
    '꺼져': ['꺼지라', '꺼져라', '껴져'],
    '죽어': ['주거', '죽으라', '뒤져'],
    '미친놈': ['미친넘', '미친눔', '미친노무'],
}

def dialect_attack(text: str) -> str:
    """방언/비표준어 변형.

    Example: 병신 → 뱅신, 꺼져 → 꺼지라
    """
    result = text

    for standard, dialects in DIALECT_MAP.items():
        if standard in result and random.random() < 0.5:
            result = result.replace(standard, random.choice(dialects), 1)

    return result


# 호환용 자모 (Compatibility Jamo)
COMPAT_CHOSEONG = {
    'ㄱ': 'ᄀ', 'ㄴ': 'ᄂ', 'ㄷ': 'ᄃ', 'ㄹ': 'ᄅ', 'ㅁ': 'ᄆ',
    'ㅂ': 'ᄇ', 'ㅅ': 'ᄉ', 'ㅇ': 'ᄋ', 'ㅈ': 'ᄌ', 'ㅊ': 'ᄎ',
    'ㅋ': 'ᄏ', 'ㅌ': 'ᄐ', 'ㅍ': 'ᄑ', 'ㅎ': 'ᄒ',
}

def compat_jamo_attack(text: str) -> str:
    """호환용 자모 변환: 시각적으로 동일하지만 다른 유니코드.

    Example: ㅅㅂ → ᄉᄇ
    """
    result = []
    for char in text:
        if char in COMPAT_CHOSEONG and random.random() < 0.6:
            result.append(COMPAT_CHOSEONG[char])
        else:
            result.append(char)
    return ''.join(result)


def partial_mask_attack(text: str) -> str:
    """부분 마스킹: 일부 글자를 X, *, O 등으로 대체.

    Example: 시발 → 시X, 병신 → 병X
    """
    masks = ['X', '*', 'O', '○', '●', '□', '■', '_']
    result = list(text)

    for i, char in enumerate(text):
        if is_hangul_syllable(char) and random.random() < 0.25:
            result[i] = random.choice(masks)

    return ''.join(result)


# 유니코드 한글 변형 (Halfwidth/Fullwidth)
FULLWIDTH_MAP = {
    'ㄱ': 'ㄱ', 'ㄴ': 'ㄴ', 'ㄷ': 'ㄷ', 'ㄹ': 'ㄹ', 'ㅁ': 'ㅁ',
    'ㅂ': 'ㅂ', 'ㅅ': 'ㅅ', 'ㅇ': 'ㅇ', 'ㅈ': 'ㅈ',
}

def unicode_variant_attack(text: str) -> str:
    """유니코드 변형: 반각/전각 한글 혼용.

    Example: 시발 → 시ㅂㅏㄹ (혼합 형태)
    """
    result = []

    for char in text:
        decomposed = decompose_syllable(char)
        if decomposed and random.random() < 0.4:
            cho, jung, jong = decomposed
            # 일부만 자모로 분리
            if random.random() < 0.5:
                result.append(cho + jung + jong)
            else:
                result.append(char)
        else:
            result.append(char)

    return ''.join(result)


def context_injection_attack(text: str) -> str:
    """맥락 주입: 앞뒤에 무관한 텍스트 삽입.

    Example: 시발 → 오늘 날씨가 시발 좋네요
    """
    prefixes = [
        '사실', '근데', '아 진짜', '와', '헐', '음', '그냥', '뭐',
        '오늘', '아까', '방금', '이거', '저기',
    ]
    suffixes = [
        '네요', 'ㅋㅋ', 'ㅎㅎ', '...', '임', '이야', '인듯', '같아',
        '하네', '하다', '해서', '인데', '라고',
    ]

    result = text
    if random.random() < 0.5:
        result = random.choice(prefixes) + ' ' + result
    if random.random() < 0.5:
        result = result + ' ' + random.choice(suffixes)

    return result


def leet_korean_attack(text: str) -> str:
    """한글 리트(Leet) 변형: 숫자와 기호로 치환.

    Example: 시발 → 시8ㅏㄹ, 병신 → 병$ㅣㄴ
    """
    leet_map = {
        'ㅏ': ['ㅏ', 'a', '4'],
        'ㅓ': ['ㅓ', '3'],
        'ㅗ': ['ㅗ', '0', 'o'],
        'ㅜ': ['ㅜ', 'u'],
        'ㅣ': ['ㅣ', '1', 'i', 'l'],
        'ㅡ': ['ㅡ', '-', '_'],
        'ㅅ': ['ㅅ', '$', 's'],
        'ㅂ': ['ㅂ', '8', 'b'],
        '발': ['8', '발', 'ㅂㅏㄹ'],
        '신': ['$ㅣㄴ', '신', 'ㅅㅣㄴ'],
    }

    result = text
    for original, leets in leet_map.items():
        if original in result and random.random() < 0.5:
            result = result.replace(original, random.choice(leets), 1)

    return result


def syllable_swap_attack(text: str) -> str:
    """음절 교환: 인접한 음절 위치 교환.

    Example: 병신 → 신병, 시발놈 → 발시놈
    """
    chars = list(text)
    hangul_indices = [i for i, c in enumerate(chars) if is_hangul_syllable(c)]

    if len(hangul_indices) >= 2:
        # 랜덤하게 인접한 두 음절 교환
        for _ in range(random.randint(1, 2)):
            if len(hangul_indices) >= 2:
                idx = random.randint(0, len(hangul_indices) - 2)
                i, j = hangul_indices[idx], hangul_indices[idx + 1]
                if abs(i - j) == 1:  # 인접한 경우만
                    chars[i], chars[j] = chars[j], chars[i]

    return ''.join(chars)


def typo_attack(text: str) -> str:
    """의도적 오타: 키보드 인접 키로 치환.

    Example: 시발 → 씨발, 병신 → 벼신
    """
    # 두벌식 키보드 인접 키 맵
    adjacent_keys = {
        'ㅅ': ['ㅆ', 'ㄷ', 'ㅈ'],
        'ㅂ': ['ㅃ', 'ㅈ', 'ㅁ'],
        'ㅇ': ['ㅎ', 'ㄹ'],
        'ㄴ': ['ㅁ', 'ㄹ'],
        'ㅎ': ['ㄱ', 'ㅇ'],
        'ㅁ': ['ㄴ', 'ㅂ'],
        'ㄱ': ['ㄲ', 'ㅎ', 'ㅋ'],
        'ㄷ': ['ㄸ', 'ㅅ', 'ㅌ'],
        'ㅈ': ['ㅉ', 'ㅅ', 'ㅂ'],
        'ㅏ': ['ㅑ', 'ㅓ'],
        'ㅓ': ['ㅕ', 'ㅏ'],
        'ㅗ': ['ㅛ', 'ㅜ'],
        'ㅜ': ['ㅠ', 'ㅗ'],
    }

    result = []
    for char in text:
        decomposed = decompose_syllable(char)
        if decomposed and random.random() < 0.3:
            cho, jung, jong = decomposed
            # 초성 또는 중성 변형
            if cho in adjacent_keys and random.random() < 0.5:
                cho = random.choice(adjacent_keys[cho])
            if jung in adjacent_keys and random.random() < 0.5:
                jung = random.choice(adjacent_keys[jung])
            result.append(compose_syllable(cho, jung, jong))
        else:
            result.append(char)

    return ''.join(result)


def heavy_mixed_attack(text: str) -> str:
    """강력한 복합 공격: 여러 고급 전략 조합."""
    strategies = [
        chosung_attack,
        slang_attack,
        space_insertion_attack,
        partial_mask_attack,
        leet_korean_attack,
        typo_attack,
        zero_width_attack,
    ]

    result = text
    selected = random.sample(strategies, k=random.randint(3, 5))
    for strategy in selected:
        result = strategy(result)

    return result


# =============================================================================
# KOTOX-based Attack Strategies (2025)
# =============================================================================

# KOTOX 딕셔너리 로드 (lazy loading)
_KOTOX_DICTS: dict | None = None

def _load_kotox_dicts() -> dict:
    """Load KOTOX dictionaries lazily."""
    global _KOTOX_DICTS
    if _KOTOX_DICTS is not None:
        return _KOTOX_DICTS

    # KOTOX 데이터 경로 찾기
    possible_paths = [
        Path(__file__).parent.parent.parent.parent / "data" / "korean" / "KOTOX" / "rules",
        Path("/home/resshome/project/EvoGuard/ml-service/data/korean/KOTOX/rules"),
    ]

    rules_path = None
    for p in possible_paths:
        if p.exists():
            rules_path = p
            break

    if rules_path is None:
        # 기본 딕셔너리 사용
        _KOTOX_DICTS = {
            "iconic": {"consonant_dict": {}, "vowel_dict": {}, "yamin_dict": {}},
            "transliteration": {"meaning_dict": {}},
            "replace": {"power_replace_map": {}, "vowel_replace_map": {}},
        }
        return _KOTOX_DICTS

    try:
        with open(rules_path / "iconic_dictionary.json", "r", encoding="utf-8") as f:
            iconic = json.load(f)
        with open(rules_path / "transliterational_dictionary.json", "r", encoding="utf-8") as f:
            transliteration = json.load(f)
        with open(rules_path / "replace.json", "r", encoding="utf-8") as f:
            replace = json.load(f)

        _KOTOX_DICTS = {
            "iconic": iconic,
            "transliteration": transliteration,
            "replace": replace,
        }
    except Exception:
        _KOTOX_DICTS = {
            "iconic": {"consonant_dict": {}, "vowel_dict": {}, "yamin_dict": {}},
            "transliteration": {"meaning_dict": {}},
            "replace": {"power_replace_map": {}, "vowel_replace_map": {}},
        }

    return _KOTOX_DICTS


def iconic_consonant_attack(text: str) -> str:
    """도상적 자모 대체: 자음/모음을 시각적으로 유사한 문자로 치환.

    Example: 시발 → 人┃ㅂㅏㄹ, ㄱ → 勹
    """
    dicts = _load_kotox_dicts()
    consonant_dict = dicts["iconic"].get("consonant_dict", {})
    vowel_dict = dicts["iconic"].get("vowel_dict", {})

    result = []
    for char in text:
        decomposed = decompose_syllable(char)
        if decomposed and random.random() < 0.4:
            cho, jung, jong = decomposed
            # 초성 대체
            if cho in consonant_dict and random.random() < 0.5:
                cho = random.choice(consonant_dict[cho])
            # 중성 대체 (종성 없는 경우)
            if not jong and jung in vowel_dict and random.random() < 0.3:
                jung = random.choice(vowel_dict[jung])
            # 재조합 시도
            try:
                result.append(compose_syllable(cho, jung, jong))
            except (ValueError, IndexError):
                result.append(cho + jung + jong)
        else:
            result.append(char)

    return ''.join(result)


def yamin_attack(text: str) -> str:
    """야민정음 공격: 시각적으로 유사한 한글 음절로 치환.

    Example: 귀엽다 → 커엽다, 명품 → 띵품
    """
    dicts = _load_kotox_dicts()
    yamin_dict = dicts["iconic"].get("yamin_dict", {})

    result = text
    for original, replacements in yamin_dict.items():
        if original in result and random.random() < 0.6:
            result = result.replace(original, random.choice(replacements), 1)

    return result


def cjk_semantic_attack(text: str) -> str:
    """한자 의미 대체: 한글을 발음이 같은 한자로 치환.

    Example: 수상해 → 水상해, 남자 → 男자
    """
    dicts = _load_kotox_dicts()
    meaning_dict = dicts["transliteration"].get("meaning_dict", {})

    result = text
    for korean, hanja_list in meaning_dict.items():
        if korean in result and random.random() < 0.5:
            result = result.replace(korean, random.choice(hanja_list), 1)

    return result


def syllable_anagram_attack(text: str) -> str:
    """음절 배열 교란: 단어 내 중간 음절 순서를 섞음.

    Example: 오랜만에 → 오만랜에, 외국여행 → 외여국행
    """
    words = text.split()
    result = []

    for word in words:
        if len(word) <= 2:
            result.append(word)
            continue

        if random.random() < 0.6:
            chars = list(word)
            # 첫 글자와 마지막 글자 유지, 중간만 섞기
            if len(chars) >= 3:
                middle = chars[1:-1]
                if len(middle) > 1:
                    shuffled = middle[:]
                    for _ in range(3):
                        random.shuffle(shuffled)
                        if shuffled != middle:
                            break
                    chars = [chars[0]] + shuffled + [chars[-1]]
            result.append(''.join(chars))
        else:
            result.append(word)

    return ' '.join(result)


# KOTOX 기호 집합
KOTOX_SYMBOLS = {
    "hearts": ['♡', '♥', '♤', '♧'],
    "stars": ['★', '☆', '✦', '✧', '✩', '✪'],
    "circles": ['○', '●', '◎', '◯', '◈', '◉'],
    "brackets": ['【', '】', '《', '》', '「', '」', '『', '』'],
    "punctuation": ['‥', '…', '、', '。', '¿', '？'],
    "emotions": ['ε♡з', 'T^T', '≥ㅇ≤', '≥ㅅ≤', '≥ㅂ≤'],
    "special": ['¸', 'º', '°', '˛', '˚', '¯', '´'],
}

def symbol_comprehensive_attack(text: str) -> str:
    """종합 기호 추가: 다양한 특수 기호를 텍스트에 삽입.

    Example: 시발 → 시°♡발《》, 병신 → ★병...신★
    """
    result = list(text)

    # 단어 사이에 기호 삽입
    for i in range(len(result) - 1, 0, -1):
        if random.random() < 0.2:
            symbol_type = random.choice(list(KOTOX_SYMBOLS.keys()))
            symbol = random.choice(KOTOX_SYMBOLS[symbol_type])
            result.insert(i, symbol)

    # 앞뒤에 괄호 추가
    if random.random() < 0.3:
        bracket = random.choice([('《', '》'), ('「', '」'), ('【', '】')])
        result = [bracket[0]] + result + [bracket[1]]

    # 끝에 감정 표현 추가
    if random.random() < 0.2:
        emotion = random.choice(KOTOX_SYMBOLS["emotions"])
        result.append(' ' + emotion)

    return ''.join(result)


def kotox_mixed_attack(text: str) -> str:
    """KOTOX 복합 공격: KOTOX 기반 여러 전략 조합."""
    strategies = [
        iconic_consonant_attack,
        yamin_attack,
        cjk_semantic_attack,
        syllable_anagram_attack,
        symbol_comprehensive_attack,
        space_insertion_attack,
        zero_width_attack,
    ]

    result = text
    selected = random.sample(strategies, k=random.randint(2, 4))
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
