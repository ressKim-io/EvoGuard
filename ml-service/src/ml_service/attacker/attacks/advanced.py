"""Advanced Korean attack strategies for adversarial text generation.

고급 공격 전략:
1. 역순 공격 (Reverse) - 시발 → 발시
2. 신조어/줄임말 (Slang) - 시발 → ㅅㅂ
3. 방언 변형 (Dialect) - 병신 → 뱅신
4. 호환용 자모 (Compat Jamo) - ㅅㅂ → ᄉᄇ
5. 부분 마스킹 (Partial Mask) - 시발 → 시X
6. 유니코드 변형 (Unicode Variant)
7. 맥락 주입 (Context Injection)
8. 리트 변형 (Leet Korean) - 시발 → 시8ㅏㄹ
9. 음절 교환 (Syllable Swap) - 병신 → 신병
10. 의도적 오타 (Typo Attack)
11. 강력한 복합 공격 (Heavy Mixed)
"""

from __future__ import annotations

import random

from ..hangul_utils import (
    is_hangul_syllable,
    decompose_syllable,
    compose_syllable,
)
from .basic import (
    chosung_attack,
    space_insertion_attack,
    zero_width_attack,
)


# =============================================================================
# 데이터 매핑
# =============================================================================

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

# 호환용 자모 (Compatibility Jamo)
COMPAT_CHOSEONG = {
    'ㄱ': 'ᄀ', 'ㄴ': 'ᄂ', 'ㄷ': 'ᄃ', 'ㄹ': 'ᄅ', 'ㅁ': 'ᄆ',
    'ㅂ': 'ᄇ', 'ㅅ': 'ᄉ', 'ㅇ': 'ᄋ', 'ㅈ': 'ᄌ', 'ㅊ': 'ᄎ',
    'ㅋ': 'ᄏ', 'ㅌ': 'ᄐ', 'ㅍ': 'ᄑ', 'ㅎ': 'ᄒ',
}


# =============================================================================
# Attack Strategy Implementations
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


def slang_attack(text: str) -> str:
    """신조어/줄임말 변환.

    Example: 시발 → ㅅㅂ, 병신 → ㅂㅅ
    """
    result = text

    for original, slangs in SLANG_MAP.items():
        if original in result and random.random() < 0.7:
            result = result.replace(original, random.choice(slangs), 1)

    return result


def dialect_attack(text: str) -> str:
    """방언/비표준어 변형.

    Example: 병신 → 뱅신, 꺼져 → 꺼지라
    """
    result = text

    for standard, dialects in DIALECT_MAP.items():
        if standard in result and random.random() < 0.5:
            result = result.replace(standard, random.choice(dialects), 1)

    return result


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


# Export all attack functions
__all__ = [
    'reverse_attack',
    'slang_attack',
    'dialect_attack',
    'compat_jamo_attack',
    'partial_mask_attack',
    'unicode_variant_attack',
    'context_injection_attack',
    'leet_korean_attack',
    'syllable_swap_attack',
    'typo_attack',
    'heavy_mixed_attack',
    'SLANG_MAP',
    'DIALECT_MAP',
    'COMPAT_CHOSEONG',
]
