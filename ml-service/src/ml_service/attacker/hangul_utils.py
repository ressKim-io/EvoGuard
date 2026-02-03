"""Korean Hangul utility functions for text processing.

한글 유니코드 처리를 위한 유틸리티 함수들:
- 한글 음절 분해/조합
- 초성/중성/종성 추출
- 한글 여부 판별
"""

from __future__ import annotations

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
