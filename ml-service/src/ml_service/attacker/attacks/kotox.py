"""KOTOX-based attack strategies for Korean text obfuscation.

KOTOX (2025) 기반 공격 전략:
1. 도상적 자모 대체 (Iconic consonant)
2. 야민정음 (Yamin)
3. 한자 의미 대체 (CJK semantic)
4. 음절 배열 교란 (Syllable anagram)
5. 종합 기호 추가 (Symbol comprehensive)
6. KOTOX 복합 공격 (KOTOX mixed)
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from ..hangul_utils import decompose_syllable, compose_syllable
from .basic import space_insertion_attack, zero_width_attack


# =============================================================================
# KOTOX Dictionary Loading
# =============================================================================

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


# =============================================================================
# KOTOX Symbol Sets
# =============================================================================

KOTOX_SYMBOLS = {
    "hearts": ['♡', '♥', '♤', '♧'],
    "stars": ['★', '☆', '✦', '✧', '✩', '✪'],
    "circles": ['○', '●', '◎', '◯', '◈', '◉'],
    "brackets": ['【', '】', '《', '》', '「', '」', '『', '』'],
    "punctuation": ['‥', '…', '、', '。', '¿', '？'],
    "emotions": ['ε♡з', 'T^T', '≥ㅇ≤', '≥ㅅ≤', '≥ㅂ≤'],
    "special": ['¸', 'º', '°', '˛', '˚', '¯', '´'],
}


# =============================================================================
# Attack Strategy Implementations
# =============================================================================

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


# Export all
__all__ = [
    'iconic_consonant_attack',
    'yamin_attack',
    'cjk_semantic_attack',
    'syllable_anagram_attack',
    'symbol_comprehensive_attack',
    'kotox_mixed_attack',
    'KOTOX_SYMBOLS',
    '_load_kotox_dicts',
]
