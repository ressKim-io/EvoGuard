"""Korean text attack strategies package.

모든 공격 전략을 하나의 네임스페이스로 통합합니다.

모듈 구성:
- basic: 기본 공격 전략 (10개)
- advanced: 고급 공격 전략 (11개)
- community: 커뮤니티 특화 공격 (3개)
- kotox: KOTOX 기반 공격 (6개)
"""

from .basic import (
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
    NUMBER_SUBSTITUTION,
    KOREAN_TO_ENGLISH,
    SIMILAR_CHARS,
)

from .advanced import (
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
    SLANG_MAP,
    DIALECT_MAP,
    COMPAT_CHOSEONG,
)

from .community import (
    community_slang_attack,
    phonetic_transform_attack,
    emoji_combo_attack,
    COMMUNITY_SLANG,
    PHONETIC_MAP,
    EMOJI_COMBOS,
    TEXT_TO_EMOJI,
    ENDING_EMOJIS,
)

from .kotox import (
    iconic_consonant_attack,
    yamin_attack,
    cjk_semantic_attack,
    syllable_anagram_attack,
    symbol_comprehensive_attack,
    kotox_mixed_attack,
    KOTOX_SYMBOLS,
)


__all__ = [
    # === Basic attacks ===
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
    # Basic data
    'NUMBER_SUBSTITUTION',
    'KOREAN_TO_ENGLISH',
    'SIMILAR_CHARS',

    # === Advanced attacks ===
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
    # Advanced data
    'SLANG_MAP',
    'DIALECT_MAP',
    'COMPAT_CHOSEONG',

    # === Community attacks ===
    'community_slang_attack',
    'phonetic_transform_attack',
    'emoji_combo_attack',
    # Community data
    'COMMUNITY_SLANG',
    'PHONETIC_MAP',
    'EMOJI_COMBOS',
    'TEXT_TO_EMOJI',
    'ENDING_EMOJIS',

    # === KOTOX attacks ===
    'iconic_consonant_attack',
    'yamin_attack',
    'cjk_semantic_attack',
    'syllable_anagram_attack',
    'symbol_comprehensive_attack',
    'kotox_mixed_attack',
    # KOTOX data
    'KOTOX_SYMBOLS',
]
