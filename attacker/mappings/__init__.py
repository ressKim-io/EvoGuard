"""Character mapping tables for evasion strategies."""

from attacker.mappings.homoglyph_map import HOMOGLYPH_MAP
from attacker.mappings.leetspeak_map import LEETSPEAK_MAPS
from attacker.mappings.unicode_map import (
    COMBINING_DIACRITICS,
    FULLWIDTH_MAP,
    JAMO_MAP,
    ZERO_WIDTH_CHARS,
)

__all__ = [
    "COMBINING_DIACRITICS",
    "FULLWIDTH_MAP",
    "HOMOGLYPH_MAP",
    "JAMO_MAP",
    "LEETSPEAK_MAPS",
    "ZERO_WIDTH_CHARS",
]
