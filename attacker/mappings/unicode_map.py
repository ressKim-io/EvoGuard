"""Unicode character mappings for evasion strategies."""

# Zero-width characters for invisible insertion
ZERO_WIDTH_CHARS: tuple[str, ...] = (
    "\u200b",  # Zero Width Space
    "\u200c",  # Zero Width Non-Joiner
    "\u200d",  # Zero Width Joiner
    "\u2060",  # Word Joiner
    "\ufeff",  # Zero Width No-Break Space (BOM)
)

# Combining diacritical marks for character obfuscation
COMBINING_DIACRITICS: tuple[str, ...] = (
    "\u0308",  # Combining Diaeresis (umlaut)
    "\u0301",  # Combining Acute Accent
    "\u0300",  # Combining Grave Accent
    "\u0302",  # Combining Circumflex
    "\u0303",  # Combining Tilde
    "\u0307",  # Combining Dot Above
    "\u0327",  # Combining Cedilla
    "\u0328",  # Combining Ogonek
    "\u030a",  # Combining Ring Above
    "\u030c",  # Combining Caron
)

# Fullwidth character mapping (ASCII to fullwidth)
FULLWIDTH_MAP: dict[str, str] = {
    # Uppercase letters
    "A": "\uff21", "B": "\uff22", "C": "\uff23", "D": "\uff24", "E": "\uff25",
    "F": "\uff26", "G": "\uff27", "H": "\uff28", "I": "\uff29", "J": "\uff2a",
    "K": "\uff2b", "L": "\uff2c", "M": "\uff2d", "N": "\uff2e", "O": "\uff2f",
    "P": "\uff30", "Q": "\uff31", "R": "\uff32", "S": "\uff33", "T": "\uff34",
    "U": "\uff35", "V": "\uff36", "W": "\uff37", "X": "\uff38", "Y": "\uff39",
    "Z": "\uff3a",
    # Lowercase letters
    "a": "\uff41", "b": "\uff42", "c": "\uff43", "d": "\uff44", "e": "\uff45",
    "f": "\uff46", "g": "\uff47", "h": "\uff48", "i": "\uff49", "j": "\uff4a",
    "k": "\uff4b", "l": "\uff4c", "m": "\uff4d", "n": "\uff4e", "o": "\uff4f",
    "p": "\uff50", "q": "\uff51", "r": "\uff52", "s": "\uff53", "t": "\uff54",
    "u": "\uff55", "v": "\uff56", "w": "\uff57", "x": "\uff58", "y": "\uff59",
    "z": "\uff5a",
    # Digits
    "0": "\uff10", "1": "\uff11", "2": "\uff12", "3": "\uff13", "4": "\uff14",
    "5": "\uff15", "6": "\uff16", "7": "\uff17", "8": "\uff18", "9": "\uff19",
}

# Korean Jamo decomposition mapping (syllable to jamo)
# Initial consonants (Choseong)
_CHOSEONG = (
    "ㄱ", "ㄲ", "ㄴ", "ㄷ", "ㄸ", "ㄹ", "ㅁ", "ㅂ", "ㅃ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅉ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)

# Medial vowels (Jungseong)
_JUNGSEONG = (
    "ㅏ", "ㅐ", "ㅑ", "ㅒ", "ㅓ", "ㅔ", "ㅕ", "ㅖ", "ㅗ", "ㅘ",
    "ㅙ", "ㅚ", "ㅛ", "ㅜ", "ㅝ", "ㅞ", "ㅟ", "ㅠ", "ㅡ", "ㅢ", "ㅣ",
)

# Final consonants (Jongseong) - empty string for no final
_JONGSEONG = (
    "", "ㄱ", "ㄲ", "ㄳ", "ㄴ", "ㄵ", "ㄶ", "ㄷ", "ㄹ", "ㄺ",
    "ㄻ", "ㄼ", "ㄽ", "ㄾ", "ㄿ", "ㅀ", "ㅁ", "ㅂ", "ㅄ", "ㅅ",
    "ㅆ", "ㅇ", "ㅈ", "ㅊ", "ㅋ", "ㅌ", "ㅍ", "ㅎ",
)

# Hangul syllable Unicode range
_HANGUL_BASE = 0xAC00
_HANGUL_END = 0xD7A3


def decompose_korean_syllable(char: str) -> str | None:
    """Decompose a Korean syllable into its Jamo components.

    Args:
        char: Single Korean syllable character

    Returns:
        Decomposed Jamo string or None if not a Korean syllable
    """
    code = ord(char)
    if not (_HANGUL_BASE <= code <= _HANGUL_END):
        return None

    offset = code - _HANGUL_BASE
    cho_idx = offset // (21 * 28)
    jung_idx = (offset % (21 * 28)) // 28
    jong_idx = offset % 28

    cho = _CHOSEONG[cho_idx]
    jung = _JUNGSEONG[jung_idx]
    jong = _JONGSEONG[jong_idx]

    return cho + jung + jong


# Export decomposition for direct use
JAMO_MAP = {
    "decompose": decompose_korean_syllable,
    "choseong": _CHOSEONG,
    "jungseong": _JUNGSEONG,
    "jongseong": _JONGSEONG,
}
