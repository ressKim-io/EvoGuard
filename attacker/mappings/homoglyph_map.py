"""Homoglyph character mappings for visual similarity attacks."""
# ruff: noqa: RUF003

# Homoglyph mapping: ASCII character -> list of visually similar Unicode characters
HOMOGLYPH_MAP: dict[str, tuple[str, ...]] = {
    # Lowercase Latin -> Cyrillic/Greek/Mathematical/Other
    "a": (
        "\u0430",  # Cyrillic а
        "\u03b1",  # Greek alpha α
        "\u0251",  # Latin alpha ɑ
        "\u1d00",  # Small capital A ᴀ
    ),
    "b": (
        "\u0184",  # Latin Tone Six Ƅ
        "\u042c",  # Cyrillic Ь (soft sign)
        "\u13cf",  # Cherokee Ꮟ
    ),
    "c": (
        "\u0441",  # Cyrillic с
        "\u03f2",  # Greek lunate sigma ϲ
        "\u217d",  # Roman numeral ⅽ
        "\u1d04",  # Small capital C ᴄ
    ),
    "d": (
        "\u0501",  # Cyrillic ԁ
        "\u217e",  # Roman numeral ⅾ
        "\u13e7",  # Cherokee Ꮷ
    ),
    "e": (
        "\u0435",  # Cyrillic е
        "\u0117",  # Latin ė
        "\u0229",  # Latin ȩ
        "\u1eb9",  # Latin ẹ
    ),
    "f": (
        "\u017f",  # Latin long s ſ
        "\u0192",  # Latin ƒ
    ),
    "g": (
        "\u0261",  # Latin script g ɡ
        "\u0581",  # Armenian ց
    ),
    "h": (
        "\u04bb",  # Cyrillic һ
        "\u0570",  # Armenian հ
        "\u13c2",  # Cherokee Ꮒ
    ),
    "i": (
        "\u0456",  # Cyrillic і
        "\u03b9",  # Greek iota ι
        "\u0131",  # Latin dotless i ı
        "\u2170",  # Roman numeral ⅰ
        "\u1d09",  # Turned i ᴉ
    ),
    "j": (
        "\u0458",  # Cyrillic ј
        "\u03f3",  # Greek yot ϳ
    ),
    "k": (
        "\u03ba",  # Greek kappa κ
        "\u043a",  # Cyrillic к
    ),
    "l": (
        "\u04cf",  # Cyrillic ӏ
        "\u0399",  # Greek capital iota Ι
        "\u2113",  # Script small l ℓ
        "\u217c",  # Roman numeral ⅼ
    ),
    "m": (
        "\u043c",  # Cyrillic м
        "\u217f",  # Roman numeral ⅿ
    ),
    "n": (
        "\u0578",  # Armenian ո
        "\u03c0",  # Greek pi π (stretched)
    ),
    "o": (
        "\u043e",  # Cyrillic о
        "\u03bf",  # Greek omicron ο
        "\u0585",  # Armenian օ
        "\u0ce6",  # Kannada ೦
        "\u0d66",  # Malayalam ൦
    ),
    "p": (
        "\u0440",  # Cyrillic р
        "\u03c1",  # Greek rho ρ
        "\u2374",  # APL rho ⍴
    ),
    "q": (
        "\u051b",  # Cyrillic ԛ
        "\u0566",  # Armenian զ
    ),
    "r": (
        "\u0433",  # Cyrillic г
        "\u0263",  # Latin gamma ɣ
    ),
    "s": (
        "\u0455",  # Cyrillic ѕ
        "\u03c2",  # Greek final sigma ς
        "\u0282",  # Latin s with hook ʂ
    ),
    "t": (
        "\u03c4",  # Greek tau τ
        "\u0442",  # Cyrillic т
    ),
    "u": (
        "\u03c5",  # Greek upsilon υ
        "\u057d",  # Armenian ս
        "\u1d1c",  # Small capital U ᴜ
    ),
    "v": (
        "\u03bd",  # Greek nu ν
        "\u0475",  # Cyrillic izhitsa ѵ
        "\u2174",  # Roman numeral ⅴ
    ),
    "w": (
        "\u0461",  # Cyrillic omega ѡ
        "\u03c9",  # Greek omega ω
    ),
    "x": (
        "\u0445",  # Cyrillic х
        "\u03c7",  # Greek chi χ
        "\u2179",  # Roman numeral ⅹ
    ),
    "y": (
        "\u0443",  # Cyrillic у
        "\u03b3",  # Greek gamma γ
        "\u0443",  # Cyrillic у
    ),
    "z": (
        "\u0290",  # Latin z with retroflex hook ʐ
        "\u01b6",  # Latin z with stroke ƶ
    ),
    # Uppercase Latin -> Cyrillic/Greek
    "A": (
        "\u0410",  # Cyrillic А
        "\u0391",  # Greek Alpha Α
        "\u13aa",  # Cherokee Ꭺ
    ),
    "B": (
        "\u0412",  # Cyrillic В
        "\u0392",  # Greek Beta Β
        "\u13f4",  # Cherokee Ᏼ
    ),
    "C": (
        "\u0421",  # Cyrillic С
        "\u03f9",  # Greek lunate sigma Ϲ
        "\u216d",  # Roman numeral Ⅽ
    ),
    "D": (
        "\u216e",  # Roman numeral Ⅾ
        "\u13a0",  # Cherokee Ꭰ
    ),
    "E": (
        "\u0415",  # Cyrillic Е
        "\u0395",  # Greek Epsilon Ε
        "\u13ac",  # Cherokee Ꭼ
    ),
    "F": (
        "\u03dc",  # Greek Digamma Ϝ
    ),
    "G": (
        "\u050c",  # Cyrillic Ԍ
        "\u13c0",  # Cherokee Ꮐ
    ),
    "H": (
        "\u041d",  # Cyrillic Н
        "\u0397",  # Greek Eta Η
        "\u13bb",  # Cherokee Ꮋ
    ),
    "I": (
        "\u0406",  # Cyrillic І
        "\u0399",  # Greek Iota Ι
        "\u2160",  # Roman numeral Ⅰ
    ),
    "J": (
        "\u0408",  # Cyrillic Ј
        "\u13ab",  # Cherokee Ꭻ
    ),
    "K": (
        "\u039a",  # Greek Kappa Κ
        "\u041a",  # Cyrillic К
        "\u13e6",  # Cherokee Ꮶ
    ),
    "L": (
        "\u13de",  # Cherokee Ꮮ
        "\u216c",  # Roman numeral Ⅼ
    ),
    "M": (
        "\u041c",  # Cyrillic М
        "\u039c",  # Greek Mu Μ
        "\u216f",  # Roman numeral Ⅿ
    ),
    "N": (
        "\u039d",  # Greek Nu Ν
        "\u2115",  # Double-struck N ℕ
    ),
    "O": (
        "\u041e",  # Cyrillic О
        "\u039f",  # Greek Omicron Ο
        "\u13a4",  # Cherokee Ꭴ
    ),
    "P": (
        "\u0420",  # Cyrillic Р
        "\u03a1",  # Greek Rho Ρ
        "\u13b3",  # Cherokee Ꮃ
    ),
    "Q": (
        "\u051a",  # Cyrillic Ԛ
    ),
    "R": (
        "\u13a1",  # Cherokee Ꭱ
        "\u0280",  # Small capital R ʀ
    ),
    "S": (
        "\u0405",  # Cyrillic Ѕ
        "\u13da",  # Cherokee Ꮪ
    ),
    "T": (
        "\u0422",  # Cyrillic Т
        "\u03a4",  # Greek Tau Τ
        "\u13a2",  # Cherokee Ꭲ
    ),
    "U": (
        "\u054d",  # Armenian Ս
    ),
    "V": (
        "\u2164",  # Roman numeral Ⅴ
        "\u13d9",  # Cherokee Ꮩ
    ),
    "W": (
        "\u13b3",  # Cherokee Ꮃ
        "\u13ef",  # Cherokee Ᏻ
    ),
    "X": (
        "\u0425",  # Cyrillic Х
        "\u03a7",  # Greek Chi Χ
        "\u2169",  # Roman numeral Ⅹ
    ),
    "Y": (
        "\u03a5",  # Greek Upsilon Υ
        "\u04ae",  # Cyrillic Ү
        "\u13a9",  # Cherokee Ꭹ
    ),
    "Z": (
        "\u0396",  # Greek Zeta Ζ
        "\u13c3",  # Cherokee Ꮓ
    ),
    # Digits
    "0": (
        "\u041e",  # Cyrillic О
        "\u039f",  # Greek Omicron Ο
        "\u2070",  # Superscript ⁰
    ),
    "1": (
        "\u0031",  # Digit one 1
        "\u2170",  # Roman numeral ⅰ
        "\u0399",  # Greek Iota Ι
        "\u04cf",  # Cyrillic palochka ӏ
    ),
    "2": (
        "\u01a7",  # Latin Tone Two Ƨ
        "\u03e8",  # Coptic Ϩ
    ),
    "3": (
        "\u0417",  # Cyrillic З
        "\u01b7",  # Latin Ezh Ʒ
    ),
    "4": (
        "\u13ce",  # Cherokee Ꮞ
    ),
    "5": (
        "\u01bc",  # Latin Tone Five Ƽ
    ),
    "6": (
        "\u0431",  # Cyrillic б
        "\u13ee",  # Cherokee Ᏺ
    ),
    "7": (
        "\u13f4",  # Cherokee Ᏼ (similar shape)
    ),
    "8": (
        "\u0222",  # Latin Ȣ
        "\u0b03",  # Oriya ଃ
    ),
    "9": (
        "\u0a68",  # Gurmukhi ੨
        "\u13ed",  # Cherokee Ᏹ
    ),
}
