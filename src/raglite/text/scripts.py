"""Script tables shared by the chunker and the keyword tokenizer.

These are the same explicit code point ranges as the TypeScript SDK
(``src/text/scripts.ts``): Python's ``re`` and ``unicodedata`` have no script
property, and both SDKs must split text identically.
"""

import unicodedata
from typing import Sequence, Tuple

# Scripts written without spaces between words: Thai, Lao, Myanmar, Khmer, Han, kana.
_UNSPACED_RANGES: Sequence[Tuple[int, int]] = (
    (0x0E00, 0x0EFF),  # Thai, Lao
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
    (0x2E80, 0x2FDF),  # CJK radicals, Kangxi radicals
    (0x3005, 0x3007),  # 々 〆 〇
    (0x3021, 0x3029),  # Hangzhou numerals
    (0x3038, 0x303B),
    (0x3040, 0x30FF),  # Hiragana, Katakana
    (0x31F0, 0x31FF),  # Katakana phonetic extensions
    (0x3400, 0x4DBF),  # CJK extension A
    (0x4E00, 0x9FFF),  # CJK unified ideographs
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0xFF66, 0xFF9F),  # Halfwidth katakana
    (0x20000, 0x323AF),  # CJK extensions B-H, compatibility supplement
)

# Hangul separates words with spaces, but its syllables still need bigrams for keyword search.
_HANGUL_RANGES: Sequence[Tuple[int, int]] = (
    (0x1100, 0x11FF),
    (0x3130, 0x318F),
    (0xA960, 0xA97F),
    (0xAC00, 0xD7FF),
)


def _in_ranges(cp: int, ranges: Sequence[Tuple[int, int]]) -> bool:
    for lo, hi in ranges:
        if cp < lo:
            return False
        if cp <= hi:
            return True
    return False


def is_unspaced_char(cp: int) -> bool:
    """True for a character of a script written without spaces between words."""
    return _in_ranges(cp, _UNSPACED_RANGES)


def is_bigram_char(cp: int) -> bool:
    """True for a character the keyword tokenizer indexes as character bigrams."""
    return _in_ranges(cp, _UNSPACED_RANGES) or _in_ranges(cp, _HANGUL_RANGES)


def is_mark(ch: str) -> bool:
    """True for a combining mark (Unicode category M), which belongs to the character before it."""
    return unicodedata.category(ch).startswith("M")


def has_unspaced_text(text: str) -> bool:
    """True if ``text`` contains any character of a script written without spaces."""
    return any(is_unspaced_char(ord(ch)) for ch in text)
