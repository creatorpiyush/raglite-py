import re
from typing import List, Tuple

from ..errors import ChunkingError
from ..text.scripts import has_unspaced_text, is_mark, is_unspaced_char
from .base import BaseChunker

# The exact set JavaScript's \s matches. Python's \s differs (it includes
# \x1c-\x1f and \x85 but not ﻿), which would make the two SDKs chunk the
# same text differently.
_JS_WHITESPACE = re.compile(
    "[\t\n\v\f\r    -     　﻿]+"
)

# One chunking unit: (text, spaced). ``spaced`` is False when the unit
# continues the previous unit's word, so no space goes between them.
_Unit = Tuple[str, bool]


class RecursiveChunker(BaseChunker):
    """Word-based recursive chunker with overlap.

    Scripts written without spaces (Chinese, Japanese, Thai, ...) have no word
    boundaries to split on, so each of their characters counts as one unit.
    Without this a whole unspaced document would be a single "word" and
    therefore a single chunk of any length.
    """

    def split(self, text: str) -> List[str]:
        if not _JS_WHITESPACE.sub("", text):
            return []
        if self.overlap >= self.chunk_size:
            raise ChunkingError(
                f"overlap ({self.overlap}) must be smaller than chunkSize ({self.chunk_size})"
            )

        words = [w for w in _JS_WHITESPACE.split(text) if w]
        if has_unspaced_text(text):
            units = [u for w in words for u in _word_units(w)]
        else:
            units = [(w, True) for w in words]
        if len(units) <= self.chunk_size:
            return [_render(units)]

        step = self.chunk_size - self.overlap
        chunks: List[str] = []

        start = 0
        while start < len(units):
            end = start + self.chunk_size
            slice_units = units[start:end]
            if not slice_units:
                break
            chunks.append(_render(slice_units))
            if end >= len(units):
                break
            start += step

        return chunks


def _word_units(word: str) -> List[_Unit]:
    """Split a word into units: each unspaced-script character (with its
    combining marks) is a unit, and every run of other characters is a unit."""
    units: List[List] = []
    run = ""
    for ch in word:
        if is_mark(ch) and run == "" and units:
            units[-1][0] += ch
        elif is_mark(ch):
            run += ch
        elif is_unspaced_char(ord(ch)):
            if run:
                units.append([run, not units])
            run = ""
            units.append([ch, not units])
        else:
            run += ch
    if run:
        units.append([run, not units])
    return [(text, spaced) for text, spaced in units]


def _render(units: List[_Unit]) -> str:
    out = ""
    for i, (text, spaced) in enumerate(units):
        if i > 0 and spaced:
            out += " "
        out += text
    return out
