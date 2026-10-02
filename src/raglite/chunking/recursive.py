import re
from typing import List

from ..errors import ChunkingError
from .base import BaseChunker

# The exact set JavaScript's \s matches. Python's \s differs (it includes
# \x1c-\x1f and \x85 but not \ufeff), which would make the two SDKs chunk the
# same text differently.
_JS_WHITESPACE = re.compile(
    "[\t\n\v\f\r \u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]+"
)


class RecursiveChunker(BaseChunker):
    def split(self, text: str) -> List[str]:
        if not _JS_WHITESPACE.sub("", text):
            return []
        if self.overlap >= self.chunk_size:
            raise ChunkingError(
                f"overlap ({self.overlap}) must be smaller than chunkSize ({self.chunk_size})"
            )

        words = [w for w in _JS_WHITESPACE.split(text) if w]
        if len(words) <= self.chunk_size:
            return [" ".join(words)]

        step = self.chunk_size - self.overlap
        chunks: List[str] = []

        start = 0
        while start < len(words):
            end = start + self.chunk_size
            slice_words = words[start:end]
            if not slice_words:
                break
            chunks.append(" ".join(slice_words))
            if end >= len(words):
                break
            start += step

        return chunks
