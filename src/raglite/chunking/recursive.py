import re
from typing import List
from .base import BaseChunker
from ..errors import ChunkingError


class RecursiveChunker(BaseChunker):
    def split(self, text: str) -> List[str]:
        if not text.strip():
            return []
        if self.overlap >= self.chunk_size:
            raise ChunkingError(
                f"overlap ({self.overlap}) must be smaller than chunkSize ({self.chunk_size})"
            )

        words = [w for w in re.split(r"\s+", text) if w]
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
