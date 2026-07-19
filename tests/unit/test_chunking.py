"""
Unit tests for RecursiveChunker — mirrors tests/unit/chunking.test.ts
"""
import pytest

from raglite.chunking.recursive import RecursiveChunker
from raglite.errors import ChunkingError


class TestRecursiveChunker:
    def test_returns_empty_for_empty_input(self):
        chunker = RecursiveChunker(10, 2)
        assert chunker.split("") == []
        assert chunker.split("   \n  \t") == []

    def test_single_chunk_when_shorter_than_chunk_size(self):
        chunker = RecursiveChunker(50, 5)
        chunks = chunker.split("one two three four")
        assert len(chunks) == 1
        assert chunks[0] == "one two three four"

    def test_splits_long_text_into_overlapping_chunks(self):
        words = " ".join(f"w{i}" for i in range(50))
        chunker = RecursiveChunker(10, 2)
        chunks = chunker.split(words)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.split()) <= 10

        # Overlap: last 2 words of chunk[0] == first 2 words of chunk[1]
        first_tail = chunks[0].split()[-2:]
        second_head = chunks[1].split()[:2]
        assert first_tail == second_head

    def test_covers_every_word_at_least_once(self):
        words = [f"word{i}" for i in range(25)]
        chunker = RecursiveChunker(6, 2)
        chunks = chunker.split(" ".join(words))
        covered = set()
        for c in chunks:
            for w in c.split():
                covered.add(w)
        for w in words:
            assert w in covered

    def test_raises_when_overlap_ge_chunk_size(self):
        chunker = RecursiveChunker(5, 5)
        words = " ".join(f"w{i}" for i in range(20))
        with pytest.raises(ChunkingError):
            chunker.split(words)

    def test_normalizes_whitespace_in_output(self):
        chunker = RecursiveChunker(100, 10)
        chunks = chunker.split("hello\n\n\n   world\t\tfoo")
        assert chunks[0] == "hello world foo"
