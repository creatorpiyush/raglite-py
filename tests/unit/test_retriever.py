"""
Unit tests for Retriever — mirrors tests/unit/retriever.test.ts
All provider calls are mocked so tests run offline.
"""
import math
from unittest.mock import MagicMock

from raglite.retrieval.retriever import Retriever
from raglite.types import ChunkMetadata
from raglite.vectordb.base import VectorSearchHit


def unit_vec(vec):
    s = sum(v * v for v in vec)
    n = math.sqrt(s)
    return [v / n for v in vec]


def make_hit(id_, text, score):
    return VectorSearchHit(
        id=id_,
        text=text,
        metadata=ChunkMetadata(source="doc.txt", chunk=1, totalChunks=3),
        score=score,
        distance=1 - score,
    )


class TestRetriever:
    def test_returns_results_above_threshold(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = unit_vec([1, 0, 0])

        store = MagicMock()
        store.search.return_value = [
            make_hit("a", "high score", 0.9),
            make_hit("b", "low score", 0.1),
        ]

        retriever = Retriever(embedder, store)
        results = retriever.retrieve("test query", top_k=5, score_threshold=0.5)

        assert len(results) == 1
        assert results[0].id == "a"

    def test_returns_all_results_with_zero_threshold(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = unit_vec([1, 0, 0])

        store = MagicMock()
        store.search.return_value = [
            make_hit("a", "first", 0.9),
            make_hit("b", "second", 0.5),
            make_hit("c", "third", 0.1),
        ]

        retriever = Retriever(embedder, store)
        results = retriever.retrieve("test query", top_k=5, score_threshold=0.0)
        assert len(results) == 3

    def test_calls_embed_query_with_the_correct_text(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = unit_vec([1, 0])
        store = MagicMock()
        store.search.return_value = []

        retriever = Retriever(embedder, store)
        retriever.retrieve("my search query", top_k=3)
        embedder.embed_query.assert_called_once_with("my search query")
