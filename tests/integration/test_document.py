"""
Integration tests for Document build/cache lifecycle — mirrors tests/integration/document.test.ts
All embedding calls are mocked so tests run offline.
"""
import os
import shutil
import tempfile
from unittest.mock import patch

import pytest

from raglite.core.document import Document
from raglite.errors import FileNotIndexedError, LoaderError


def unit_vec_384():
    """Return a fake 384-dim unit vector (all-MiniLM-L6-v2 dimensions)."""
    v = [1.0] + [0.0] * 383
    return v


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="raglite_integ_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def sample_txt(tmp_dir):
    path = os.path.join(tmp_dir, "sample.txt")
    content = " ".join(f"word{i}" for i in range(200))
    with open(path, "w") as f:
        f.write(content)
    return path


def mock_embed_documents(texts):
    return [unit_vec_384() for _ in texts]


def mock_embed_query(text):
    return unit_vec_384()


class TestDocumentBuild:
    def test_raises_when_file_does_not_exist(self, tmp_dir):
        doc = Document(
            os.path.join(tmp_dir, "nonexistent.txt"),
            {"storeDir": tmp_dir},
        )
        with pytest.raises(LoaderError, match="File does not exist"):
            doc.build()

    def test_build_indexes_the_document(self, tmp_dir, sample_txt):
        with patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_documents",
            side_effect=mock_embed_documents,
        ), patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_query",
            side_effect=mock_embed_query,
        ):
            doc = Document(sample_txt, {"storeDir": tmp_dir})
            result = doc.build()
            assert result["cached"] is False
            assert result["chunkCount"] > 0
            assert result["embeddingProvider"] == "local"
            assert doc.chunk_count == result["chunkCount"]

    def test_build_reuses_cached_index(self, tmp_dir, sample_txt):
        with patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_documents",
            side_effect=mock_embed_documents,
        ), patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_query",
            side_effect=mock_embed_query,
        ):
            doc1 = Document(sample_txt, {"storeDir": tmp_dir})
            result1 = doc1.build()
            assert result1["cached"] is False

            doc2 = Document(sample_txt, {"storeDir": tmp_dir})
            result2 = doc2.build()
            assert result2["cached"] is True
            assert result2["chunkCount"] == result1["chunkCount"]

    def test_rebuild_forces_fresh_index(self, tmp_dir, sample_txt):
        with patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_documents",
            side_effect=mock_embed_documents,
        ), patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_query",
            side_effect=mock_embed_query,
        ):
            doc1 = Document(sample_txt, {"storeDir": tmp_dir})
            doc1.build()

            doc2 = Document(sample_txt, {"storeDir": tmp_dir})
            result2 = doc2.build({"rebuild": True})
            assert result2["cached"] is False

    def test_search_raises_before_build(self, tmp_dir, sample_txt):
        doc = Document(sample_txt, {"storeDir": tmp_dir})
        with pytest.raises(FileNotIndexedError):
            doc.search("query")

    def test_search_returns_results_after_build(self, tmp_dir, sample_txt):
        with patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_documents",
            side_effect=mock_embed_documents,
        ), patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_query",
            side_effect=mock_embed_query,
        ):
            doc = Document(sample_txt, {"storeDir": tmp_dir})
            doc.build()
            results = doc.search("word5")
            assert isinstance(results, list)
            # All results should have non-negative scores
            for r in results:
                assert r.score >= 0.0

    def test_chunk_count_matches_build_result(self, tmp_dir, sample_txt):
        with patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_documents",
            side_effect=mock_embed_documents,
        ), patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_query",
            side_effect=mock_embed_query,
        ):
            doc = Document(sample_txt, {"storeDir": tmp_dir})
            result = doc.build()
            assert doc.chunk_count == result["chunkCount"]
