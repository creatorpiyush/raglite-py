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


class TestDocumentFromText:
    def test_indexes_inline_text_and_reuses_it_by_id(self, tmp_dir):
        text = " ".join(f"word{i}" for i in range(60))
        opts = {"storeDir": tmp_dir, "chunkSize": 20, "overlap": 5}
        LocalEmbedder = __import__(
            "raglite.embeddings.local", fromlist=["LocalEmbedder"]
        ).LocalEmbedder
        with patch.object(
            LocalEmbedder, "embed_documents", side_effect=mock_embed_documents
        ), patch.object(LocalEmbedder, "embed_query", side_effect=mock_embed_query):
            doc = Document.from_text("faq-42", text, opts)
            assert doc.build()["cached"] is False
            assert doc.search("word1", {"topK": 1})[0].metadata.source == "faq-42"

            assert Document.from_text("faq-42", text, opts).build()["cached"] is True
            changed = Document.from_text("faq-42", text + " updated", opts)
            assert changed.build()["cached"] is False

    def test_namespace_matches_typescript(self, tmp_dir):
        # TS: namespaceFromPath(`text:${id}`)
        from raglite.utils.hash import namespace_from_path

        doc = Document.from_text("faq-42", "x", {"storeDir": tmp_dir})
        assert doc.namespace == namespace_from_path("text:faq-42")

    def test_rejects_empty_text(self, tmp_dir):
        with pytest.raises(LoaderError):
            Document.from_text("empty", "", {"storeDir": tmp_dir}).build()


class TestRetiredEmbeddingModel:
    """Indexes built with an embedding model the provider has shut down."""

    def _retired_index(self, tmp_dir, sample_txt):
        import json

        opts = {"storeDir": tmp_dir, "chunkSize": 20, "overlap": 5, "logLevel": "silent"}
        doc = Document(sample_txt, opts)
        doc.build()
        # Pretend it was built with Google's text-embedding-004, which Google has shut down.
        meta_path = os.path.join(tmp_dir, doc.namespace, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta.update(embeddingProvider="google", embeddingModel="text-embedding-004")
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        return opts

    def _patched(self):
        LocalEmbedder = __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder
        return (
            patch.object(LocalEmbedder, "embed_documents", side_effect=mock_embed_documents),
            patch.object(LocalEmbedder, "embed_query", side_effect=mock_embed_query),
        )

    def test_search_without_build_explains_how_to_fix_it(self, tmp_dir, sample_txt):
        from raglite.errors import RagLiteError

        docs, query = self._patched()
        with docs, query:
            opts = self._retired_index(tmp_dir, sample_txt)
            with pytest.raises(RagLiteError, match=r"shut down.*Call build\(\)"):
                Document(sample_txt, opts).search("word1")

    def test_build_re_embeds_with_the_current_default(self, tmp_dir, sample_txt):
        docs, query = self._patched()
        with docs as embed_documents, query:
            opts = self._retired_index(tmp_dir, sample_txt)
            embed_documents.reset_mock()
            # Default provider is local; the retired index's provider is google, so pass google
            # without a model and stub its embedder.
            from raglite.embeddings.remote import RemoteEmbedder

            with patch.object(RemoteEmbedder, "embed_documents", side_effect=mock_embed_documents) as remote:
                result = Document(sample_txt, opts).build()
            assert result["cached"] is False
            assert result["embeddingModel"] == "gemini-embedding-2"
            assert remote.call_count == 1

    def test_build_keeps_the_index_when_that_model_is_requested(self, tmp_dir, sample_txt):
        docs, query = self._patched()
        with docs, query:
            opts = self._retired_index(tmp_dir, sample_txt)
            embeddings = {"provider": "google", "model": "text-embedding-004", "apiKey": "k"}
            result = Document(sample_txt, {**opts, "embeddings": embeddings}).build()
            assert result["cached"] is True


class TestChunkLocations:
    def _chunks(self, doc):
        return doc.store.list_chunks()

    def test_records_the_page_of_each_pdf_chunk(self, tmp_dir):
        from tests.pdf_helper import make_pdf

        path = os.path.join(tmp_dir, "guide.pdf")
        with open(path, "wb") as f:
            f.write(make_pdf(["alpha beta gamma delta", "epsilon zeta", "eta theta iota"]))
        LocalEmbedder = __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder
        with patch.object(LocalEmbedder, "embed_documents", side_effect=mock_embed_documents):
            doc = Document(path, {"storeDir": tmp_dir, "chunkSize": 4, "overlap": 0, "logLevel": "silent"})
            doc.build()
        got = [(c.text, getattr(c.metadata, "page", None), getattr(c.metadata, "pageEnd", None)) for c in self._chunks(doc)]
        assert got == [
            ("alpha beta gamma delta", 1, None),
            ("epsilon zeta eta theta", 2, 3),
            ("iota", 3, None),
        ]

    def test_records_the_heading_path_of_each_markdown_chunk(self, tmp_dir):
        path = os.path.join(tmp_dir, "guide.md")
        with open(path, "w") as f:
            f.write("# Policies\nIntro.\n## Refunds\nRefunds take thirty days.")
        LocalEmbedder = __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder
        with patch.object(LocalEmbedder, "embed_documents", side_effect=mock_embed_documents):
            doc = Document(path, {"storeDir": tmp_dir, "chunkSize": 3, "overlap": 0, "logLevel": "silent"})
            doc.build()
        sections = [getattr(c.metadata, "section", None) for c in self._chunks(doc)]
        assert sections == ["Policies", "Policies > Refunds", "Policies > Refunds"]

    def test_stored_metadata_has_no_null_location_keys(self, tmp_dir, sample_txt):
        import json

        LocalEmbedder = __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder
        with patch.object(LocalEmbedder, "embed_documents", side_effect=mock_embed_documents):
            doc = Document(sample_txt, {"storeDir": tmp_dir, "logLevel": "silent"})
            doc.build()
        with open(os.path.join(tmp_dir, doc.namespace, "chunks.json")) as f:
            stored = json.load(f)
        assert not {"page", "pageEnd", "section"} & set(stored[0]["metadata"])


class TestBuildProgressAndCancel:
    MANY = " ".join(f"word{i}" for i in range(200))

    def _embedder(self):
        LocalEmbedder = __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder
        return patch.object(LocalEmbedder, "embed_documents", side_effect=mock_embed_documents)

    def test_reports_progress_after_each_batch(self, tmp_dir):
        path = os.path.join(tmp_dir, "big.txt")
        with open(path, "w") as f:
            f.write(self.MANY)
        progress = []
        with self._embedder():
            Document(path, {"storeDir": tmp_dir, "chunkSize": 2, "overlap": 0, "logLevel": "silent"}).build(
                on_progress=progress.append
            )
        # 100 chunks in batches of 64.
        assert progress == [
            {"source": "big.txt", "embedded": 64, "total": 100},
            {"source": "big.txt", "embedded": 100, "total": 100},
        ]

    def test_cancel_stops_the_build_and_keeps_the_previous_index(self, tmp_dir):
        import threading

        from raglite.errors import BuildCancelledError

        path = os.path.join(tmp_dir, "big.txt")
        with open(path, "w") as f:
            f.write(self.MANY)
        opts = {"storeDir": tmp_dir, "chunkSize": 2, "overlap": 0, "logLevel": "silent"}
        with self._embedder() as embed:
            Document(path, opts).build()
            embed.reset_mock()
            cancel = threading.Event()
            with pytest.raises(BuildCancelledError):
                Document(path, opts).build(rebuild=True, cancel=cancel, on_progress=lambda _: cancel.set())
            assert embed.call_count == 1  # stopped after the first batch
        reopened = Document(path, opts)
        assert reopened.search("word5", {"mode": "keyword", "topK": 1})
        assert reopened.chunk_count == 100
