"""
Keyword and hybrid retrieval, the keyword index upgrade path, and the index
format 1 -> 2 upgrade. Mirrors raglite/tests/integration/hybrid.test.ts.
"""
import hashlib
import json
import math
import os
from typing import List, Optional
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from raglite.api.server import build_app
from raglite.constants import INDEX_FORMAT_VERSION
from raglite.core.collection import DocumentCollection
from raglite.core.document import Document
from raglite.errors import ConfigError
from raglite.types import IndexMetadata, StoredChunk
from raglite.vectordb.base import VectorSearchHit, VectorStore


class MockEmbedder:
    """Deterministic hash-based embedder, like tests/helpers/mock-embedder.ts."""

    provider = "local"

    def __init__(self, model: str = "mock-model", dimensions: int = 16):
        self.model = model
        self.dimensions = dimensions
        self.embed_documents_calls = 0
        self.embed_query_calls = 0

    def _vector(self, text: str) -> List[float]:
        digest = hashlib.sha256(text.lower().encode()).digest()
        vec = [(digest[i % len(digest)] / 255) * 2 - 1 for i in range(self.dimensions)]
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):
        self.embed_documents_calls += 1
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        self.embed_query_calls += 1
        return self._vector(text)


EMBEDDERS: List[MockEmbedder] = []


@pytest.fixture(autouse=True)
def mock_embedder():
    EMBEDDERS.clear()

    def create(config):
        embedder = MockEmbedder(getattr(config, "model", None) or "mock-model")
        EMBEDDERS.append(embedder)
        return embedder

    with patch("raglite.core.document.create_embedder", side_effect=create):
        yield


def embed_calls() -> int:
    return sum(e.embed_documents_calls for e in EMBEDDERS)


def silent(tmp_dir, **extra):
    return {"storeDir": os.path.join(tmp_dir, ".raglite"), "logLevel": "silent", **extra}


def write(tmp_dir, name, text):
    path = os.path.join(tmp_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def runbook() -> str:
    lines = [
        f"Section {i + 1} describes routine maintenance of the billing service." for i in range(20)
    ]
    lines[13] = "If checkout fails with ERR_4021 the payment gateway timed out, so retry later."
    return "\n".join(lines)


def build_runbook(tmp_dir, **extra) -> Document:
    doc = Document(write(tmp_dir, "runbook.txt", runbook()), silent(tmp_dir, **extra))
    doc.build(chunk_size=11, overlap=0)
    return doc


class TestHybridSearch:
    def test_finds_exact_error_code_vector_search_misses(self, tmp_dir):
        doc = build_runbook(tmp_dir)
        assert "ERR_4021" not in doc.search("ERR_4021", top_k=1)[0].text

        for mode in ("keyword", "hybrid"):
            top = doc.search("ERR_4021", top_k=1, mode=mode)[0]
            assert "ERR_4021" in top.text
            assert top.scores.keyword > 0
            assert 0 < top.score <= 1
            assert top.distance == pytest.approx(1 - top.score)

    def test_matches_typescript_results(self, tmp_dir):
        # Same corpus, chunking and embedder as hybrid.test.ts.
        doc = build_runbook(tmp_dir)
        top = doc.search("ERR_4021", {"topK": 1, "mode": "keyword"})[0]
        assert top.text.startswith("ERR_4021 the payment gateway timed out")
        assert top.scores.keyword == pytest.approx(6.921877475107216, abs=1e-9)

    def test_vector_mode_output_is_unchanged(self, tmp_dir):
        doc = build_runbook(tmp_dir)
        results = doc.search("billing", top_k=3)
        assert len(results) == 3
        assert results[0].scores is None
        assert "scores" not in results[0].model_dump(by_alias=True)

    def test_configured_default_mode_skips_query_embedding(self, tmp_dir):
        doc = build_runbook(tmp_dir, retrieval={"mode": "keyword"})
        top = doc.search("ERR_4021", top_k=1)[0]
        assert "ERR_4021" in top.text
        assert EMBEDDERS[-1].embed_query_calls == 0

    def test_score_threshold_filters_vector_list_only(self, tmp_dir):
        doc = build_runbook(tmp_dir)
        results = doc.search("ERR_4021", top_k=5, mode="hybrid", score_threshold=2)
        assert len(results) == 1
        assert results[0].scores.vector is None
        assert "ERR_4021" in results[0].text

    def test_rejects_invalid_options(self, tmp_dir):
        doc = build_runbook(tmp_dir)
        with pytest.raises(ConfigError, match="retrieval mode"):
            doc.search("x", mode="fuzzy")
        with pytest.raises(ConfigError, match="rrfK"):
            doc.search("x", mode="hybrid", hybrid={"rrfK": 0})
        with pytest.raises(ConfigError, match="weights"):
            doc.search("x", mode="hybrid", hybrid={"weights": {"vector": 0, "keyword": 0}})

    def test_writes_keyword_index_next_to_vector_index(self, tmp_dir):
        doc = build_runbook(tmp_dir)
        path = os.path.join(tmp_dir, ".raglite", doc.store_namespace, "keyword.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["tokenizer"] == "raglite-v1"
        assert len(data["chunks"]) == doc.chunk_count


class TestKeywordIndexUpgrade:
    def test_rebuilds_missing_keyword_index_without_reembedding(self, tmp_dir):
        built = build_runbook(tmp_dir)
        os.remove(os.path.join(tmp_dir, ".raglite", built.store_namespace, "keyword.json"))
        before = embed_calls()

        doc = Document(os.path.join(tmp_dir, "runbook.txt"), silent(tmp_dir))
        assert doc.build(chunk_size=11, overlap=0)["cached"] is True
        top = doc.search("ERR_4021", top_k=1, mode="hybrid")[0]
        assert "ERR_4021" in top.text
        assert embed_calls() == before
        assert os.path.exists(os.path.join(tmp_dir, ".raglite", doc.store_namespace, "keyword.json"))

    def test_reads_keyword_index_written_by_typescript_layout(self, tmp_dir):
        doc = build_runbook(tmp_dir)
        path = os.path.join(tmp_dir, ".raglite", doc.store_namespace, "keyword.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        assert set(data) == {"format", "tokenizer", "index", "chunks", "lengths", "postings"}
        assert set(data["index"]) == {"sourceHash", "createdAt", "chunkCount"}

    def test_falls_back_to_vector_search_when_store_cannot_list(self, tmp_dir, capsys):
        store = BareStore("bare")
        opts = {"storeDir": os.path.join(tmp_dir, "elsewhere"), "vectorStore": store, "logLevel": "info"}
        built = Document(write(tmp_dir, "a.txt", "alpha beta gamma"), opts)
        built.build()
        os.remove(os.path.join(tmp_dir, "elsewhere", built.store_namespace, "keyword.json"))

        fresh = Document(os.path.join(tmp_dir, "a.txt"), opts)
        results = fresh.search("alpha", mode="keyword", score_threshold=-1)
        assert len(results) == 1
        assert results[0].scores is None
        fresh.search("alpha", mode="keyword", score_threshold=-1)
        assert capsys.readouterr().err.count("No keyword index") == 1

    def test_prefers_native_keyword_search(self, tmp_dir):
        store = BareStore("native")
        store.native = lambda query, top_k: [
            VectorSearchHit(
                id="native_1",
                text=f"native hit for {query}",
                metadata={"source": "n.txt", "chunk": 1, "totalChunks": 1},
                score=3,
                distance=0,
            )
        ]
        doc = Document(write(tmp_dir, "n.txt", "alpha beta"), silent(tmp_dir, vectorStore=store))
        doc.build()
        top = doc.search("anything", mode="keyword")[0]
        assert top.id == "native_1"
        assert top.scores.keyword == 3


class TestIndexFormatUpgrade:
    def _build_then_set_format(self, tmp_dir, text, fmt):
        path = write(tmp_dir, "policy.txt", text)
        doc = Document(path, silent(tmp_dir))
        doc.build()
        meta_path = os.path.join(tmp_dir, ".raglite", doc.store_namespace, "metadata.json")
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        meta["formatVersion"] = fmt
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return Document(path, silent(tmp_dir)).build(), meta_path

    def test_upgrades_format_1_in_place_when_chunking_is_unchanged(self, tmp_dir):
        result, meta_path = self._build_then_set_format(
            tmp_dir, "Refunds are issued within 30 days.", 1
        )
        assert result["cached"] is True
        with open(meta_path, encoding="utf-8") as f:
            assert json.load(f)["formatVersion"] == INDEX_FORMAT_VERSION

    def test_rebuilds_format_1_index_of_unspaced_text(self, tmp_dir):
        result, _ = self._build_then_set_format(tmp_dir, "返金は三十日以内に行われます。", 1)
        assert result["cached"] is False


class TestCollectionHybrid:
    def _collection(self, tmp_dir):
        write(tmp_dir, "a.txt", "Refunds are issued within 30 days of purchase.")
        write(tmp_dir, "b.txt", "Error ERR_4021 means the payment gateway timed out.")
        write(tmp_dir, "c.txt", "The billing page lists every payment and refund.")
        collection = DocumentCollection(
            [os.path.join(tmp_dir, f) for f in ("a.txt", "b.txt", "c.txt")], silent(tmp_dir)
        )
        collection.build()
        return collection

    def test_fuses_across_documents_and_survives_one_failure(self, tmp_dir):
        collection = self._collection(tmp_dir)
        top = collection.search("ERR_4021 payment", top_k=3, mode="hybrid")[0]
        assert top.metadata.source == "b.txt"
        assert top.scores.fused > 0

        broken = next(d for d in collection.get_documents() if d.file_path.endswith("c.txt"))
        with patch.object(broken, "retrieve_candidates", side_effect=RuntimeError("store offline")):
            results = collection.search("payment", top_k=3, mode="keyword")
        assert [r.metadata.source for r in results] == ["b.txt"]

    def test_throws_when_every_document_fails(self, tmp_dir):
        collection = self._collection(tmp_dir)
        for doc in collection.get_documents():
            doc.retrieve_candidates = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down"))
        with pytest.raises(RuntimeError, match="down"):
            collection.search("alpha", mode="hybrid")

    def test_serves_a_collection_over_http(self, tmp_dir):
        collection = self._collection(tmp_dir)
        client = TestClient(build_app(collection, {}))
        resp = client.post("/search", json={"query": "ERR_4021", "topK": 1, "mode": "keyword"})
        assert resp.status_code == 200
        assert resp.json()["results"][0]["metadata"]["source"] == "b.txt"


class TestApiMode:
    def test_search_accepts_mode_and_rejects_unknown(self, tmp_dir):
        doc = build_runbook(tmp_dir)
        client = TestClient(build_app(doc, {}))
        ok = client.post("/search", json={"query": "ERR_4021", "topK": 1, "mode": "keyword"})
        assert ok.status_code == 200
        result = ok.json()["results"][0]
        assert "ERR_4021" in result["text"]
        assert set(result["scores"]) == {"keyword", "fused"}

        vector = client.post("/search", json={"query": "billing", "topK": 1}).json()["results"][0]
        assert "scores" not in vector

        assert client.post("/search", json={"query": "x", "mode": "fuzzy"}).status_code in (400, 422)


class BareStore(VectorStore):
    """A minimal custom store with no list_chunks, like a remote store would be."""

    def __init__(self, namespace: str):
        self._namespace = namespace
        self._chunks: List[StoredChunk] = []
        self._metadata: Optional[IndexMetadata] = None
        self.native = None

    @property
    def namespace(self) -> str:
        return self._namespace

    def load(self) -> None:
        pass

    def reset(self) -> None:
        self._chunks = []

    def add(self, chunks: List[StoredChunk]) -> None:
        self._chunks.extend(chunks)

    def search(self, embedding: List[float], top_k: int) -> List[VectorSearchHit]:
        return [
            VectorSearchHit(
                id=c.id,
                text=c.text,
                metadata=c.metadata,
                score=sum(a * b for a, b in zip(c.embedding, embedding)),
                distance=0,
            )
            for c in self._chunks[:top_k]
        ]

    def count(self) -> int:
        return len(self._chunks)

    def save_index_metadata(self, metadata: IndexMetadata) -> None:
        self._metadata = metadata

    def read_index_metadata(self) -> Optional[IndexMetadata]:
        return self._metadata

    def keyword_search(self, query, top_k):
        return self.native(query, top_k) if self.native else None
