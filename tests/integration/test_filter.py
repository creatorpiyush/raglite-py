"""
Metadata filters across stores, modes and collections. Mirrors
raglite/tests/integration/filter.test.ts.
"""
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from raglite.api.server import build_app
from raglite.core.collection import DocumentCollection
from raglite.core.document import Document
from raglite.errors import ConfigError
from raglite.types import ChunkMetadata, StoredChunk
from raglite.vectordb.memory import MemoryVectorStore
from raglite.vectordb.pinecone import PineconeVectorStore
from raglite.vectordb.qdrant import QdrantVectorStore

from .test_hybrid import mock_embedder, write  # noqa: F401  (autouse fixture)

MODES = ["vector", "keyword", "hybrid"]

# Three sections of eight words each, so chunkSize 8 gives one chunk per section.
HANDBOOK = "\n".join(
    [
        "# Intro",
        "Every refund request starts here today.",
        "# Billing",
        "A refund goes back to card.",
        "# Shipping",
        "Damaged parcels get a refund too.",
    ]
)


def silent(tmp_dir, **extra):
    # Mock embeddings can have negative cosine similarity; keep every hit.
    return {
        "storeDir": os.path.join(tmp_dir, ".raglite"),
        "logLevel": "silent",
        "scoreThreshold": -1,
        **extra,
    }


def handbook(tmp_dir, **options):
    doc = Document(write(tmp_dir, "handbook.md", HANDBOOK), silent(tmp_dir, **options))
    doc.build({"chunkSize": 8, "overlap": 0})
    return doc


def sections(hits):
    return sorted(h.metadata.section for h in hits)


@pytest.mark.parametrize("mode", MODES)
def test_filters_on_chunk_metadata(tmp_dir, mode):
    doc = handbook(tmp_dir)
    assert sections(doc.search("refund", mode=mode, top_k=10)) == ["Billing", "Intro", "Shipping"]
    hits = doc.search(
        "refund", mode=mode, top_k=10, filter={"section": {"in": ["Billing", "Shipping"]}}
    )
    assert sections(hits) == ["Billing", "Shipping"]


def test_applies_the_filter_before_the_top_k_cut(tmp_dir):
    doc = handbook(tmp_dir)
    for section in ["Intro", "Billing", "Shipping"]:
        assert sections(doc.search("refund", top_k=1, filter={"section": section})) == [section]


@pytest.mark.parametrize("mode", MODES)
def test_adds_custom_metadata_and_filters_on_it(tmp_dir, mode):
    doc = handbook(tmp_dir, metadata={"tenant": "acme", "tier": 2})
    hits = doc.search("refund", mode=mode, filter={"tenant": "acme", "tier": {"gte": 2}})
    assert hits
    for hit in hits:
        assert hit.metadata.tenant == "acme" and hit.metadata.tier == 2

    assert doc.search("refund", mode=mode, filter={"tenant": "globex"}) == []
    assert doc.search("refund", mode=mode, filter={"source": "other.md"}) == []
    hits = doc.search("refund", {"mode": mode, "filter": {"source": "handbook.md", "section": "Intro"}})
    assert len(hits) == 1


def test_drops_hits_from_a_store_that_ignores_the_filter(tmp_dir):
    class IgnoresFilter(MemoryVectorStore):
        def search(self, embedding, top_k):  # a store written before 2.2
            return super().search(embedding, top_k)

    store = IgnoresFilter(os.path.join(tmp_dir, "custom"), "ns")
    doc = handbook(tmp_dir, vectorStore=store)
    assert sections(doc.search("refund", top_k=10, filter={"section": "Billing"})) == ["Billing"]


def test_changing_custom_metadata_needs_no_rebuild(tmp_dir):
    handbook(tmp_dir, metadata={"tenant": "acme"})
    doc = Document(
        write(tmp_dir, "handbook.md", HANDBOOK), silent(tmp_dir, metadata={"tenant": "globex"})
    )
    assert doc.build({"chunkSize": 8, "overlap": 0})["cached"] is True
    assert doc.search("refund", filter={"tenant": "globex"}) != []


def test_rejects_invalid_filters_and_metadata(tmp_dir):
    doc = handbook(tmp_dir)
    with pytest.raises(ConfigError):
        doc.search("refund", filter={"page": {"ne": 3}})
    with pytest.raises(ConfigError):
        Document("x.md", {"metadata": {"page": 3}})
    with pytest.raises(ConfigError):
        Document("x.md", {"metadata": {"tags": ["a"]}})


@pytest.mark.parametrize("mode", MODES)
def test_collection_filters_by_source(tmp_dir, mode):
    os.makedirs(os.path.join(tmp_dir, "docs"))
    write(tmp_dir, "docs/a.md", "# A\nThe refund window is 30 days.")
    write(tmp_dir, "docs/b.md", "# B\nA refund needs a receipt.")
    collection = DocumentCollection(os.path.join(tmp_dir, "docs"), silent(tmp_dir))
    collection.build()

    all_hits = collection.search("refund", mode=mode, top_k=10)
    assert {h.metadata.source for h in all_hits} == {"a.md", "b.md"}
    hits = collection.search("refund", mode=mode, top_k=10, filter={"source": "b.md"})
    assert [h.metadata.source for h in hits] == ["b.md"]


def test_api_accepts_filter_and_rejects_invalid_ones(tmp_dir):
    client = TestClient(build_app(handbook(tmp_dir), {}))
    ok = client.post("/search", json={"query": "refund", "topK": 10, "filter": {"section": "Billing"}})
    assert ok.status_code == 200
    assert [r["metadata"]["section"] for r in ok.json()["results"]] == ["Billing"]

    bad = client.post("/search", json={"query": "refund", "filter": {"section": {"like": "B%"}}})
    assert bad.status_code == 400
    assert bad.json()["error"] == "ConfigError"


def test_qdrant_sends_filter_as_payload_conditions_alongside_namespace():
    bodies = []

    def fake_request(self, method, path, body=None):
        if path.endswith("/points/search"):
            bodies.append(body)
        return {"result": []}

    with patch.object(QdrantVectorStore, "_request", fake_request):
        store = QdrantVectorStore("http://qdrant:6333", "ns-a", collection_name="shared")
        store.search(
            [1.0, 0.0], 3, {"section": "Refunds", "chunk": {"in": [1, 2]}, "page": {"gte": 3, "lt": 5}}
        )
    assert bodies[0]["filter"] == {
        "must": [
            {"key": "namespace", "match": {"value": "ns-a"}},
            {"key": "metadata.section", "match": {"value": "Refunds"}},
            {"key": "metadata.chunk", "match": {"any": [1, 2]}},
            {"key": "metadata.page", "range": {"gte": 3, "lt": 5}},
        ]
    }


def test_pinecone_stores_and_returns_locations_and_sends_filter():
    calls = {}

    def fake_request(self, method, path, body=None):
        calls[path.split("?")[0]] = body
        if path.startswith("/query"):
            return {
                "matches": [
                    {
                        "id": "c1",
                        "score": 0.9,
                        "metadata": {
                            "text": "t",
                            "source": "a.pdf",
                            "chunk": 1.0,
                            "totalChunks": 2.0,
                            "page": 3.0,
                            "section": "Refunds",
                        },
                    }
                ]
            }
        return {}

    with patch.object(PineconeVectorStore, "_request", fake_request):
        store = PineconeVectorStore("https://idx.pinecone.io", "k", "ns")
        store.add(
            [
                StoredChunk(
                    id="c1",
                    text="t",
                    embedding=[1.0, 0.0],
                    metadata=ChunkMetadata(
                        source="a.pdf", chunk=1, totalChunks=2, page=3, section="Refunds"
                    ),
                ),
                StoredChunk(
                    id="c2",
                    text="u",
                    embedding=[0.0, 1.0],
                    metadata=ChunkMetadata(source="a.pdf", chunk=2, totalChunks=2),
                ),
            ]
        )
        hits = store.search([1.0, 0.0], 3, {"page": {"gte": 3}, "section": {"in": ["Refunds"]}})

    vectors = calls["/vectors/upsert"]["vectors"]
    assert vectors[0]["metadata"]["page"] == 3 and vectors[0]["metadata"]["section"] == "Refunds"
    assert "page" not in vectors[1]["metadata"]
    assert calls["/query"]["filter"] == {"page": {"$gte": 3}, "section": {"$in": ["Refunds"]}}
    assert hits[0].metadata.page == 3 and isinstance(hits[0].metadata.page, int)
    assert hits[0].metadata.section == "Refunds"
