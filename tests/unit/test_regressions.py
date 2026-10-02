"""
Regression tests for URL indexing and shared vector stores across documents.
"""
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from raglite.core.collection import DocumentCollection
from raglite.core.document import Document
from raglite.embeddings.local import LocalEmbedder
from raglite.errors import ConfigError
from raglite.types import ChunkMetadata, EmbeddingProviderConfig, StoredChunk
from raglite.vectordb import MemoryVectorStore
from raglite.vectordb.qdrant import QdrantVectorStore


def unit_vec_384():
    return [1.0] + [0.0] * 383


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="raglite_regress_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_document_builds_from_url(tmp_dir):
    with patch(
        "raglite.loaders.web.WebLoader.load", return_value="Refunds are issued within 30 days."
    ), patch.object(
        LocalEmbedder,
        "embed_documents",
        side_effect=lambda texts: [unit_vec_384() for _ in texts],
    ):
        doc = Document("https://example.com/policy", {"storeDir": tmp_dir, "logLevel": "silent"})
        result = doc.build()

    assert result["chunkCount"] == 1
    assert result["cached"] is False


def test_collection_rejects_shared_store_instance(tmp_dir):
    (Path(tmp_dir) / "a.txt").write_text("first document", encoding="utf-8")
    (Path(tmp_dir) / "b.txt").write_text("second document", encoding="utf-8")
    store = MemoryVectorStore(str(Path(tmp_dir) / ".raglite"), "shared")

    collection = DocumentCollection(tmp_dir, {"vectorStore": store, "logLevel": "silent"})
    with pytest.raises(ConfigError):
        collection.build()


def test_qdrant_shared_collection_is_scoped_to_namespace():
    calls = []

    def fake_request(self, method, path, body=None):
        calls.append((self.namespace, method, path, body))
        return {"result": []}

    with patch.object(QdrantVectorStore, "_request", fake_request):
        a = QdrantVectorStore("http://qdrant:6333", "ns-a", collection_name="shared")
        b = QdrantVectorStore("http://qdrant:6333", "ns-b", collection_name="shared")

        a.reset()
        a.add(
            [
                StoredChunk(
                    id="ns-a_000001",
                    text="hello",
                    embedding=[1.0, 0.0, 0.0],
                    metadata=ChunkMetadata(source="a.txt", chunk=1, totalChunks=1),
                )
            ]
        )
        a.search([1.0, 0.0, 0.0], 3)
        a.read_index_metadata()
        b.read_index_metadata()

    assert not any(method == "DELETE" for _, method, _, _ in calls)

    delete = next(c for c in calls if c[2].endswith("/points/delete?wait=true"))
    assert delete[3]["filter"]["must"][0]["match"]["value"] == "ns-a"

    upsert = next(c for c in calls if c[1] == "PUT" and c[2].endswith("/points"))
    assert upsert[3]["points"][0]["payload"]["namespace"] == "ns-a"

    search = next(c for c in calls if c[2].endswith("/points/search"))
    assert search[3]["filter"]["must"][0]["match"]["value"] == "ns-a"

    meta_ids = [c[3]["ids"][0] for c in calls if c[1] == "POST" and c[2] == "/collections/shared/points"]
    assert len(meta_ids) == 2
    assert meta_ids[0] != meta_ids[1]


def test_qdrant_dedicated_collection_keeps_existing_behaviour():
    calls = []

    def fake_request(self, method, path, body=None):
        calls.append((method, path, body))
        return {"result": []}

    with patch.object(QdrantVectorStore, "_request", fake_request):
        store = QdrantVectorStore("http://qdrant:6333", "ns-a")
        store.reset()
        store.search([1.0, 0.0, 0.0], 3)

    assert ("DELETE", "/collections/raglite_ns-a", None) in calls
    search = next(c for c in calls if c[1].endswith("/points/search"))
    assert "filter" not in search[2]


class FakeEmbedder:
    def __init__(self, config):
        self.provider = config.provider
        self.model = config.model or "fake-model"
        self.dimensions = 384

    def embed_documents(self, texts):
        return [unit_vec_384() for _ in texts]

    def embed_query(self, text):
        return unit_vec_384()


@pytest.fixture
def embedder_configs():
    configs = []

    def fake_create(config):
        configs.append(config)
        return FakeEmbedder(config)

    with patch("raglite.core.document.create_embedder", side_effect=fake_create):
        yield configs


def test_url_reindexes_when_page_content_changes(tmp_dir, embedder_configs):
    page = {"text": "Refunds are issued within 30 days."}
    opts = {"storeDir": tmp_dir, "logLevel": "silent"}
    url = "https://example.com/policy"

    with patch("raglite.loaders.web.WebLoader.load", side_effect=lambda: page["text"]):
        assert Document(url, opts).build()["cached"] is False
        assert Document(url, opts).build()["cached"] is True
        page["text"] = "Refunds are issued within 60 days."
        assert Document(url, opts).build()["cached"] is False


def test_query_embedder_uses_index_provider_after_reload(tmp_dir, embedder_configs):
    path = Path(tmp_dir) / "policy.txt"
    path.write_text("Refunds are issued within 30 days.", encoding="utf-8")
    opts = {"storeDir": tmp_dir, "logLevel": "silent"}

    Document(str(path), opts).build(
        embeddings=EmbeddingProviderConfig(provider="openai", model="text-embedding-3-small")
    )
    embedder_configs.clear()
    Document(str(path), opts).search("refund")

    assert len(embedder_configs) == 1
    assert embedder_configs[0].provider == "openai"
    assert embedder_configs[0].model == "text-embedding-3-small"
    assert embedder_configs[0].apiKey is None


def test_query_embedder_reuses_credentials_for_same_provider(tmp_dir, embedder_configs):
    path = Path(tmp_dir) / "policy.txt"
    path.write_text("Refunds are issued within 30 days.", encoding="utf-8")
    opts = {
        "storeDir": tmp_dir,
        "logLevel": "silent",
        "embeddings": {"provider": "openai", "apiKey": "sk-test"},
    }

    Document(str(path), opts).build()
    embedder_configs.clear()
    Document(str(path), opts).search("refund")

    assert embedder_configs[0].apiKey == "sk-test"


def _two_doc_collection(tmp_dir):
    (Path(tmp_dir) / "policy.txt").write_text("Refunds within 30 days.", encoding="utf-8")
    (Path(tmp_dir) / "shipping.txt").write_text("Delivery takes 2 days.", encoding="utf-8")
    collection = DocumentCollection(
        tmp_dir, {"storeDir": str(Path(tmp_dir) / ".raglite"), "logLevel": "silent"}
    )
    collection.build()
    return collection


def test_collection_search_survives_one_failing_document(tmp_dir, embedder_configs):
    collection = _two_doc_collection(tmp_dir)
    broken = next(iter(collection.documents.values()))
    with patch.object(broken, "search", side_effect=RuntimeError("boom")):
        hits = collection.search("refund", top_k=5)
    assert len(hits) == 1


def test_collection_search_raises_when_every_document_fails(tmp_dir, embedder_configs):
    collection = _two_doc_collection(tmp_dir)
    docs = list(collection.documents.values())
    with patch.object(docs[0], "search", side_effect=RuntimeError("bad api key")), patch.object(
        docs[1], "search", side_effect=RuntimeError("bad api key")
    ):
        with pytest.raises(RuntimeError, match="bad api key"):
            collection.search("refund")
