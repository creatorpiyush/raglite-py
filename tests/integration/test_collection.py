import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from raglite.api.server import build_app
from raglite.core.collection import DocumentCollection

SAMPLE1 = "Refund Policy. Refunds are issued within 30 days of purchase."
SAMPLE2 = "Shipping Info. Delivery takes 2 business days worldwide."
SAMPLE3 = json.dumps({"product": "Acme Widget", "warranty": "1 year"})


def unit_vec_384():
    return [1.0] + [0.0] * 383


def mock_embed_documents(texts):
    return [unit_vec_384() for _ in texts]


def mock_embed_query(text):
    return unit_vec_384()


def test_collection_indexes_and_searches_directory():
    temp_dir = tempfile.mkdtemp()
    store_dir = tempfile.mkdtemp()
    try:
        (Path(temp_dir) / "policy.txt").write_text(SAMPLE1, encoding="utf-8")
        (Path(temp_dir) / "shipping.md").write_text(SAMPLE2, encoding="utf-8")
        (Path(temp_dir) / "product.json").write_text(SAMPLE3, encoding="utf-8")

        with patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_documents",
            side_effect=mock_embed_documents,
        ), patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_query",
            side_effect=mock_embed_query,
        ):
            collection = DocumentCollection(
                temp_dir,
                options={
                    "storeDir": store_dir,
                    "embeddings": {"provider": "local"},
                    "logLevel": "silent",
                },
            )

            res = collection.build()
            assert res.totalDocuments == 3
            assert res.totalChunks == 3
            assert len(res.errors) == 0

            hits = collection.search("Refund policy", top_k=3)
            assert len(hits) > 0
            assert any("Refunds are issued" in h.text for h in hits)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(store_dir, ignore_errors=True)


def test_collection_fastapi_server_endpoints():
    temp_dir = tempfile.mkdtemp()
    store_dir = tempfile.mkdtemp()
    try:
        (Path(temp_dir) / "policy.txt").write_text(SAMPLE1, encoding="utf-8")

        with patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_documents",
            side_effect=mock_embed_documents,
        ), patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_query",
            side_effect=mock_embed_query,
        ):
            collection = DocumentCollection(
                temp_dir,
                options={
                    "storeDir": store_dir,
                    "embeddings": {"provider": "local"},
                    "logLevel": "silent",
                },
            )
            collection.build()

            app = build_app(collection, {})
            client = TestClient(app)

            health_res = client.get("/health")
            assert health_res.status_code == 200
            assert health_res.json()["status"] == "ok"
            assert health_res.json()["chunks"] == 1

            search_res = client.post("/search", json={"query": "refund policy"})
            assert search_res.status_code == 200
            assert len(search_res.json()["results"]) == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        shutil.rmtree(store_dir, ignore_errors=True)
