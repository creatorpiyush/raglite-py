"""
Integration tests for FastAPI REST endpoints — mirrors tests/integration/api.test.ts
Uses FastAPI TestClient for in-memory HTTP request/response validation.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from raglite.api.server import build_app
from raglite.types import SearchResult, ChunkMetadata, AnswerResult


def make_result(id_, text):
    return SearchResult(
        id=id_,
        text=text,
        metadata=ChunkMetadata(source="doc.txt", chunk=1, totalChunks=1),
        score=0.9,
        distance=0.1,
    )


@pytest.fixture
def mock_doc():
    doc = MagicMock()
    doc.chunk_count = 10
    doc.store_namespace = "ns-abc"
    doc.resolved_config = MagicMock()
    doc.resolved_config.chunkSize = 500
    doc.resolved_config.overlap = 50
    doc.resolved_config.topK = 5
    doc.resolved_config.embeddings.provider = "local"
    doc.resolved_config.embeddings.model = "m"
    doc.resolved_config.llm = MagicMock()
    doc.resolved_config.llm.provider = "openai"
    return doc


class TestApiServer:
    def test_health_endpoint_no_auth_required(self, mock_doc):
        # Even with token auth, health endpoint should be public
        app = build_app(mock_doc, {"bearerToken": "secret-token"})
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["chunks"] == 10
        assert data["namespace"] == "ns-abc"

    def test_unauthorized_if_token_missing_on_info(self, mock_doc):
        app = build_app(mock_doc, {"bearerToken": "secret-token"})
        client = TestClient(app)
        resp = client.get("/info")
        assert resp.status_code == 401
        assert resp.json() == {"detail": "Unauthorized"}

    def test_authorized_with_bearer_token(self, mock_doc):
        app = build_app(mock_doc, {"bearerToken": "secret-token"})
        client = TestClient(app)
        resp = client.get("/info", headers={"Authorization": "Bearer secret-token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunkSize"] == 500

    def test_info_endpoint_returns_document_details(self, mock_doc):
        app = build_app(mock_doc, {})
        client = TestClient(app)
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunkSize"] == 500
        assert data["overlap"] == 50
        assert data["chunks"] == 10
        assert data["embeddings"]["provider"] == "local"

    def test_search_returns_hits(self, mock_doc):
        mock_doc.search.return_value = [make_result("1", "Hello World")]
        app = build_app(mock_doc, {})
        client = TestClient(app)

        resp = client.post("/search", json={"query": "test query", "topK": 3})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) == 1
        assert results[0]["id"] == "1"
        assert results[0]["text"] == "Hello World"
        mock_doc.search.assert_called_once_with("test query", top_k=3, score_threshold=None)

    def test_ask_returns_answer(self, mock_doc):
        llm = {"provider": "openai", "apiKey": "sk-test"}
        mock_doc.ask.return_value = AnswerResult(
            text="AI generated answer.",
            provider="openai",
            model="gpt-4o-mini",
            usage={"promptTokens": 5, "completionTokens": 5, "totalTokens": 10},
            finishReason="stop"
        )
        app = build_app(mock_doc, {"llm": llm})
        client = TestClient(app)

        resp = client.post("/ask", json={"question": "What is the answer?", "topK": 2})
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "AI generated answer."
        assert data["provider"] == "openai"

    def test_ask_disabled_when_llm_missing(self, mock_doc):
        app = build_app(mock_doc, {})
        client = TestClient(app)
        resp = client.post("/ask", json={"question": "What is the answer?"})
        assert resp.status_code == 503
        assert "No LLM provider configured" in resp.json()["message"]
