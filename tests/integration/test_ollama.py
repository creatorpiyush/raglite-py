"""
Integration tests for the Ollama provider — live end-to-end tests.

Requires:
  - Ollama running locally (ollama serve)
  - Models pulled:
      ollama pull embeddinggemma
      ollama pull gemma3

Run all Ollama tests:
    pytest -m ollama -v

Skip them in CI / offline:
    pytest -m "not ollama"
"""
import os
import shutil
import tempfile
import pytest
from raglite.core.document import Document
from raglite.errors import FileNotIndexedError

# ── Shared config ─────────────────────────────────────────────────────────────
from tests.conftest import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL, OLLAMA_EMBED_MODEL

EMBED_CONFIG = {
    "provider": "ollama",
    "model":    OLLAMA_EMBED_MODEL,
    "baseURL":  OLLAMA_BASE_URL,
}

LLM_CONFIG = {
    "provider": "ollama",
    "model":    OLLAMA_LLM_MODEL,
    "baseURL":  OLLAMA_BASE_URL,
}

POLICY_TEXT = """
Acme Corp Refund Policy

We stand behind every purchase. If you are not satisfied for any reason,
you may request a full refund within 30 days of purchase. Refunds are
processed within 5 business days.

Shipping Policy

Standard shipping takes 3 to 5 business days within the continental United
States. Express shipping (1 to 2 business days) is available for an
additional fee. We do not ship to P.O. boxes.

Warranty

All Acme products carry a limited one-year warranty covering manufacturing
defects. The warranty does not cover damage caused by misuse, accidents,
or unauthorized modifications. To claim warranty service, contact
support@acme.example with proof of purchase.

Privacy

We do not sell your personal data. Your information is used only to
fulfill your orders and to provide customer support. You may request
deletion of your account at any time.
""".strip()


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_store(tmp_path_factory):
    """Module-scoped temp dir reused across all Ollama tests (avoids re-embedding)."""
    d = tmp_path_factory.mktemp("raglite_ollama_")
    yield str(d)
    shutil.rmtree(str(d), ignore_errors=True)


@pytest.fixture(scope="module")
def policy_file(tmp_store):
    """Write the policy document once for the whole module."""
    path = os.path.join(tmp_store, "policy.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(POLICY_TEXT)
    return path


@pytest.fixture(scope="module")
def built_doc(ollama_available, policy_file, tmp_store):
    """
    Module-scoped fixture: builds the index once with Ollama embeddings and
    reuses it across all tests in this module (saves embedding round-trips).
    """
    doc = Document(policy_file, {
        "embeddings": EMBED_CONFIG,
        "llm":        LLM_CONFIG,
        "chunkSize":  100,
        "overlap":    10,
        "storeDir":   tmp_store,
    })
    result = doc.build()
    assert result["embeddingProvider"] == "ollama"
    assert result["chunkCount"] > 0
    return doc


# ── Test Classes ──────────────────────────────────────────────────────────────

@pytest.mark.ollama
class TestOllamaEmbeddings:
    """Tests for the Ollama embedding provider."""

    def test_build_creates_index_with_correct_provider(self, built_doc):
        meta = built_doc.store.read_index_metadata()
        assert meta.embeddingProvider == "ollama"
        assert meta.embeddingModel == OLLAMA_EMBED_MODEL

    def test_embedding_dimensions_are_non_zero(self, built_doc):
        meta = built_doc.store.read_index_metadata()
        assert meta.embeddingDimensions > 0

    def test_chunk_count_matches_stored_metadata(self, built_doc):
        meta = built_doc.store.read_index_metadata()
        assert built_doc.chunk_count == meta.chunkCount

    def test_direct_embed_query_returns_correct_dims(self, ollama_available):
        from raglite.embeddings.factory import create_embedder
        embedder = create_embedder(EMBED_CONFIG)
        vec = embedder.embed_query("refund policy")
        assert isinstance(vec, list)
        assert len(vec) > 0
        # Should be L2-normalised (magnitude ≈ 1.0)
        import math
        magnitude = math.sqrt(sum(v * v for v in vec))
        assert abs(magnitude - 1.0) < 1e-3

    def test_embed_documents_returns_one_vector_per_text(self, ollama_available):
        from raglite.embeddings.factory import create_embedder
        embedder = create_embedder(EMBED_CONFIG)
        texts = ["refund policy", "shipping time", "warranty coverage"]
        vecs = embedder.embed_documents(texts)
        assert len(vecs) == len(texts)
        for v in vecs:
            assert len(v) > 0


@pytest.mark.ollama
class TestOllamaIndexCache:
    """Tests for cache reuse with Ollama indexes."""

    def test_second_build_returns_cached_true(self, ollama_available, policy_file, tmp_store):
        doc = Document(policy_file, {
            "embeddings": EMBED_CONFIG,
            "llm":        LLM_CONFIG,
            "chunkSize":  100,
            "overlap":    10,
            "storeDir":   tmp_store,
        })
        result = doc.build()
        assert result["cached"] is True

    def test_rebuild_true_forces_fresh_index(self, ollama_available, policy_file, tmp_store):
        doc = Document(policy_file, {
            "embeddings": EMBED_CONFIG,
            "llm":        LLM_CONFIG,
            "chunkSize":  100,
            "overlap":    10,
            "storeDir":   tmp_store,
        })
        result = doc.build({"rebuild": True})
        assert result["cached"] is False
        assert result["chunkCount"] > 0

    def test_changed_chunk_size_invalidates_cache(self, ollama_available, policy_file, tmp_path):
        store = str(tmp_path)
        doc1 = Document(policy_file, {
            "embeddings": EMBED_CONFIG,
            "chunkSize": 80,
            "overlap": 8,
            "storeDir": store,
        })
        r1 = doc1.build()
        assert r1["cached"] is False

        doc2 = Document(policy_file, {
            "embeddings": EMBED_CONFIG,
            "chunkSize": 120,   # different → must rebuild
            "overlap": 8,
            "storeDir": store,
        })
        r2 = doc2.build()
        assert r2["cached"] is False


@pytest.mark.ollama
class TestOllamaSearch:
    """Tests for semantic search using Ollama embeddings."""

    def test_search_returns_non_empty_results(self, built_doc):
        hits = built_doc.search("refund policy", top_k=3)
        assert len(hits) > 0

    def test_search_results_have_valid_scores(self, built_doc):
        hits = built_doc.search("shipping delivery time", top_k=3)
        for hit in hits:
            assert 0.0 <= hit.score <= 1.0
            assert hit.distance >= 0.0
            assert abs((hit.score + hit.distance) - 1.0) < 1e-4

    def test_search_ordered_by_descending_score(self, built_doc):
        hits = built_doc.search("warranty manufacturing defects", top_k=5)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_result_count(self, built_doc):
        hits_1 = built_doc.search("refund", top_k=1)
        hits_3 = built_doc.search("refund", top_k=3)
        assert len(hits_1) == 1
        assert len(hits_3) <= 3

    def test_score_threshold_filters_low_scores(self, built_doc):
        hits = built_doc.search("completely unrelated gibberish xyzzy", top_k=5, score_threshold=0.99)
        # With a very high threshold, we expect 0 results
        assert len(hits) == 0

    def test_search_result_metadata_is_correct(self, built_doc):
        hits = built_doc.search("refund policy", top_k=1)
        assert len(hits) == 1
        meta = hits[0].metadata
        assert meta.source == "policy.txt"
        assert meta.chunk >= 1
        assert meta.totalChunks >= 1

    def test_refund_query_returns_refund_chunk(self, built_doc):
        hits = built_doc.search("refund", top_k=2)
        top = hits[0]
        assert "refund" in top.text.lower() or "purchase" in top.text.lower()

    def test_privacy_query_returns_relevant_chunk(self, built_doc):
        hits = built_doc.search("privacy personal data", top_k=2)
        combined = " ".join(h.text.lower() for h in hits)
        assert "data" in combined or "privacy" in combined or "personal" in combined


@pytest.mark.ollama
class TestOllamaAsk:
    """Tests for RAG Q&A using the Ollama LLM."""

    def test_ask_returns_non_empty_answer(self, built_doc):
        answer = built_doc.ask("What is the refund policy?")
        assert answer.text.strip() != ""

    def test_ask_provider_is_ollama(self, built_doc):
        answer = built_doc.ask("What is the refund policy?")
        assert answer.provider == "ollama"

    def test_ask_model_matches_config(self, built_doc):
        answer = built_doc.ask("What is the refund policy?")
        assert answer.model == OLLAMA_LLM_MODEL

    def test_ask_usage_contains_token_counts(self, built_doc):
        answer = built_doc.ask("How long does standard shipping take?")
        assert answer.usage is not None
        assert answer.usage.get("totalTokens", 0) > 0

    def test_ask_answer_is_grounded_in_context(self, built_doc):
        answer = built_doc.ask("How long does standard shipping take?")
        # The document says "3 to 5 business days"
        text = answer.text.lower()
        assert any(kw in text for kw in ["3", "three", "5", "five", "business day", "shipping"])

    def test_ask_warranty_question(self, built_doc):
        answer = built_doc.ask("What does the warranty cover?")
        text = answer.text.lower()
        assert any(kw in text for kw in ["warranty", "defect", "manufacturing", "one-year", "year"])

    def test_ask_raises_without_llm_config(self, ollama_available, policy_file, tmp_path):
        doc = Document(policy_file, {
            "embeddings": EMBED_CONFIG,
            "storeDir": str(tmp_path),
        })
        doc.build()
        from raglite.errors import RagLiteError
        with pytest.raises(RagLiteError, match="No LLM provider"):
            doc.ask("What is the refund policy?")

    def test_ask_with_llm_override_at_call_time(self, built_doc):
        """LLM config can be passed at ask-time, overriding the document default."""
        answer = built_doc.ask(
            "What is the refund policy?",
            {"llm": LLM_CONFIG, "topK": 2},
        )
        assert answer.text.strip() != ""
        assert answer.provider == "ollama"


@pytest.mark.ollama
class TestOllamaStreaming:
    """Tests for streaming answers using the Ollama LLM."""

    def test_stream_yields_string_chunks(self, built_doc):
        chunks = list(built_doc.ask_stream("Summarize this document in one sentence."))
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, str)

    def test_stream_concatenates_to_non_empty_answer(self, built_doc):
        text = "".join(built_doc.ask_stream("What is the refund policy?"))
        assert text.strip() != ""

    def test_stream_and_ask_are_consistent(self, built_doc):
        """Both streaming and non-streaming paths should return plausible answers."""
        q = "How long does standard shipping take?"
        streamed = "".join(built_doc.ask_stream(q)).lower()
        asked    = built_doc.ask(q).text.lower()
        # Both should mention shipping-related terms
        for text in (streamed, asked):
            assert any(kw in text for kw in ["3", "five", "5", "business", "shipping", "day"])

    def test_stream_raises_without_llm_config(self, ollama_available, policy_file, tmp_path):
        doc = Document(policy_file, {
            "embeddings": EMBED_CONFIG,
            "storeDir": str(tmp_path),
        })
        doc.build()
        from raglite.errors import RagLiteError
        with pytest.raises(RagLiteError, match="No LLM provider"):
            list(doc.ask_stream("What is the refund policy?"))

    def test_stream_alias_ask_stream_equals_ask_Stream(self, built_doc):
        """askStream() TypeScript alias should work identically to ask_stream()."""
        chunks = list(built_doc.askStream("What is the warranty period?"))
        assert len(chunks) > 0
