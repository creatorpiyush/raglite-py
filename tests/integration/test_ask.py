"""
Integration tests for ask / stream path — mirrors tests/integration/ask.test.ts
All LLM and embedding calls are mocked so tests run offline.
"""
import os
import shutil
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from raglite.core.document import Document
from raglite.errors import RagLiteError


def unit_vec_384():
    return [1.0] + [0.0] * 383


def mock_embed_documents(texts):
    return [unit_vec_384() for _ in texts]


def mock_embed_query(text):
    return unit_vec_384()


MOCK_ANSWER_DICT = {
    "text": "This is the mocked answer.",
    "provider": "openai",
    "model": "gpt-4o-mini",
    "usage": {"promptTokens": 10, "completionTokens": 20, "totalTokens": 30},
    "finishReason": "stop",
}


@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp(prefix="raglite_ask_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def indexed_doc(tmp_dir):
    path = os.path.join(tmp_dir, "doc.txt")
    content = " ".join(f"word{i}" for i in range(150))
    with open(path, "w") as f:
        f.write(content)
    with patch.object(
        __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
        "embed_documents",
        side_effect=mock_embed_documents,
    ), patch.object(
        __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
        "embed_query",
        side_effect=mock_embed_query,
    ):
        doc = Document(path, {"storeDir": tmp_dir})
        doc.build()
    return doc


class TestAsk:
    def test_ask_raises_without_llm_config(self, indexed_doc):
        with patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_query",
            side_effect=mock_embed_query,
        ):
            with pytest.raises(RagLiteError, match="No LLM provider"):
                indexed_doc.ask("What is this?")

    def test_ask_returns_answer_result(self, indexed_doc):
        llm_config = {"provider": "openai", "apiKey": "sk-test"}
        with patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_query",
            side_effect=mock_embed_query,
        ), patch("raglite.llm.answer._generate", return_value=MOCK_ANSWER_DICT):
            result = indexed_doc.ask("What is this?", {"llm": llm_config})
            assert result.text == "This is the mocked answer."
            assert result.provider == "openai"

    def test_ask_stream_yields_text_deltas(self, indexed_doc):
        llm_config = {"provider": "openai", "apiKey": "sk-test"}
        chunks = ["Hello", " ", "World"]

        with patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_query",
            side_effect=mock_embed_query,
        ), patch("raglite.llm.answer._stream", return_value=iter(chunks)):
            collected = list(indexed_doc.ask_stream("What is this?", {"llm": llm_config}))
            assert collected == chunks

    def test_ask_stream_raises_without_llm(self, indexed_doc):
        with patch.object(
            __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder,
            "embed_query",
            side_effect=mock_embed_query,
        ):
            with pytest.raises(RagLiteError, match="No LLM provider"):
                list(indexed_doc.ask_stream("What is this?"))
