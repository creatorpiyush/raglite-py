"""as_tool(): a function-calling tool over a document. Mirrors tests/unit/search_tool.test.ts."""

import json
from unittest.mock import MagicMock

from raglite.tools import search_tool
from raglite.types import ChunkMetadata, SearchResult

HIT = SearchResult(
    id="a_000001",
    text="Refunds take thirty days.",
    metadata=ChunkMetadata(source="policy.pdf", chunk=3, totalChunks=9, page=12, section="Refunds"),
    score=0.87654,
    distance=0.12346,
)


def test_searches_with_the_configured_options_and_returns_passages():
    target = MagicMock()
    target.search.return_value = [HIT]
    tool = search_tool(target, top_k=2, mode="hybrid")

    out = json.loads(tool(query="refund"))

    target.search.assert_called_once_with("refund", {"topK": 2, "mode": "hybrid"})
    assert out == [
        {"source": "policy.pdf", "chunk": 3, "page": 12, "section": "Refunds", "score": 0.8765, "text": HIT.text}
    ]


def test_tool_definitions_for_openai_and_anthropic():
    tool = search_tool(MagicMock(), name="search_policies", description="Company policies.")
    assert tool.openai_tool["type"] == "function"
    assert tool.openai_tool["function"]["name"] == "search_policies"
    assert tool.openai_tool["function"]["parameters"]["required"] == ["query"]
    assert tool.anthropic_tool == {
        "name": "search_policies",
        "description": "Company policies.",
        "input_schema": tool.parameters,
    }


def test_document_as_tool_searches_the_document(tmp_path):
    from unittest.mock import patch

    from raglite import Document

    LocalEmbedder = __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder
    vec = [1.0] + [0.0] * 383
    with patch.object(LocalEmbedder, "embed_documents", side_effect=lambda t: [vec for _ in t]), patch.object(
        LocalEmbedder, "embed_query", return_value=vec
    ):
        doc = Document.from_text("faq", "Refunds take thirty days.", {"storeDir": str(tmp_path), "logLevel": "silent"})
        doc.build()
        passages = json.loads(doc.as_tool()(query="refund"))
    assert passages[0]["source"] == "faq"
