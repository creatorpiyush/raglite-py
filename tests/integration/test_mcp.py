"""MCP server tools, called in-process. Mirrors tests/integration/mcp.test.ts."""

import json
from unittest.mock import patch

import pytest

from raglite.core.collection import DocumentCollection
from raglite.mcp_server import create_mcp_server

LocalEmbedder = __import__("raglite.embeddings.local", fromlist=["LocalEmbedder"]).LocalEmbedder
ANSWER = {"text": "Thirty days [1].", "provider": "openai", "model": "m", "usage": {}, "finishReason": "stop"}


@pytest.fixture(autouse=True)
def stub_models():
    vec = [1.0] + [0.0] * 383
    with patch.object(LocalEmbedder, "embed_documents", side_effect=lambda t: [vec for _ in t]), patch.object(
        LocalEmbedder, "embed_query", return_value=vec
    ), patch("raglite.llm.answer._generate", return_value=ANSWER):
        yield


def make_server(tmp_path, llm=None):
    docs = tmp_path / "docs"
    docs.mkdir(parents=True)
    (docs / "refunds.txt").write_text("The refund window is thirty days. ERR_4021 means the card was declined.")
    (docs / "shipping.md").write_text("# Shipping\n\nDelivery takes three to five business days.")
    options = {"storeDir": str(tmp_path / ".raglite"), "logLevel": "silent"}
    if llm:
        options["llm"] = llm
    collection = DocumentCollection(str(docs), options)
    collection.build()
    return create_mcp_server(collection)


def text_of(result):
    return result.content[0].text


async def test_offers_ask_only_when_an_llm_is_configured(tmp_path):
    without = make_server(tmp_path / "a")
    assert sorted(t.name for t in await without.list_tools()) == ["list_sources", "search"]
    with_llm = make_server(tmp_path / "b", {"provider": "openai", "apiKey": "sk-test"})
    assert sorted(t.name for t in await with_llm.list_tools()) == ["ask", "list_sources", "search"]


async def test_tool_inputs_match_the_typescript_sdk(tmp_path):
    server = make_server(tmp_path, {"provider": "openai", "apiKey": "sk-test"})
    tools = {t.name: t for t in await server.list_tools()}
    assert set(tools["search"].input_schema["properties"]) == {"query", "topK", "mode"}
    assert set(tools["ask"].input_schema["properties"]) == {"question", "topK", "mode"}


async def test_search_returns_passages_as_json(tmp_path):
    server = make_server(tmp_path)
    result = await server.call_tool("search", {"query": "ERR_4021", "mode": "keyword", "topK": 1})
    hits = json.loads(text_of(result))
    assert len(hits) == 1
    assert hits[0]["source"] == "refunds.txt" and hits[0]["chunk"] == 1
    assert "ERR_4021" in hits[0]["text"]


async def test_ask_answers_from_the_documents(tmp_path):
    server = make_server(tmp_path, {"provider": "openai", "apiKey": "sk-test"})
    result = await server.call_tool("ask", {"question": "Refund window?"})
    assert text_of(result) == "Thirty days [1]."


async def test_list_sources(tmp_path):
    server = make_server(tmp_path)
    sources = json.loads(text_of(await server.call_tool("list_sources", {})))
    assert sorted(s["source"] for s in sources) == ["refunds.txt", "shipping.md"]
    assert all(s["chunks"] == 1 for s in sources)


async def test_invalid_input_is_rejected(tmp_path):
    from mcp.server.mcpserver.exceptions import ToolError

    server = make_server(tmp_path)
    # In-process the error is raised; over the protocol the client gets an error result.
    with pytest.raises(ToolError, match="less than or equal to 50"):
        await server.call_tool("search", {"query": "x", "topK": 999})
