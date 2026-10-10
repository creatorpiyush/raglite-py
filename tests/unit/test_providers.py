"""
Provider wiring: default models, missing extras, and the google-genai client.
"""
import sys
import types
from unittest.mock import MagicMock

import pytest

from raglite.embeddings.models import DEFAULT_EMBEDDING_MODELS
from raglite.embeddings.remote import RemoteEmbedder
from raglite.errors import ConfigError
from raglite.llm.answer import _generate, _stream
from raglite.llm.factory import ResolvedLLM, create_llm
from raglite.llm.models import DEFAULT_LLM_MODELS
from raglite.types import EmbeddingProviderConfig

# Model IDs the providers have shut down; a default must never be one of these.
RETIRED = [
    "claude-3-5-sonnet-20241022",
    "gemini-2.0-flash",
    "text-embedding-004",
    "command-r-plus",
    "llama-3.3-70b-versatile",
    "grok-2-latest",
]


def test_defaults_use_no_retired_models():
    defaults = [*DEFAULT_LLM_MODELS.values(), *DEFAULT_EMBEDDING_MODELS.values()]
    for model in RETIRED:
        assert model not in defaults


def test_missing_llm_extra_raises_config_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(ConfigError, match=r"raglite-toolkit\[anthropic\]"):
        create_llm({"provider": "anthropic", "apiKey": "k"})


def test_missing_embedding_extra_is_not_wrapped(monkeypatch):
    monkeypatch.setitem(sys.modules, "voyageai", None)
    embedder = RemoteEmbedder.create(EmbeddingProviderConfig(provider="voyage", apiKey="k"))
    with pytest.raises(ConfigError, match=r"raglite-toolkit\[voyage\]"):
        embedder.embed_documents(["hi"])


def _google_llm(client):
    return ResolvedLLM("google", "gemini-3.8-flash", client, temperature=0.0, max_tokens=50)


def test_google_generate_uses_genai_client():
    client = MagicMock()
    resp = client.models.generate_content.return_value
    resp.text = "answer"
    resp.usage_metadata.prompt_token_count = 3
    resp.usage_metadata.candidates_token_count = 4
    resp.usage_metadata.total_token_count = 7

    out = _generate(_google_llm(client), "sys", "user")

    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "gemini-3.8-flash"
    assert kwargs["contents"] == "user"
    assert kwargs["config"] == {
        "system_instruction": "sys",
        "temperature": 0.0,
        "max_output_tokens": 50,
    }
    assert out["text"] == "answer"
    assert out["usage"]["totalTokens"] == 7


def test_google_stream_yields_chunk_text():
    client = MagicMock()
    client.models.generate_content_stream.return_value = [
        types.SimpleNamespace(text="Hel"),
        types.SimpleNamespace(text=None),
        types.SimpleNamespace(text="lo"),
    ]
    assert list(_stream(_google_llm(client), "sys", "user")) == ["Hel", "lo"]


def test_google_embeddings_use_genai_client(monkeypatch):
    client = MagicMock()
    client.models.embed_content.return_value = types.SimpleNamespace(
        embeddings=[types.SimpleNamespace(values=[3.0, 4.0]), types.SimpleNamespace(values=[0.0, 2.0])]
    )
    genai = types.SimpleNamespace(Client=MagicMock(return_value=client))
    google = types.ModuleType("google")
    google.genai = genai
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.genai", genai)

    embedder = RemoteEmbedder.create(EmbeddingProviderConfig(provider="google", apiKey="k"))
    vectors = embedder.embed_documents(["a", "b"])

    genai.Client.assert_called_once_with(api_key="k", http_options=None)
    kwargs = client.models.embed_content.call_args.kwargs
    assert kwargs == {"model": "gemini-embedding-2", "contents": ["a", "b"]}
    assert vectors == [[0.6, 0.8], [0.0, 1.0]]


def _anthropic_llm(client, temperature=None):
    return ResolvedLLM("anthropic", "claude-sonnet-5-5", client, temperature=temperature)


def test_anthropic_defaults_send_no_temperature(monkeypatch):
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=MagicMock()))
    assert create_llm({"provider": "anthropic", "apiKey": "k"}).temperature is None
    assert create_llm({"provider": "openai", "apiKey": "k"}).temperature == 0.0


def test_anthropic_generate_skips_thinking_and_omits_unset_temperature():
    client = MagicMock()
    resp = client.messages.create.return_value
    resp.content = [
        types.SimpleNamespace(type="thinking", thinking=""),
        types.SimpleNamespace(type="text", text="The answer."),
    ]
    resp.usage.input_tokens, resp.usage.output_tokens = 5, 6

    out = _generate(_anthropic_llm(client), "sys", "user")

    kwargs = client.messages.create.call_args.kwargs
    assert "temperature" not in kwargs and "extra_body" not in kwargs
    assert kwargs["max_tokens"] == 16000
    assert out["text"] == "The answer."


def test_anthropic_configured_temperature_goes_in_extra_body():
    client = MagicMock()
    client.messages.create.return_value.content = []
    _generate(_anthropic_llm(client, temperature=0.2), "sys", "user")
    assert client.messages.create.call_args.kwargs["extra_body"] == {"temperature": 0.2}


def _cohere_llm(client):
    return ResolvedLLM("cohere", "command-a-03-2025", client, temperature=0.0)


def test_cohere_generate_uses_v2_chat():
    client = MagicMock()
    resp = client.chat.return_value
    resp.message.content = [types.SimpleNamespace(type="text", text="Hi.")]
    resp.usage.tokens = types.SimpleNamespace(input_tokens=3.0, output_tokens=2.0)
    resp.finish_reason = "COMPLETE"

    out = _generate(_cohere_llm(client), "sys", "user")

    kwargs = client.chat.call_args.kwargs
    assert kwargs["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "user"},
    ]
    assert out["text"] == "Hi."
    assert out["usage"] == {"promptTokens": 3, "completionTokens": 2, "totalTokens": 5}
    assert out["finishReason"] == "COMPLETE"


def test_cohere_stream_yields_content_deltas():
    def delta(text):
        content = types.SimpleNamespace(text=text)
        return types.SimpleNamespace(
            type="content-delta",
            delta=types.SimpleNamespace(message=types.SimpleNamespace(content=content)),
        )

    client = MagicMock()
    client.chat_stream.return_value = [
        types.SimpleNamespace(type="message-start", delta=None),
        delta("Hel"),
        delta("lo"),
        types.SimpleNamespace(type="message-end", delta=None),
    ]
    assert list(_stream(_cohere_llm(client), "sys", "user")) == ["Hel", "lo"]


def test_cohere_embeddings_use_v2_float_embeddings(monkeypatch):
    client = MagicMock()
    client.embed.return_value.embeddings.float_ = [[3.0, 4.0]]
    cohere = types.SimpleNamespace(ClientV2=MagicMock(return_value=client))
    monkeypatch.setitem(sys.modules, "cohere", cohere)

    embedder = RemoteEmbedder.create(EmbeddingProviderConfig(provider="cohere", apiKey="k"))

    assert embedder.embed_query("q") == [0.6, 0.8]
    kwargs = client.embed.call_args.kwargs
    assert kwargs["input_type"] == "search_query"
    assert kwargs["embedding_types"] == ["float"]
