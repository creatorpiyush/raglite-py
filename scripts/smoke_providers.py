"""Live smoke test: one embedding call and one answer per provider, with the default models.

    python scripts/smoke_providers.py

A provider runs only when its API key is set (Ollama: OLLAMA_BASE_URL). Exits 1 if any
provider that ran failed, so a retired or mistyped default model fails before a release.
The TypeScript SDK's scripts/smoke-providers.ts reads the same variables.
"""

import os
import sys
import time
from functools import partial
from typing import Callable, Optional

from raglite.embeddings import create_embedder
from raglite.embeddings.models import DEFAULT_EMBEDDING_MODELS
from raglite.llm import generate_answer
from raglite.llm.models import DEFAULT_LLM_MODELS
from raglite.types import (
    ChunkMetadata,
    EmbeddingProviderConfig,
    EmbeddingProviderName,
    LLMProviderName,
    SearchResult,
)

KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "cohere": "COHERE_API_KEY",
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
    "voyage": "VOYAGE_API_KEY",
    "ollama": "OLLAMA_BASE_URL",
}
LLM_PROVIDERS: list[LLMProviderName] = ["openai", "anthropic", "google", "mistral", "cohere", "groq", "xai", "ollama"]
EMBEDDING_PROVIDERS: list[EmbeddingProviderName] = ["openai", "google", "mistral", "cohere", "voyage", "ollama"]

CONTEXT = [
    SearchResult(
        id="smoke_000001",
        text="The refund window is 30 days from the date of purchase.",
        metadata=ChunkMetadata(source="smoke", chunk=1, totalChunks=1),
        score=1.0,
        distance=0.0,
    )
]


def credentials(provider: str) -> Optional[dict]:
    """apiKey, or baseURL for Ollama; None when the variable is unset."""
    value = os.environ.get(KEY_VARS[provider])
    if not value:
        return None
    return {"baseURL": value} if provider == "ollama" else {"apiKey": value}


ran = 0
failures = 0


def check(kind: str, provider: str, model: str, run: Callable[[], str]) -> None:
    global ran, failures
    label = f"{kind:<9} {provider:<9} {model}"
    if credentials(provider) is None:
        print(f"SKIP  {label}  ({KEY_VARS[provider]} not set)")
        return
    ran += 1
    start = time.monotonic()
    try:
        detail = run()
        print(f"PASS  {label}  {time.monotonic() - start:.1f}s  {detail}")
    except Exception as err:
        failures += 1
        print(f"FAIL  {label}  {describe(err.__cause__ or err)[:200]}")


def describe(error: BaseException) -> str:
    """One line; SDK HTTP errors as status + body (Cohere's message starts with the headers)."""
    status, body = getattr(error, "status_code", None), getattr(error, "body", None)
    if status is not None and body is not None:
        return f"HTTP {status}: {body}"
    lines = str(error).splitlines()
    return lines[0] if lines else type(error).__name__


def embed(provider: EmbeddingProviderName) -> str:
    config = EmbeddingProviderConfig(provider=provider, **(credentials(provider) or {}))
    vector = create_embedder(config).embed_query("refund policy")
    if not vector:
        raise RuntimeError("empty embedding")
    return f"{len(vector)} dims"


def answer(provider: LLMProviderName) -> str:
    result = generate_answer(
        {
            "llm": {"provider": provider, **(credentials(provider) or {})},
            "question": "How long is the refund window? Answer in a few words.",
            "context": CONTEXT,
        }
    )
    text = (result.text or "").strip()
    if not text:
        raise RuntimeError(f"empty answer (finish reason: {result.finishReason})")
    return repr(text[:60])


for e in EMBEDDING_PROVIDERS:
    check("embedding", e, DEFAULT_EMBEDDING_MODELS[e], partial(embed, e))
for m in LLM_PROVIDERS:
    check("llm", m, DEFAULT_LLM_MODELS[m], partial(answer, m))

if ran == 0:
    print("\nNo provider credentials set; nothing was tested.")
print(f"\n{ran - failures}/{ran} passed")
sys.exit(1 if failures else 0)
