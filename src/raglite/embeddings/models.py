from ..types import EmbeddingProviderName

DEFAULT_EMBEDDING_MODELS: dict[EmbeddingProviderName, str] = {
    "openai": "text-embedding-3-small",
    "google": "gemini-embedding-2",
    "mistral": "mistral-embed",
    "cohere": "embed-english-v3.0",
    "voyage": "voyage-3",
    "ollama": "nomic-embed-text",
    "local": "all-MiniLM-L6-v2",
}

# Models the provider has shut down. An index built with one must be re-embedded.
RETIRED_EMBEDDING_MODELS: dict[EmbeddingProviderName, tuple[str, ...]] = {
    "google": ("text-embedding-004", "embedding-001"),
}


def is_retired_embedding_model(provider: EmbeddingProviderName, model: str) -> bool:
    return model in RETIRED_EMBEDDING_MODELS.get(provider, ())
