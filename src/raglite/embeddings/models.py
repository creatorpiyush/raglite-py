from ..types import EmbeddingProviderName

DEFAULT_EMBEDDING_MODELS: dict[EmbeddingProviderName, str] = {
    "openai": "text-embedding-3-small",
    "google": "text-embedding-004",
    "mistral": "mistral-embed",
    "cohere": "embed-english-v3.0",
    "voyage": "voyage-3",
    "ollama": "nomic-embed-text",
    "local": "all-MiniLM-L6-v2",
}
