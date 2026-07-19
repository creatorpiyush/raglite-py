from typing import TYPE_CHECKING

from ..errors import VectorDBError
from .base import VectorStore
from .memory import MemoryVectorStore

if TYPE_CHECKING:
    from ..types import VectorStoreProviderConfig


def create_vector_store(config: "VectorStoreProviderConfig", namespace: str) -> VectorStore:
    """Create a VectorStore from a configuration object."""
    provider = config.provider

    if provider == "memory":
        store_dir = getattr(config, "store_dir", None) or ".raglite"
        return MemoryVectorStore(store_dir, namespace)

    if provider == "qdrant":
        from .qdrant import QdrantVectorStore  # deferred import — optional dep

        return QdrantVectorStore(
            url=getattr(config, "url", "http://localhost:6333"),
            namespace=namespace,
            api_key=getattr(config, "api_key", None),
            collection_name=getattr(config, "index_name", None),
        )

    if provider == "pinecone":
        from .pinecone import PineconeVectorStore  # deferred import — optional dep

        api_key = getattr(config, "api_key", None)
        url = getattr(config, "url", None)
        if not api_key or not url:
            raise VectorDBError("Pinecone requires both 'url' and 'api_key'")
        return PineconeVectorStore(url=url, api_key=api_key, namespace=namespace)

    raise VectorDBError(f"Unsupported vector store provider: {provider!r}")
