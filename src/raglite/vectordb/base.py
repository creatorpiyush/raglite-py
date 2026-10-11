from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from ..types import ChunkMetadata, IndexMetadata, StoredChunk


class VectorSearchHit(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata
    score: float
    distance: float


class IndexedChunk(BaseModel):
    """A stored chunk without its embedding."""

    id: str
    text: str
    metadata: ChunkMetadata


class VectorStore(ABC):
    @property
    @abstractmethod
    def namespace(self) -> str:
        pass

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def add(self, chunks: List[StoredChunk]) -> None:
        pass

    @abstractmethod
    def search(
        self, embedding: List[float], top_k: int, filter: Optional[Dict[str, Any]] = None
    ) -> List[VectorSearchHit]:
        """``filter`` holds conditions on per-chunk metadata (``chunk``,
        ``totalChunks``, ``page``, ``pageEnd``, ``section``); see
        ``matches_filter()``. Apply it before the ``top_k`` cut. Stores
        written before 2.2 take no ``filter`` argument; Document then filters
        their results itself, which may return fewer than ``top_k``."""

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def save_index_metadata(self, metadata: IndexMetadata) -> None:
        pass

    @abstractmethod
    def read_index_metadata(self) -> Optional[IndexMetadata]:
        pass

    def list_chunks(self) -> Optional[List[IndexedChunk]]:
        """Optional. Every stored chunk, without embeddings, or None if the
        store cannot list them. Lets keyword and hybrid search rebuild a
        missing keyword index from the store instead of asking for a full
        rebuild."""
        return None

    def keyword_search(
        self, query: str, top_k: int, filter: Optional[Dict[str, Any]] = None
    ) -> Optional[List[VectorSearchHit]]:
        """Optional. Native keyword search (for example a full-text index),
        used in place of the built-in BM25 index; None if unsupported. Higher
        ``score`` means a better match."""
        return None

    # TypeScript compatibility aliases
    def saveIndexMetadata(self, metadata: IndexMetadata) -> None:
        self.save_index_metadata(metadata)

    def readIndexMetadata(self) -> Optional[IndexMetadata]:
        return self.read_index_metadata()


def chunk_metadata_from(raw: Dict[str, Any]) -> ChunkMetadata:
    """ChunkMetadata from a remote store's payload, keeping page and section."""
    fields = {k: v for k, v in raw.items() if k != "text" and v is not None}
    for key in ("chunk", "totalChunks", "page", "pageEnd"):
        # Pinecone returns stored numbers as floats.
        if isinstance(fields.get(key), float) and fields[key].is_integer():
            fields[key] = int(fields[key])
    fields.setdefault("source", "")
    fields.setdefault("chunk", 0)
    fields.setdefault("totalChunks", 0)
    return ChunkMetadata.model_validate(fields)
