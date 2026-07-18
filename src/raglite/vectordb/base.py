from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel

from ..types import ChunkMetadata, StoredChunk, IndexMetadata


class VectorSearchHit(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata
    score: float
    distance: float


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
    def search(self, embedding: List[float], top_k: int) -> List[VectorSearchHit]:
        pass

    @abstractmethod
    def count(self) -> int:
        pass

    @abstractmethod
    def save_index_metadata(self, metadata: IndexMetadata) -> None:
        pass

    @abstractmethod
    def read_index_metadata(self) -> Optional[IndexMetadata]:
        pass

    # TypeScript compatibility aliases
    def saveIndexMetadata(self, metadata: IndexMetadata) -> None:
        self.save_index_metadata(metadata)

    def readIndexMetadata(self) -> Optional[IndexMetadata]:
        return self.read_index_metadata()
