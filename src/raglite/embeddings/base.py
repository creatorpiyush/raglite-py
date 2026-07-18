from abc import ABC, abstractmethod
from typing import List, Optional

from ..types import EmbeddingProviderName


class Embedder(ABC):
    @property
    @abstractmethod
    def provider(self) -> EmbeddingProviderName:
        pass

    @property
    @abstractmethod
    def model(self) -> str:
        pass

    @property
    @abstractmethod
    def dimensions(self) -> Optional[int]:
        pass

    @dimensions.setter
    @abstractmethod
    def dimensions(self, val: Optional[int]) -> None:
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass

    # TypeScript compatibility aliases
    def embedDocuments(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embedQuery(self, text: str) -> List[float]:
        return self.embed_query(text)
