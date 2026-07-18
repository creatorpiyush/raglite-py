from abc import ABC, abstractmethod
from typing import List


class BaseChunker(ABC):
    def __init__(self, chunk_size: int, overlap: int):
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def split(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass
