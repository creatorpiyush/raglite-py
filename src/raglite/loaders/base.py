from abc import ABC, abstractmethod
from typing import List, Optional


class BaseLoader(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path
        # Text of each page, set by load() for paged formats such as PDF.
        self.pages: Optional[List[str]] = None

    @abstractmethod
    def load(self) -> str:
        """Load document text asynchronously or synchronously."""
        pass
