from abc import ABC, abstractmethod


class BaseLoader(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def load(self) -> str:
        """Load document text asynchronously or synchronously."""
        pass
