from .base import BaseLoader
from ..errors import LoaderError


class MarkdownLoader(BaseLoader):
    def load(self) -> str:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as cause:
            raise LoaderError(f"Failed to load Markdown: {self.file_path}", cause=cause)
