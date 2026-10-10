from pypdf import PdfReader

from ..errors import LoaderError
from .base import BaseLoader


class PdfLoader(BaseLoader):
    def load(self) -> str:
        try:
            reader = PdfReader(self.file_path)
            # Keep empty pages so page numbers stay right; they add only whitespace.
            self.pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(self.pages).strip()
        except Exception as cause:
            raise LoaderError(f"Failed to load PDF: {self.file_path}", cause=cause)
