from pypdf import PdfReader
from .base import BaseLoader
from ..errors import LoaderError


class PdfLoader(BaseLoader):
    def load(self) -> str:
        try:
            reader = PdfReader(self.file_path)
            text_parts = []
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n".join(text_parts).strip()
        except Exception as cause:
            raise LoaderError(f"Failed to load PDF: {self.file_path}", cause=cause)
