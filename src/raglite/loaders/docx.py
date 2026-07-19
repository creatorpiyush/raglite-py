import docx

from ..errors import LoaderError
from .base import BaseLoader


class DocxLoader(BaseLoader):
    def load(self) -> str:
        try:
            doc = docx.Document(self.file_path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return "\n".join(full_text).strip()
        except Exception as cause:
            raise LoaderError(f"Failed to load DOCX: {self.file_path}", cause=cause)
