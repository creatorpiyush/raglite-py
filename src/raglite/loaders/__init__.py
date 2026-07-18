import os
from .base import BaseLoader
from .docx import DocxLoader
from .json import JsonLoader
from .markdown import MarkdownLoader
from .pdf import PdfLoader
from .txt import TxtLoader
from ..errors import UnsupportedFileTypeError

LOADERS = {
    ".pdf": PdfLoader,
    ".txt": TxtLoader,
    ".json": JsonLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".docx": DocxLoader,
}


def get_loader(file_path: str) -> BaseLoader:
    """Get the appropriate loader instance for the given file path."""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    ctor = LOADERS.get(ext)
    if not ctor:
        supported = ", ".join(LOADERS.keys())
        raise UnsupportedFileTypeError(
            f'Unsupported file type: "{ext}". Supported: {supported}'
        )
    return ctor(file_path)


# Alias for TS parity
getLoader = get_loader
