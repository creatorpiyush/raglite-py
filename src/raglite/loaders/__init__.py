import os
from pathlib import Path
from typing import Callable, Dict

from ..errors import UnsupportedFileTypeError
from .base import BaseLoader
from .directory import DirectoryLoader, DirectoryLoadResult
from .docx import DocxLoader
from .json import JsonLoader
from .markdown import MarkdownLoader
from .pdf import PdfLoader
from .txt import TxtLoader
from .web import WebLoader

LOADERS: Dict[str, Callable[[str], BaseLoader]] = {
    ".pdf": PdfLoader,
    ".txt": TxtLoader,
    ".json": JsonLoader,
    ".md": MarkdownLoader,
    ".markdown": MarkdownLoader,
    ".docx": DocxLoader,
}


def is_url(path: str) -> bool:
    """Check if the target string is a web HTTP/HTTPS URL."""
    return path.startswith("http://") or path.startswith("https://")


def is_supported_file(file_path: str | Path) -> bool:
    """Check if the given file extension is supported by RAGLite loaders."""
    _, ext = os.path.splitext(str(file_path))
    return ext.lower() in LOADERS


def get_loader(file_path: str) -> BaseLoader:
    """Get the appropriate loader instance for the given file path or URL."""
    if is_url(file_path):
        return WebLoader(file_path)

    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    ctor = LOADERS.get(ext)
    if not ctor:
        supported = ", ".join(LOADERS.keys())
        raise UnsupportedFileTypeError(
            f'Unsupported file type: "{ext}". Supported: {supported}'
        )
    return ctor(file_path)


# Aliases for TS parity
getLoader = get_loader
isUrl = is_url
isSupportedFile = is_supported_file

__all__ = [
    "BaseLoader",
    "DirectoryLoader",
    "DirectoryLoadResult",
    "DocxLoader",
    "JsonLoader",
    "MarkdownLoader",
    "PdfLoader",
    "TxtLoader",
    "WebLoader",
    "get_loader",
    "getLoader",
    "is_supported_file",
    "isSupportedFile",
    "is_url",
    "isUrl",
]
