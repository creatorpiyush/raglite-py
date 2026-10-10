import importlib
from types import ModuleType


class RagLiteError(Exception):
    """Base error for the RAGLite toolkit."""
    def __init__(self, message: str, cause: Exception | None = None):
        super().__init__(message)
        self.message = message
        if cause is not None:
            self.__cause__ = cause


class UnsupportedFileTypeError(RagLiteError):
    """Raised when file extension is not supported by loaders."""
    pass


class FileNotIndexedError(RagLiteError):
    """Raised when trying to search/ask an unindexed document."""
    pass


class LoaderError(RagLiteError):
    """Raised when file loading fails."""
    pass


class ChunkingError(RagLiteError):
    """Raised when chunking configuration or execution fails."""
    pass


class EmbeddingError(RagLiteError):
    """Raised when embedding generation fails."""
    pass


class VectorDBError(RagLiteError):
    """Raised when vector database actions fail."""
    pass


class LLMError(RagLiteError):
    """Raised when LLM interactions fail."""
    pass


class ConfigError(RagLiteError):
    """Raised when configuration values are invalid."""
    pass


class BuildCancelledError(RagLiteError):
    """Raised when build() is cancelled; the previous index is kept."""
    pass


def import_optional(module: str, extra: str, purpose: str) -> ModuleType:
    """Import an optional dependency, or raise ConfigError naming the extra to install."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as cause:
        raise ConfigError(
            f"{purpose} requires the raglite-toolkit[{extra}] extra. "
            f"Install it with: pip install 'raglite-toolkit[{extra}]'",
            cause=cause,
        ) from cause
