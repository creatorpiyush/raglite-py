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
