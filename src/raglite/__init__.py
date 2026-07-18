from .constants import PACKAGE_NAME, PACKAGE_VERSION

VERSION = PACKAGE_VERSION

from .errors import (
    RagLiteError,
    UnsupportedFileTypeError,
    FileNotIndexedError,
    LoaderError,
    ChunkingError,
    EmbeddingError,
    VectorDBError,
    LLMError,
    ConfigError,
)

from .types import (
    LLMProviderName,
    EmbeddingProviderName,
    LLMProviderConfig,
    EmbeddingProviderConfig,
    ChunkMetadata,
    StoredChunk,
    SearchResult,
    AnswerResult,
    IndexMetadata,
)

from .config import DocumentOptions, resolve_config

from .utils.logger import create_logger

from .loaders import (
    BaseLoader,
    TxtLoader,
    MarkdownLoader,
    JsonLoader,
    PdfLoader,
    DocxLoader,
    get_loader,
)

from .chunking import BaseChunker, RecursiveChunker

from .vectordb import MemoryVectorStore

from .retrieval import Retriever

from .embeddings import (
    DEFAULT_EMBEDDING_MODELS,
    LocalEmbedder,
    RemoteEmbedder,
    create_embedder,
)

from .llm import (
    DEFAULT_LLM_MODELS,
    build_system_prompt,
    build_user_prompt,
    create_llm,
    generate_answer,
    stream_answer,
)

from .core.document import Document

__all__ = [
    "PACKAGE_NAME",
    "PACKAGE_VERSION",
    "VERSION",
    "RagLiteError",
    "UnsupportedFileTypeError",
    "FileNotIndexedError",
    "LoaderError",
    "ChunkingError",
    "EmbeddingError",
    "VectorDBError",
    "LLMError",
    "ConfigError",
    "LLMProviderName",
    "EmbeddingProviderName",
    "LLMProviderConfig",
    "EmbeddingProviderConfig",
    "ChunkMetadata",
    "StoredChunk",
    "SearchResult",
    "AnswerResult",
    "IndexMetadata",
    "DocumentOptions",
    "resolve_config",
    "create_logger",
    "BaseLoader",
    "TxtLoader",
    "MarkdownLoader",
    "JsonLoader",
    "PdfLoader",
    "DocxLoader",
    "get_loader",
    "BaseChunker",
    "RecursiveChunker",
    "MemoryVectorStore",
    "Retriever",
    "DEFAULT_EMBEDDING_MODELS",
    "LocalEmbedder",
    "RemoteEmbedder",
    "create_embedder",
    "DEFAULT_LLM_MODELS",
    "build_system_prompt",
    "build_user_prompt",
    "create_llm",
    "generate_answer",
    "stream_answer",
    "Document",
]
