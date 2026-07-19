from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .constants import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_STORE_DIRNAME,
    DEFAULT_TOP_K,
)
from .types import EmbeddingProviderConfig, LLMProviderConfig, VectorStoreProviderConfig


class DocumentOptions(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    chunkSize: Optional[int] = Field(default=None, alias="chunkSize")
    overlap: Optional[int] = None
    topK: Optional[int] = Field(default=None, alias="topK")
    scoreThreshold: Optional[float] = Field(default=None, alias="scoreThreshold")
    storeDir: Optional[str] = Field(default=None, alias="storeDir")
    embeddings: Optional[EmbeddingProviderConfig] = None
    llm: Optional[LLMProviderConfig] = None
    vectorStore: Optional[VectorStoreProviderConfig] = Field(default=None, alias="vectorStore")
    logLevel: Optional[str] = Field(default=None, alias="logLevel")


class ResolvedConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    chunkSize: int = Field(..., alias="chunkSize")
    overlap: int
    topK: int = Field(..., alias="topK")
    scoreThreshold: float = Field(..., alias="scoreThreshold")
    storeDir: str = Field(..., alias="storeDir")
    embeddings: EmbeddingProviderConfig
    llm: Optional[LLMProviderConfig] = None
    vectorStore: Optional[VectorStoreProviderConfig] = Field(default=None, alias="vectorStore")
    logLevel: str = Field(..., alias="logLevel")


def resolve_config(options: Optional[Union[DocumentOptions, Dict[str, Any]]] = None) -> ResolvedConfig:
    if options is None:
        opts = DocumentOptions()
    elif isinstance(options, dict):
        opts = DocumentOptions.model_validate(options)
    else:
        opts = options

    chunk_size = opts.chunkSize if opts.chunkSize is not None else DEFAULT_CHUNK_SIZE
    overlap = opts.overlap if opts.overlap is not None else DEFAULT_CHUNK_OVERLAP
    top_k = opts.topK if opts.topK is not None else DEFAULT_TOP_K
    score_threshold = opts.scoreThreshold if opts.scoreThreshold is not None else DEFAULT_SCORE_THRESHOLD
    store_dir = opts.storeDir if opts.storeDir is not None else DEFAULT_STORE_DIRNAME
    embeddings = opts.embeddings if opts.embeddings is not None else EmbeddingProviderConfig(provider="local")
    log_level = opts.logLevel if opts.logLevel is not None else "info"

    return ResolvedConfig(
        chunkSize=chunk_size,
        overlap=overlap,
        topK=top_k,
        scoreThreshold=score_threshold,
        storeDir=store_dir,
        embeddings=embeddings,
        llm=opts.llm,
        vectorStore=opts.vectorStore,
        logLevel=log_level
    )
