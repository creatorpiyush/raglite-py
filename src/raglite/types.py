from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

LLMProviderName = Literal[
    "openai",
    "anthropic",
    "google",
    "mistral",
    "cohere",
    "groq",
    "xai",
    "ollama"
]

EmbeddingProviderName = Literal[
    "openai",
    "google",
    "mistral",
    "cohere",
    "voyage",
    "ollama",
    "local"
]

VectorStoreProviderName = Literal["memory", "qdrant", "pinecone"]


class VectorStoreProviderConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    provider: VectorStoreProviderName
    url: Optional[str] = None
    api_key: Optional[str] = Field(default=None, alias="apiKey")
    index_name: Optional[str] = Field(default=None, alias="indexName")
    store_dir: Optional[str] = Field(default=None, alias="storeDir")


class LLMProviderConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    provider: LLMProviderName
    model: Optional[str] = None
    apiKey: Optional[str] = Field(default=None, alias="apiKey")
    baseURL: Optional[str] = Field(default=None, alias="baseURL")
    temperature: Optional[float] = None
    maxTokens: Optional[int] = Field(default=None, alias="maxTokens")


class EmbeddingProviderConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    provider: EmbeddingProviderName
    model: Optional[str] = None
    apiKey: Optional[str] = Field(default=None, alias="apiKey")
    baseURL: Optional[str] = Field(default=None, alias="baseURL")


class ChunkMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    source: str
    chunk: int
    totalChunks: int


class StoredChunk(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    text: str
    embedding: List[float]
    metadata: ChunkMetadata


class SearchResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    text: str
    metadata: ChunkMetadata
    score: float
    distance: float


class AnswerResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    text: str
    provider: str
    model: str
    usage: Optional[Dict[str, Optional[int]]] = None
    finishReason: Optional[str] = Field(default=None, alias="finishReason")


class IndexMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    version: str
    source: str
    sourceHash: str = Field(..., alias="sourceHash")
    chunkSize: int = Field(..., alias="chunkSize")
    overlap: int
    embeddingProvider: EmbeddingProviderName = Field(..., alias="embeddingProvider")
    embeddingModel: str = Field(..., alias="embeddingModel")
    embeddingDimensions: int = Field(..., alias="embeddingDimensions")
    chunkCount: int = Field(..., alias="chunkCount")
    createdAt: str = Field(..., alias="createdAt")
