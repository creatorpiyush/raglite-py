from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_serializer

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


RetrievalMode = Literal["vector", "keyword", "hybrid"]
"""``vector``: embedding similarity only (the default). ``keyword``: BM25 over
chunk texts only; good for exact terms such as error codes or SKUs.
``hybrid``: both, merged with Reciprocal Rank Fusion."""


class HybridWeights(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    vector: Optional[float] = None
    keyword: Optional[float] = None


class HybridOptions(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    # RRF constant; larger values flatten the difference between ranks (default 60).
    rrfK: Optional[float] = Field(
        default=None, validation_alias=AliasChoices("rrfK", "rrf_k"), serialization_alias="rrfK"
    )
    # Results taken from each retriever before fusion (default max(50, top_k * 4)).
    candidates: Optional[int] = None
    # Relative weight of each retriever in hybrid mode (default 1 each).
    weights: Optional[HybridWeights] = None


class RetrievalOptions(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    mode: Optional[RetrievalMode] = None
    hybrid: Optional[HybridOptions] = None


class SearchScores(BaseModel):
    """Per-retriever scores of a hybrid or keyword search result."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")

    vector: Optional[float] = None  # cosine similarity, when in the vector results
    keyword: Optional[float] = None  # BM25 score, when in the keyword results
    fused: Optional[float] = None  # raw RRF score; ``score`` is this normalised to [0, 1]

    @model_serializer(mode="wrap")
    def _drop_missing(self, handler: Any) -> Dict[str, Any]:
        return {k: v for k, v in handler(self).items() if v is not None}


class SearchResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    id: str
    text: str
    metadata: ChunkMetadata
    # Cosine similarity in vector mode; normalised fused rank score in keyword and hybrid modes.
    score: float
    distance: float
    # Set in keyword and hybrid modes.
    scores: Optional[SearchScores] = None

    @model_serializer(mode="wrap")
    def _drop_missing_scores(self, handler: Any) -> Dict[str, Any]:
        # Keeps vector-mode output identical to earlier releases.
        data = handler(self)
        if data.get("scores") is None:
            data.pop("scores", None)
        return data


class Citation(BaseModel):
    """A passage the answer cited as [n]."""

    model_config = ConfigDict(populate_by_name=True)

    n: int  # the number used in the answer, e.g. 2 for "[2]"
    source: str
    chunk: int
    page: Optional[int] = None
    pageEnd: Optional[int] = None
    section: Optional[str] = None
    text: str  # the cited passage

    @model_serializer(mode="wrap")
    def _omit_unknown(self, handler: Any) -> Dict[str, Any]:
        # Leave out page and section when unknown, as the TypeScript SDK does.
        return {k: v for k, v in handler(self).items() if v is not None}


class AnswerResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    text: str
    provider: str
    model: str
    usage: Optional[Dict[str, Optional[int]]] = None
    finishReason: Optional[str] = Field(default=None, alias="finishReason")
    # Passages the answer cites as [n], in order of first citation.
    citations: List[Citation] = Field(default_factory=list)


class IndexMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    version: str  # package version that built the index (informational)
    # Index layout version; see INDEX_FORMAT_VERSION. None on indexes built before 1.2.2.
    formatVersion: Optional[int] = Field(default=None, alias="formatVersion")
    source: str
    sourceHash: str = Field(..., alias="sourceHash")
    chunkSize: int = Field(..., alias="chunkSize")
    overlap: int
    embeddingProvider: EmbeddingProviderName = Field(..., alias="embeddingProvider")
    embeddingModel: str = Field(..., alias="embeddingModel")
    embeddingDimensions: int = Field(..., alias="embeddingDimensions")
    chunkCount: int = Field(..., alias="chunkCount")
    createdAt: str = Field(..., alias="createdAt")
