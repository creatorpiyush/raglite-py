from typing import List, Optional

from ..errors import EmbeddingError, import_optional
from ..types import EmbeddingProviderConfig, EmbeddingProviderName
from .base import Embedder
from .models import DEFAULT_EMBEDDING_MODELS


class LocalEmbedder(Embedder):
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._dimensions: Optional[int] = None
        self._model = None

    @property
    def provider(self) -> EmbeddingProviderName:
        return "local"

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> Optional[int]:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, val: Optional[int]) -> None:
        self._dimensions = val

    @classmethod
    def create(cls, config: EmbeddingProviderConfig) -> "LocalEmbedder":
        model = config.model or DEFAULT_EMBEDDING_MODELS["local"]
        if "all-MiniLM-L6-v2" in model:
            model = "all-MiniLM-L6-v2"
        return cls(model)

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        st = import_optional(
            "sentence_transformers", "local", 'The "local" embedding provider'
        )
        try:
            self._model = st.SentenceTransformer(self._model_name)
        except Exception as cause:
            raise EmbeddingError(
                f"Failed to load local embedding model '{self._model_name}'", cause=cause
            ) from cause
        return self._model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        embeddings = model.encode(texts, normalize_embeddings=True)
        vectors = [list(map(float, v)) for v in embeddings]
        if self._dimensions is None and vectors:
            self._dimensions = len(vectors[0])
        return vectors

    def embed_query(self, text: str) -> List[float]:
        vecs = self.embed_documents([text])
        if not vecs:
            raise EmbeddingError("Empty embedding returned by local model")
        return vecs[0]
