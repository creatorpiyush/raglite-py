import math
import os
from typing import List, Optional

from ..errors import EmbeddingError
from ..types import EmbeddingProviderConfig, EmbeddingProviderName
from .base import Embedder
from .models import DEFAULT_EMBEDDING_MODELS


def normalize(vector: List[float]) -> List[float]:
    s = sum(v * v for v in vector)
    norm = math.sqrt(s)
    if norm == 0:
        return list(vector)
    return [v / norm for v in vector]


class RemoteEmbedder(Embedder):
    def __init__(
        self,
        provider: EmbeddingProviderName,
        model: str,
        config: EmbeddingProviderConfig,
    ):
        self._provider = provider
        self._model = model
        self._config = config
        self._dimensions: Optional[int] = None

    @property
    def provider(self) -> EmbeddingProviderName:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> Optional[int]:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, val: Optional[int]) -> None:
        self._dimensions = val

    @classmethod
    def create(cls, config: EmbeddingProviderConfig) -> "RemoteEmbedder":
        if config.provider == "local":
            raise EmbeddingError(
                "RemoteEmbedder cannot be constructed for the 'local' provider"
            )
        model = config.model or DEFAULT_EMBEDDING_MODELS[config.provider]
        return cls(config.provider, model, config)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            vectors = self._embed(texts, is_query=False)
            normalized = [normalize(v) for v in vectors]
            if self._dimensions is None and normalized:
                self._dimensions = len(normalized[0])
            return normalized
        except Exception as cause:
            raise EmbeddingError(
                f"Failed to embed {len(texts)} document(s) via {self.provider}",
                cause=cause,
            )

    def embed_query(self, text: str) -> List[float]:
        try:
            vectors = self._embed([text], is_query=True)
            if not vectors:
                raise EmbeddingError("Empty embedding returned by remote provider")
            normalized = normalize(vectors[0])
            if self._dimensions is None:
                self._dimensions = len(normalized)
            return normalized
        except Exception as cause:
            raise EmbeddingError(
                f"Failed to embed query via {self.provider}", cause=cause
            )

    def _embed(self, texts: List[str], is_query: bool) -> List[List[float]]:
        p = self.provider
        apiKey = self._config.apiKey
        baseURL = self._config.baseURL

        if p == "openai":
            from openai import OpenAI

            client = OpenAI(
                api_key=apiKey or os.environ.get("OPENAI_API_KEY"), base_url=baseURL
            )
            resp = client.embeddings.create(input=texts, model=self.model)
            return [data.embedding for data in resp.data]

        elif p == "google":
            import google.generativeai as genai

            key = (
                apiKey
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            )
            genai.configure(api_key=key)
            model_name = self.model
            if not model_name.startswith("models/"):
                model_name = f"models/{model_name}"
            resp = genai.embed_content(model=model_name, contents=texts)
            embeddings = resp.get("embedding")
            if not embeddings:
                raise ValueError(
                    f"Google API returned empty embedding response: {resp}"
                )
            if embeddings and not isinstance(embeddings[0], list):
                embeddings = [embeddings]
            return embeddings

        elif p == "cohere":
            import cohere

            key = apiKey or os.environ.get("COHERE_API_KEY")
            co = cohere.Client(api_key=key, base_url=baseURL)
            input_type = "search_query" if is_query else "search_document"
            resp = co.embed(texts=texts, model=self.model, input_type=input_type)
            return resp.embeddings

        elif p == "mistral":
            from mistralai import Mistral

            key = apiKey or os.environ.get("MISTRAL_API_KEY")
            client = Mistral(api_key=key, server_url=baseURL)
            resp = client.embeddings.create(inputs=texts, model=self.model)
            return [data.embedding for data in resp.data]

        elif p == "voyage":
            import voyageai

            key = apiKey or os.environ.get("VOYAGE_API_KEY")
            vo = voyageai.Client(api_key=key)
            input_type = "query" if is_query else "document"
            resp = vo.embed(texts, model=self.model, input_type=input_type)
            return resp.embeddings

        elif p == "ollama":
            from openai import OpenAI

            raw = (baseURL or "http://localhost:11434").rstrip("/")
            base = raw if raw.endswith("/v1") else f"{raw}/v1"
            key = apiKey or "ollama"
            client = OpenAI(api_key=key, base_url=base)
            resp = client.embeddings.create(input=texts, model=self.model)
            return [data.embedding for data in resp.data]

        else:
            raise EmbeddingError(f"Unsupported remote embedding provider: {p}")
