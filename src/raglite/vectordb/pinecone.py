import json
import urllib.error
import urllib.request
from typing import List, Optional

from ..errors import VectorDBError
from ..types import ChunkMetadata, IndexMetadata, StoredChunk
from .base import VectorSearchHit, VectorStore

_METADATA_ID = "__metadata__"


class PineconeVectorStore(VectorStore):
    """Vector store backed by Pinecone Serverless (REST API, no SDK)."""

    def __init__(
        self,
        url: str,
        api_key: str,
        namespace: str,
    ) -> None:
        self._namespace = namespace
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._count_cache: int = 0

    @property
    def namespace(self) -> str:
        return self._namespace

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Api-Key": self._api_key,
        }

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = f"{self._url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise VectorDBError(f"Pinecone HTTP {e.code}: {e.reason}", cause=e)

    def _ns_params(self) -> str:
        return f"?namespace={self._namespace}"

    def load(self) -> None:
        meta = self.read_index_metadata()
        if meta:
            self._count_cache = meta.chunkCount

    def reset(self) -> None:
        try:
            self._request(
                "POST",
                f"/vectors/delete{self._ns_params()}",
                {"deleteAll": True},
            )
        except VectorDBError:
            pass
        self._count_cache = 0

    def add(self, chunks: List[StoredChunk]) -> None:
        if not chunks:
            return
        vectors = [
            {
                "id": c.id,
                "values": c.embedding,
                "metadata": {
                    "text": c.text,
                    **c.metadata.model_dump(by_alias=True),
                },
            }
            for c in chunks
        ]
        self._request(
            "POST",
            f"/vectors/upsert{self._ns_params()}",
            {"vectors": vectors},
        )
        self._count_cache += len(chunks)

    def search(self, embedding: List[float], top_k: int) -> List[VectorSearchHit]:
        try:
            data = self._request(
                "POST",
                f"/query{self._ns_params()}",
                {
                    "vector": embedding,
                    "topK": top_k + 1,
                    "includeMetadata": True,
                },
            )
        except VectorDBError:
            return []

        hits: List[VectorSearchHit] = []
        for m in data.get("matches", []):
            if m.get("id") == _METADATA_ID:
                continue
            meta_raw = m.get("metadata") or {}
            score = float(m.get("score", 0.0))
            metadata = ChunkMetadata(
                source=meta_raw.get("source", ""),
                chunk=int(meta_raw.get("chunk", 0)),
                totalChunks=int(meta_raw.get("totalChunks", 0)),
            )
            hits.append(
                VectorSearchHit(
                    id=m["id"],
                    text=meta_raw.get("text", ""),
                    metadata=metadata,
                    score=score,
                    distance=1.0 - score,
                )
            )
        return hits[:top_k]

    def count(self) -> int:
        return self._count_cache

    def save_index_metadata(self, metadata: IndexMetadata) -> None:
        dim = metadata.embeddingDimensions
        zero_vec = [0.0] * dim
        payload = metadata.model_dump(by_alias=True)
        payload["isMetadata"] = True
        self._request(
            "POST",
            f"/vectors/upsert{self._ns_params()}",
            {"vectors": [{"id": _METADATA_ID, "values": zero_vec, "metadata": payload}]},
        )
        self._count_cache = metadata.chunkCount

    def read_index_metadata(self) -> Optional[IndexMetadata]:
        try:
            data = self._request(
                "GET",
                f"/vectors/fetch?ids={_METADATA_ID}{self._ns_params().replace('?', '&')}",
            )
        except VectorDBError:
            return None

        vectors = data.get("vectors") or {}
        record = vectors.get(_METADATA_ID)
        if not record:
            return None
        m = record.get("metadata") or {}
        if not m.get("isMetadata"):
            return None
        try:
            return IndexMetadata(
                version=m.get("version", ""),
                source=m.get("source", ""),
                sourceHash=m.get("sourceHash", ""),
                chunkSize=int(m.get("chunkSize", 0)),
                overlap=int(m.get("overlap", 0)),
                embeddingProvider=m.get("embeddingProvider", "local"),
                embeddingModel=m.get("embeddingModel", ""),
                embeddingDimensions=int(m.get("embeddingDimensions", 0)),
                chunkCount=int(m.get("chunkCount", 0)),
                createdAt=m.get("createdAt", ""),
            )
        except Exception:
            return None
