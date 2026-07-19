import hashlib
import json
import urllib.error
import urllib.request
from typing import List, Optional

from ..errors import VectorDBError
from ..types import ChunkMetadata, IndexMetadata, StoredChunk
from .base import VectorSearchHit, VectorStore


def _uuid_from_str(s: str) -> str:
    digest = hashlib.md5(s.encode()).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


_METADATA_UUID = "00000000-0000-0000-0000-000000000000"


class QdrantVectorStore(VectorStore):
    """Vector store backed by a Qdrant REST API (local or cloud)."""

    def __init__(
        self,
        url: str,
        namespace: str,
        api_key: Optional[str] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        self._namespace = namespace
        self._url = url.rstrip("/")
        self._api_key = api_key
        self._collection = collection_name or f"raglite_{namespace}"
        self._count_cache: int = 0

    @property
    def namespace(self) -> str:
        return self._namespace

    def _headers(self) -> dict:
        h: dict = {"Content-Type": "application/json"}
        if self._api_key:
            h["api-key"] = self._api_key
        return h

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        url = f"{self._url}{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, headers=self._headers(), method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise VectorDBError(f"Qdrant HTTP {e.code}: {e.reason}", cause=e)

    def _ensure_collection(self, dimensions: int) -> None:
        try:
            self._request("GET", f"/collections/{self._collection}")
        except VectorDBError:
            self._request(
                "PUT",
                f"/collections/{self._collection}",
                {"vectors": {"size": dimensions, "distance": "Cosine"}},
            )

    def load(self) -> None:
        meta = self.read_index_metadata()
        if meta:
            self._count_cache = meta.chunkCount

    def reset(self) -> None:
        try:
            self._request("DELETE", f"/collections/{self._collection}")
        except VectorDBError:
            pass
        self._count_cache = 0

    def add(self, chunks: List[StoredChunk]) -> None:
        if not chunks:
            return
        self._ensure_collection(len(chunks[0].embedding))
        points = [
            {
                "id": _uuid_from_str(c.id),
                "vector": c.embedding,
                "payload": {
                    "id": c.id,
                    "text": c.text,
                    "metadata": c.metadata.model_dump(by_alias=True),
                    "isMetadata": False,
                },
            }
            for c in chunks
        ]
        self._request("PUT", f"/collections/{self._collection}/points", {"points": points})
        self._count_cache += len(chunks)

    def search(self, embedding: List[float], top_k: int) -> List[VectorSearchHit]:
        try:
            data = self._request(
                "POST",
                f"/collections/{self._collection}/points/search",
                {
                    "vector": embedding,
                    "limit": top_k + 1,
                    "with_payload": True,
                },
            )
        except VectorDBError:
            return []

        hits: List[VectorSearchHit] = []
        for r in data.get("result", []):
            payload = r.get("payload", {}) or {}
            if r.get("id") == _METADATA_UUID or payload.get("isMetadata"):
                continue
            raw_meta = payload.get("metadata") or {}
            metadata = ChunkMetadata(
                source=raw_meta.get("source", ""),
                chunk=raw_meta.get("chunk", 0),
                totalChunks=raw_meta.get("totalChunks", 0),
            )
            score = float(r.get("score", 0.0))
            hits.append(
                VectorSearchHit(
                    id=payload.get("id", r["id"]),
                    text=payload.get("text", ""),
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
        self._ensure_collection(dim)
        zero_vec = [0.0] * dim
        payload_data = metadata.model_dump(by_alias=True)
        payload_data["isMetadata"] = True
        self._request(
            "PUT",
            f"/collections/{self._collection}/points",
            {
                "points": [
                    {
                        "id": _METADATA_UUID,
                        "vector": zero_vec,
                        "payload": payload_data,
                    }
                ]
            },
        )
        self._count_cache = metadata.chunkCount

    def read_index_metadata(self) -> Optional[IndexMetadata]:
        try:
            data = self._request(
                "POST",
                f"/collections/{self._collection}/points",
                {"ids": [_METADATA_UUID], "with_payload": True},
            )
        except VectorDBError:
            return None

        results = data.get("result", [])
        if not results:
            return None
        payload = results[0].get("payload") or {}
        if not payload.get("isMetadata"):
            return None
        try:
            return IndexMetadata(
                version=payload.get("version", ""),
                source=payload.get("source", ""),
                sourceHash=payload.get("sourceHash", ""),
                chunkSize=payload.get("chunkSize", 0),
                overlap=payload.get("overlap", 0),
                embeddingProvider=payload.get("embeddingProvider", "local"),
                embeddingModel=payload.get("embeddingModel", ""),
                embeddingDimensions=payload.get("embeddingDimensions", 0),
                chunkCount=payload.get("chunkCount", 0),
                createdAt=payload.get("createdAt", ""),
            )
        except Exception:
            return None
