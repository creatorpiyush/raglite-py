import hashlib
import json
import urllib.error
import urllib.request
from typing import List, Optional

from ..errors import VectorDBError
from ..types import ChunkMetadata, IndexMetadata, StoredChunk
from .base import IndexedChunk, VectorSearchHit, VectorStore


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
        # With an explicit collection name, several namespaces may share one
        # collection, so every operation is scoped to this namespace instead
        # of touching the whole collection.
        self._shared = collection_name is not None
        self._metadata_id = (
            _uuid_from_str(f"{namespace}__metadata__") if self._shared else _METADATA_UUID
        )
        self._count_cache: int = 0

    @property
    def namespace(self) -> str:
        return self._namespace

    def _namespace_filter(self) -> dict:
        return {"must": [{"key": "namespace", "match": {"value": self._namespace}}]}

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
            if self._shared:
                self._request(
                    "POST",
                    f"/collections/{self._collection}/points/delete?wait=true",
                    {"filter": self._namespace_filter()},
                )
            else:
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
                    "namespace": self._namespace,
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
        body: dict = {
            "vector": embedding,
            "limit": top_k + 1,
            "with_payload": True,
        }
        if self._shared:
            body["filter"] = self._namespace_filter()
        try:
            data = self._request(
                "POST",
                f"/collections/{self._collection}/points/search",
                body,
            )
        except VectorDBError:
            return []

        hits: List[VectorSearchHit] = []
        for r in data.get("result", []):
            payload = r.get("payload", {}) or {}
            if r.get("id") == self._metadata_id or payload.get("isMetadata"):
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

    def list_chunks(self) -> List[IndexedChunk]:
        chunks: List[IndexedChunk] = []
        offset = None
        while True:
            body: dict = {"limit": 256, "with_payload": True, "with_vector": False}
            if offset is not None:
                body["offset"] = offset
            if self._shared:
                body["filter"] = self._namespace_filter()
            data = self._request("POST", f"/collections/{self._collection}/points/scroll", body)
            result = data.get("result") or {}
            for p in result.get("points", []):
                payload = p.get("payload") or {}
                if p.get("id") == self._metadata_id or payload.get("isMetadata"):
                    continue
                raw_meta = payload.get("metadata") or {}
                chunks.append(
                    IndexedChunk(
                        id=payload.get("id", p.get("id")),
                        text=payload.get("text", ""),
                        metadata=ChunkMetadata(
                            source=raw_meta.get("source", ""),
                            chunk=raw_meta.get("chunk", 0),
                            totalChunks=raw_meta.get("totalChunks", 0),
                        ),
                    )
                )
            offset = result.get("next_page_offset")
            if offset is None:
                break
        # Scroll returns points in id (uuid) order; restore chunk order.
        return sorted(chunks, key=lambda c: c.id)

    def save_index_metadata(self, metadata: IndexMetadata) -> None:
        dim = metadata.embeddingDimensions
        self._ensure_collection(dim)
        zero_vec = [0.0] * dim
        payload_data = metadata.model_dump(by_alias=True)
        payload_data["isMetadata"] = True
        payload_data["namespace"] = self._namespace
        self._request(
            "PUT",
            f"/collections/{self._collection}/points",
            {
                "points": [
                    {
                        "id": self._metadata_id,
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
                {"ids": [self._metadata_id], "with_payload": True},
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
                formatVersion=payload.get("formatVersion"),
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
