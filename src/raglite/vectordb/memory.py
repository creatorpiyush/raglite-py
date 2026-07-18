import os
import json
import shutil
from typing import List, Optional

from .base import VectorStore, VectorSearchHit
from ..types import StoredChunk, IndexMetadata
from ..errors import VectorDBError


class MemoryVectorStore(VectorStore):
    def __init__(self, store_dir: str, namespace: str):
        self._namespace = namespace
        self.store_dir = os.path.join(store_dir, namespace)
        self.chunks_path = os.path.join(self.store_dir, "chunks.json")
        self.metadata_path = os.path.join(self.store_dir, "metadata.json")
        self.chunks: List[StoredChunk] = []

    @property
    def namespace(self) -> str:
        return self._namespace

    def load(self) -> None:
        if not os.path.exists(self.chunks_path):
            self.chunks = []
            return
        try:
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            self.chunks = [StoredChunk.model_validate(c) for c in parsed]
        except Exception as cause:
            raise VectorDBError(
                f"Failed to load vector store at {self.chunks_path}", cause=cause
            )

    def reset(self) -> None:
        self.chunks = []
        if os.path.exists(self.store_dir):
            try:
                shutil.rmtree(self.store_dir)
            except Exception as cause:
                raise VectorDBError(
                    f"Failed to reset vector store at {self.store_dir}", cause=cause
                )

    def add(self, chunks: List[StoredChunk]) -> None:
        if not chunks:
            return
        self.chunks.extend(chunks)
        self._persist()

    def search(self, embedding: List[float], top_k: int) -> List[VectorSearchHit]:
        if not self.chunks:
            return []

        scored = []
        for c in self.chunks:
            score = self._dot(embedding, c.embedding)
            scored.append(
                VectorSearchHit(
                    id=c.id,
                    text=c.text,
                    metadata=c.metadata,
                    score=score,
                    distance=1.0 - score,
                )
            )

        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self.chunks)

    def save_index_metadata(self, metadata: IndexMetadata) -> None:
        try:
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata.model_dump(by_alias=True), f, indent=2)
        except Exception as cause:
            raise VectorDBError(
                f"Failed to save index metadata at {self.metadata_path}", cause=cause
            )

    def read_index_metadata(self) -> Optional[IndexMetadata]:
        if not os.path.exists(self.metadata_path):
            return None
        try:
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                parsed = json.load(f)
            return IndexMetadata.model_validate(parsed)
        except Exception as cause:
            raise VectorDBError(
                f"Failed to read index metadata at {self.metadata_path}", cause=cause
            )

    def _persist(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.chunks_path), exist_ok=True)
            with open(self.chunks_path, "w", encoding="utf-8") as f:
                json.dump([c.model_dump(by_alias=True) for c in self.chunks], f)
        except Exception as cause:
            raise VectorDBError(
                f"Failed to persist vector store at {self.chunks_path}", cause=cause
            )

    def _dot(self, a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))
