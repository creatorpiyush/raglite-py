"""
Custom Vector Store Example
=============================
Demonstrates implementing the VectorStore ABC to supply a fully
custom in-process vector store backed by a plain Python list.

Run:
    python examples/custom_store_example.py
"""

import math
from typing import List, Optional

from raglite import Document
from raglite.types import IndexMetadata, StoredChunk
from raglite.vectordb.base import VectorSearchHit, VectorStore


class SimpleListVectorStore(VectorStore):
    """Minimal custom vector store — no persistence, no dependencies."""

    def __init__(self, ns: str) -> None:
        self._namespace = ns
        self._chunks: List[StoredChunk] = []
        self._meta: Optional[IndexMetadata] = None

    @property
    def namespace(self) -> str:
        return self._namespace

    def load(self) -> None:
        pass  # no-op; state is in-process only

    def reset(self) -> None:
        self._chunks = []
        self._meta = None

    def add(self, chunks: List[StoredChunk]) -> None:
        self._chunks.extend(chunks)

    def search(self, embedding: List[float], top_k: int) -> List[VectorSearchHit]:
        def cosine(a: List[float], b: List[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb) if na and nb else 0.0

        scored = [
            VectorSearchHit(
                id=c.id,
                text=c.text,
                metadata=c.metadata,
                score=cosine(embedding, c.embedding),
                distance=1.0 - cosine(embedding, c.embedding),
            )
            for c in self._chunks
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._chunks)

    def save_index_metadata(self, metadata: IndexMetadata) -> None:
        self._meta = metadata

    def read_index_metadata(self) -> Optional[IndexMetadata]:
        return self._meta


def main() -> None:
    print("=== Custom Vector Store Example ===\n")

    custom_store = SimpleListVectorStore("demo-namespace")

    doc = Document(
        "./examples/sample.txt",
        {
            "embeddings": {"provider": "local"},
            "vectorStore": custom_store,   # pass instance directly
            "chunkSize": 50,
            "overlap": 10,
            "logLevel": "info",
        },
    )

    print("Building index with custom store...")
    result = doc.build(rebuild=True)
    print(f"Built: {result}\n")
    print(f"Chunks in custom store: {custom_store.count()}\n")

    query = "warranty policy"
    print(f'Searching for: "{query}"')
    hits = doc.search(query, top_k=3)
    for i, h in enumerate(hits, 1):
        preview = h.text.replace("\n", " ")[:90]
        print(f"  #{i} score={h.score:.4f}  {preview}...")

    print("\nDone.")


if __name__ == "__main__":
    main()
