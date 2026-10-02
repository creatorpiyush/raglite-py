"""BM25 keyword index; mirrors raglite/src/retrieval/keyword-index.ts.

The sidecar file layout (``keyword.json``) and the scores are identical in
both SDKs, so either can read an index the other wrote.
"""

import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence

from pydantic import BaseModel

from ..text.tokenizer import TOKENIZER_NAME, tokenize
from ..types import ChunkMetadata, IndexMetadata
from ..vectordb.base import IndexedChunk

KEYWORD_INDEX_FORMAT = 1
BM25_K1 = 1.2
BM25_B = 0.75


class KeywordHit(BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata
    score: float  # BM25 score


class KeywordIndex:
    """BM25 keyword index over a document's chunks. It is built from the chunk
    texts alone, so it never needs extra embedding calls, and it is kept in a
    sidecar file next to the vector index so it works with every store."""

    def __init__(
        self,
        chunks: List[IndexedChunk],
        lengths: List[int],
        postings: Dict[str, List[int]],
    ) -> None:
        self._chunks = chunks
        self._lengths = lengths
        self._postings = postings
        total = sum(lengths)
        self._avg_length = total / len(lengths) if lengths and total > 0 else 1

    @classmethod
    def build(cls, chunks: Sequence[Any]) -> "KeywordIndex":
        indexed = [IndexedChunk(id=c.id, text=c.text, metadata=c.metadata) for c in chunks]
        lengths: List[int] = []
        postings: Dict[str, List[int]] = {}
        for index, chunk in enumerate(indexed):
            terms = tokenize(chunk.text)
            lengths.append(len(terms))
            tf: Dict[str, int] = {}
            for term in terms:
                tf[term] = tf.get(term, 0) + 1
            for term, count in tf.items():
                postings.setdefault(term, []).extend((index, count))
        return cls(indexed, lengths, postings)

    @classmethod
    def from_file(cls, data: Dict[str, Any]) -> "KeywordIndex":
        chunks = [IndexedChunk.model_validate(c) for c in data["chunks"]]
        return cls(chunks, list(data["lengths"]), dict(data["postings"]))

    def to_file(self, metadata: IndexMetadata) -> Dict[str, Any]:
        return {
            "format": KEYWORD_INDEX_FORMAT,
            "tokenizer": TOKENIZER_NAME,
            "index": {
                "sourceHash": metadata.sourceHash,
                "createdAt": metadata.createdAt,
                "chunkCount": metadata.chunkCount,
            },
            "chunks": [c.model_dump(by_alias=True) for c in self._chunks],
            "lengths": self._lengths,
            "postings": self._postings,
        }

    def search(self, query: str, top_k: int) -> List[KeywordHit]:
        """Chunks matching any query term, best BM25 score first, ties broken by chunk id."""
        n = len(self._chunks)
        if n == 0 or top_k <= 0:
            return []

        scores: Dict[int, float] = {}
        for term in dict.fromkeys(tokenize(query)):
            posting = self._postings.get(term)
            if not posting:
                continue
            df = len(posting) / 2
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for i in range(0, len(posting), 2):
                chunk = posting[i]
                tf = posting[i + 1]
                norm = 1 - BM25_B + (BM25_B * self._lengths[chunk]) / self._avg_length
                score = (idf * (tf * (BM25_K1 + 1))) / (tf + BM25_K1 * norm)
                scores[chunk] = scores.get(chunk, 0) + score

        hits = [
            KeywordHit(
                id=self._chunks[i].id,
                text=self._chunks[i].text,
                metadata=self._chunks[i].metadata,
                score=score,
            )
            for i, score in scores.items()
        ]
        hits.sort(key=lambda h: (-h.score, h.id))
        return hits[:top_k]


def keyword_file_matches(data: Dict[str, Any], metadata: IndexMetadata) -> bool:
    """True when a keyword index file was built for exactly this vector index."""
    index = data.get("index") or {}
    chunks = data.get("chunks")
    return (
        data.get("format") == KEYWORD_INDEX_FORMAT
        and data.get("tokenizer") == TOKENIZER_NAME
        and index.get("sourceHash") == metadata.sourceHash
        and index.get("createdAt") == metadata.createdAt
        and index.get("chunkCount") == metadata.chunkCount
        and isinstance(chunks, list)
        and len(chunks) == metadata.chunkCount
    )


def read_keyword_file(path: str) -> Optional[Dict[str, Any]]:
    """Read a keyword index file, or return None if it is missing or unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def write_keyword_file(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
