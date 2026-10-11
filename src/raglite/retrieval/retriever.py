import inspect
from typing import Any, Callable, Dict, List, Optional

from ..embeddings.base import Embedder
from ..types import SearchResult
from ..vectordb.base import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def retrieve(
        self,
        query: str,
        options: Optional[Dict[str, Any]] = None,
        *,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        opts = options or {}

        # Resolve top_k/topK
        tk = top_k
        if tk is None:
            tk = opts.get("topK")
        if tk is None:
            tk = opts.get("top_k")
        if tk is None:
            raise ValueError("top_k or topK must be specified in retrieve options")

        # Resolve score_threshold/scoreThreshold
        st = score_threshold
        if st is None:
            st = opts.get("scoreThreshold")
        if st is None:
            st = opts.get("score_threshold")
        if st is None:
            st = opts.get("score_threshold")
        threshold = st if st is not None else 0.0

        embedding = self.embedder.embed_query(query)
        # Stores written before 2.2 take no filter; Document filters their results.
        hits = (
            self.store.search(embedding, tk, filter)
            if filter and accepts_filter(self.store.search)
            else self.store.search(embedding, tk)
        )

        results = []
        for h in hits:
            if h.score >= threshold:
                results.append(
                    SearchResult(
                        id=h.id,
                        text=h.text,
                        metadata=h.metadata,
                        score=h.score,
                        distance=h.distance,
                    )
                )
        return results


def accepts_filter(method: Callable[..., Any]) -> bool:
    """True when a store method takes a ``filter`` argument."""
    try:
        return "filter" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
