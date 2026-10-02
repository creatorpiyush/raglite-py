from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Sequence

from ..types import SearchResult, SearchScores


@dataclass
class RankedList:
    name: Literal["vector", "keyword"]
    weight: float
    hits: Sequence[Any]  # best first; each has id, text, metadata and score


def reciprocal_rank_fusion(
    lists: Sequence[RankedList], rrf_k: float, top_k: int
) -> List[SearchResult]:
    """Reciprocal Rank Fusion: ``fused(d) = Σ weight / (rrf_k + rank(d))`` over
    the lists, with ranks starting at 1. ``score`` is the fused score divided
    by the highest possible one, so 1 means "ranked first by every list", and
    ``scores`` keeps each list's own score."""
    active = [lst for lst in lists if lst.weight > 0]
    max_fused = sum(lst.weight / (rrf_k + 1) for lst in active)
    if max_fused == 0:
        return []

    fused: Dict[str, Dict[str, Any]] = {}
    for lst in active:
        for i, hit in enumerate(lst.hits):
            entry = fused.get(hit.id)
            if entry is None:
                entry = {"hit": hit, "fused": 0.0, "scores": {}}
                fused[hit.id] = entry
            entry["fused"] += lst.weight / (rrf_k + i + 1)
            entry["scores"][lst.name] = hit.score

    results: List[SearchResult] = []
    for entry in fused.values():
        hit = entry["hit"]
        score = entry["fused"] / max_fused
        results.append(
            SearchResult(
                id=hit.id,
                text=hit.text,
                metadata=hit.metadata,
                score=score,
                distance=1 - score,
                scores=SearchScores(**entry["scores"], fused=entry["fused"]),
            )
        )
    results.sort(key=lambda r: (-r.score, r.id))
    return results[:top_k]
