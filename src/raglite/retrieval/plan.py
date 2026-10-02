from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from ..constants import DEFAULT_RRF_K, MIN_HYBRID_CANDIDATES
from ..errors import ConfigError
from ..types import HybridOptions, RetrievalOptions

_MODES = ("vector", "keyword", "hybrid")


@dataclass
class RetrievalPlan:
    """Fully resolved retrieval settings for one search call."""

    mode: str
    top_k: int
    score_threshold: float
    rrf_k: float
    candidates: int
    vector_weight: float
    keyword_weight: float


def _hybrid(value: Union[None, HybridOptions, Dict[str, Any]]) -> HybridOptions:
    if value is None:
        return HybridOptions()
    if isinstance(value, HybridOptions):
        return value
    try:
        return HybridOptions.model_validate(value)
    except Exception as err:
        raise ConfigError(f"Invalid hybrid options: {err}") from err


def resolve_retrieval_plan(
    defaults: RetrievalOptions,
    mode: Optional[str],
    hybrid: Union[None, HybridOptions, Dict[str, Any]],
    top_k: int,
    score_threshold: float,
) -> RetrievalPlan:
    """Merge per-call options over the configured defaults and validate the result."""
    resolved_mode = mode or defaults.mode or "vector"
    if resolved_mode not in _MODES:
        raise ConfigError(
            f'Unknown retrieval mode "{resolved_mode}". Use "vector", "keyword" or "hybrid".'
        )
    call = _hybrid(hybrid)
    base = defaults.hybrid or HybridOptions()

    def pick(*values: Any) -> Any:
        return next((v for v in values if v is not None), None)

    rrf_k = pick(call.rrfK, base.rrfK, DEFAULT_RRF_K)
    candidates = pick(call.candidates, base.candidates, max(MIN_HYBRID_CANDIDATES, top_k * 4))
    cw, bw = call.weights, base.weights
    vector_weight = pick(cw and cw.vector, bw and bw.vector, 1)
    keyword_weight = pick(cw and cw.keyword, bw and bw.keyword, 1)

    if not rrf_k > 0:
        raise ConfigError(f"hybrid.rrfK must be positive, got {rrf_k}")
    if not isinstance(candidates, int) or candidates <= 0:
        raise ConfigError(f"hybrid.candidates must be a positive integer, got {candidates}")
    if not (vector_weight >= 0 and keyword_weight >= 0) or vector_weight + keyword_weight == 0:
        raise ConfigError("hybrid.weights must be non-negative and not both zero")
    # Results can only come from the candidate lists, so never take fewer than top_k.
    return RetrievalPlan(
        mode=resolved_mode,
        top_k=top_k,
        score_threshold=score_threshold,
        rrf_k=rrf_k,
        candidates=max(candidates, top_k),
        vector_weight=vector_weight,
        keyword_weight=keyword_weight,
    )
