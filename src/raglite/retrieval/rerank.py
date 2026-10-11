"""Reranking: reorder the first ``candidates`` search results with a reranking
model, an LLM or your own function, then keep the first ``top_k``.

``rerank`` is a dict such as ``{"provider": "cohere", "model": "rerank-v3.5",
"apiKey": "...", "candidates": 30}`` (providers: ``cohere``, ``voyage``,
``llm``), or a function ``(query, hits) -> hits`` that returns the hits best
first.
"""

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Union

from ..errors import ConfigError, import_optional
from ..types import LLMProviderConfig, SearchResult, SearchScores

Reranker = Callable[[str, List[SearchResult]], List[SearchResult]]
RerankOption = Union[Dict[str, Any], Reranker]

DEFAULT_RERANK_MODELS = {"cohere": "rerank-v3.5", "voyage": "rerank-2.5"}
_DEFAULT_CANDIDATES = 30
_PROVIDERS = ("cohere", "voyage", "llm")


def rerank_candidates(rerank: RerankOption, top_k: int) -> int:
    """How many results to retrieve before reranking."""
    candidates = _DEFAULT_CANDIDATES if callable(rerank) else rerank.get("candidates", _DEFAULT_CANDIDATES)
    if not isinstance(candidates, int) or isinstance(candidates, bool) or candidates <= 0:
        raise ConfigError(f"rerank.candidates must be a positive integer, got {candidates}")
    return max(candidates, top_k)


def rerank_hits(
    query: str,
    hits: List[SearchResult],
    rerank: RerankOption,
    top_k: int,
    *,
    logger: Any,
    llm: Optional[Any] = None,
) -> List[SearchResult]:
    """Reorder ``hits`` with the reranker and keep the first ``top_k``. Each
    result's ``score`` becomes the reranker's score, kept in ``scores.rerank``
    next to the retrieval scores. A reranker that fails at run time logs a
    warning and the original order is kept; a misconfigured one raises
    ConfigError."""
    if not hits:
        return hits
    run = rerank if callable(rerank) else _built_in(rerank, llm)
    try:
        ranked = run(query, hits)
    except ConfigError:
        raise
    except Exception as err:
        logger.warn(f"Reranking failed, keeping the retrieval order: {err}")
        return hits[:top_k]
    return ranked[:top_k]


def _built_in(config: Dict[str, Any], default_llm: Optional[Any]) -> Reranker:
    provider = config.get("provider")
    if provider not in _PROVIDERS:
        raise ConfigError(
            f'Unknown rerank provider "{provider}". Use "cohere", "voyage" or "llm", or pass a function.'
        )
    if provider == "llm":
        llm = config.get("llm") or default_llm
        if not llm:
            raise ConfigError('rerank provider "llm" needs an LLM: set rerank.llm or the document\'s llm.')
        return lambda query, hits: _llm_rerank(query, hits, llm)

    model = config.get("model") or DEFAULT_RERANK_MODELS[provider]
    api_key = config.get("apiKey") or config.get("api_key")
    if provider == "cohere":
        cohere = import_optional("cohere", "cohere", "Cohere reranking")
        co = cohere.ClientV2(
            api_key=api_key or os.environ.get("COHERE_API_KEY"),
            base_url=config.get("baseURL"),
        )

        def cohere_rerank(query: str, hits: List[SearchResult]) -> List[SearchResult]:
            resp = co.rerank(model=model, query=query, documents=[h.text for h in hits])
            return [_with_rerank_score(hits[r.index], r.relevance_score) for r in resp.results]

        return cohere_rerank

    voyageai = import_optional("voyageai", "voyage", "Voyage reranking")
    vo = voyageai.Client(api_key=api_key or os.environ.get("VOYAGE_API_KEY"))

    def voyage_rerank(query: str, hits: List[SearchResult]) -> List[SearchResult]:
        resp = vo.rerank(query, [h.text for h in hits], model=model)
        return [_with_rerank_score(hits[r.index], r.relevance_score) for r in resp.results]

    return voyage_rerank


def _with_rerank_score(hit: SearchResult, score: float) -> SearchResult:
    retrieval = hit.scores.model_dump() if hit.scores else {"vector": hit.score}
    return hit.model_copy(
        update={
            "score": score,
            "distance": 1.0 - score,
            "scores": SearchScores(**{**retrieval, "rerank": score}),
        }
    )


def _llm_rerank(query: str, hits: List[SearchResult], config: Any) -> List[SearchResult]:
    """Ask the LLM to order the passages. Its score is rank-based: 1 for the
    first passage down to 1/n for the last; passages it leaves out follow in
    their retrieval order."""
    from ..llm.answer import _generate
    from ..llm.factory import create_llm

    llm = create_llm(config if isinstance(config, (dict, LLMProviderConfig)) else dict(config))
    passages = "\n\n".join(f"[{i + 1}] {' '.join(h.text.split())}" for i, h in enumerate(hits))
    text = _generate(
        llm,
        "You rank passages by how well they answer a search query.",
        f"Query: {query}\n\nPassages:\n{passages}\n\n"
        "Reply with only a JSON array of the passage numbers, most relevant first, for example [3, 1, 2].",
    )["text"]
    order = parse_ranking(text, len(hits))
    rest = [i for i in range(len(hits)) if i not in order]
    n = len(hits)
    return [_with_rerank_score(hits[i], (n - rank) / n) for rank, i in enumerate(order + rest)]


def parse_ranking(text: str, count: int) -> List[int]:
    """Zero-based passage indexes from the first JSON array of numbers in ``text``."""
    match = re.search(r"\[[\d\s,]*\]", text)
    if not match:
        raise ValueError(f"the LLM did not reply with a JSON array: {text[:200]}")
    seen: Dict[int, None] = {}
    for n in json.loads(match.group(0)):
        if isinstance(n, int) and 1 <= n <= count:
            seen.setdefault(n - 1)
    return list(seen)
