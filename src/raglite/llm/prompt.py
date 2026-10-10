import re
from typing import Any, Dict, List, Optional

from ..types import Citation, SearchResult


def build_system_prompt(
    options: Optional[Dict[str, Any]] = None,
    *,
    system_hint: Optional[str] = None,
) -> str:
    opts = options or {}
    hint = system_hint
    if hint is None:
        hint = opts.get("systemHint")
    if hint is None:
        hint = opts.get("system_hint")

    base = (
        "You are a precise assistant that answers questions strictly from the provided context. "
        "If the answer is not contained in the context, respond exactly: "
        '"I could not find the answer in the provided documents."'
    )
    return f"{base}\n\n{hint}" if hint else base


def build_user_prompt(
    question: str,
    context: List[SearchResult],
    options: Optional[Dict[str, Any]] = None,
    *,
    include_citations: Optional[bool] = None,
) -> str:
    opts = options or {}
    inc_cit = include_citations
    if inc_cit is None:
        inc_cit = opts.get("includeCitations")
    if inc_cit is None:
        inc_cit = opts.get("include_citations")
    if inc_cit is None:
        inc_cit = True

    parts = []
    for index, chunk in enumerate(context):
        if inc_cit:
            tag = f"[{index + 1}] ({_describe_source(chunk)})"
        else:
            tag = f"[{index + 1}]"
        parts.append(f"{tag}\n{chunk.text}")

    context_block = "\n\n---\n\n".join(parts)
    citation_instruction = (
        "\n\nCite the passages you used with their bracketed numbers, e.g. [1], [2]."
        if inc_cit
        else ""
    )

    return f"Context:\n{context_block}\n\nQuestion: {question}{citation_instruction}\n\nAnswer:"


def _describe_source(chunk: SearchResult) -> str:
    """"policy.pdf #3, p. 12, Refunds": what the model sees next to each passage number."""
    meta = chunk.metadata
    label = f"{meta.source} #{meta.chunk}"
    page, page_end = getattr(meta, "page", None), getattr(meta, "pageEnd", None)
    if page is not None:
        label += f", pp. {page}-{page_end}" if page_end is not None else f", p. {page}"
    section = getattr(meta, "section", None)
    if section:
        label += f", {section}"
    return label


_CITATION = re.compile(r"\[([0-9]+(?:[ \t]*,[ \t]*[0-9]+)*)\]")


def extract_citations(answer: str, context: List[SearchResult]) -> List[Citation]:
    """The passages cited in an answer, in order of first citation. Understands
    "[2]", "[1][3]" and "[1, 3]"; numbers outside the context are ignored."""
    seen = set()
    citations: List[Citation] = []
    for match in _CITATION.finditer(answer):
        for part in match.group(1).split(","):
            n = int(part.strip())
            if not 1 <= n <= len(context) or n in seen:
                continue
            seen.add(n)
            chunk = context[n - 1]
            meta = chunk.metadata
            citations.append(
                Citation(
                    n=n,
                    source=meta.source,
                    chunk=meta.chunk,
                    page=getattr(meta, "page", None),
                    pageEnd=getattr(meta, "pageEnd", None),
                    section=getattr(meta, "section", None),
                    text=chunk.text,
                )
            )
    return citations


# Aliases for TS parity
buildSystemPrompt = build_system_prompt
buildUserPrompt = build_user_prompt
