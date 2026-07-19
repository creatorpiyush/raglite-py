from typing import Any, Dict, List, Optional

from ..types import SearchResult


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
            tag = f"[{index + 1}] ({chunk.metadata.source} #{chunk.metadata.chunk})"
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


# Aliases for TS parity
buildSystemPrompt = build_system_prompt
buildUserPrompt = build_user_prompt
