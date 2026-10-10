"""LLM function-calling tool over a document or collection; see Document.as_tool()."""

import json
from typing import Any, Callable, Dict, List, Optional

from .types import SearchResult

_DEFAULT_DESCRIPTION = (
    "Search the indexed documents for passages relevant to a query. "
    "Returns each passage's source and text."
)


def to_passage(hit: SearchResult) -> Dict[str, Any]:
    """A retrieved passage as tools return it: where it came from, and its text."""
    meta = hit.metadata
    passage: Dict[str, Any] = {"source": meta.source, "chunk": meta.chunk}
    for key in ("page", "section"):
        value = getattr(meta, key, None)
        if value is not None:
            passage[key] = value
    passage["score"] = round(hit.score, 4)
    passage["text"] = hit.text
    return passage


class SearchTool:
    """A search function plus its tool definition for OpenAI or Anthropic function calling.

    Pass ``tool.openai_tool`` or ``tool.anthropic_tool`` in the request's tools, and when
    the model calls it, reply with ``tool(**arguments)`` (a JSON string of passages).
    """

    def __init__(self, search: Callable[[str], List[SearchResult]], name: str, description: str):
        self.name = name
        self.description = description
        self._search = search

    @property
    def parameters(self) -> Dict[str, Any]:
        """JSON Schema of the tool's input."""
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "description": "What to look for."}
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    @property
    def openai_tool(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @property
    def anthropic_tool(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.parameters}

    def __call__(self, query: str) -> str:
        return json.dumps([to_passage(h) for h in self._search(query)])


def search_tool(
    target: Any,
    name: str = "search_documents",
    description: Optional[str] = None,
    top_k: Optional[int] = None,
    mode: Optional[str] = None,
) -> SearchTool:
    options: Dict[str, Any] = {}
    if top_k is not None:
        options["topK"] = top_k
    if mode is not None:
        options["mode"] = mode
    return SearchTool(
        lambda query: target.search(query, dict(options)),
        name,
        description or _DEFAULT_DESCRIPTION,
    )
