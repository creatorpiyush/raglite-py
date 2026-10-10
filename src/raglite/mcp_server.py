"""MCP server exposing an indexed document or collection to agents.

Same tools and inputs as the TypeScript SDK: ``search``, ``ask`` (only when an LLM
is configured) and ``list_sources``. Needs the ``[mcp]`` extra.
"""

import json
import os
from typing import Annotated, Any, Dict, Literal, Optional, Union

from pydantic import Field

from .constants import PACKAGE_VERSION
from .core.collection import DocumentCollection
from .core.document import Document
from .errors import import_optional
from .loaders import is_url
from .tools import to_passage

McpTarget = Union[Document, DocumentCollection]

_PURPOSE = "The MCP server"

TopK = Annotated[
    Optional[int], Field(ge=1, le=50, description="Number of passages to return (default 5).")
]
Mode = Annotated[
    Optional[Literal["vector", "keyword", "hybrid"]],
    Field(description="vector: by meaning. keyword: exact terms such as error codes. hybrid: both."),
]


def create_mcp_server(target: McpTarget, llm: Optional[Dict[str, Any]] = None) -> Any:
    """Build the server. Call ``build()`` on the target first.

    ``llm`` is used by the ``ask`` tool and defaults to the target's ``llm``;
    without one, ``ask`` is not offered.
    """
    mcpserver = import_optional("mcp.server.mcpserver", "mcp", _PURPOSE)
    server = mcpserver.MCPServer(name="raglite", version=PACKAGE_VERSION)

    def search(
        query: Annotated[str, Field(min_length=1, description="What to look for.")],
        topK: TopK = None,
        mode: Mode = None,
    ) -> str:
        hits = target.search(query, _retrieval_options(topK, mode))
        return json.dumps([to_passage(h) for h in hits], indent=2)

    server.add_tool(
        search,
        name="search",
        title="Search documents",
        description=(
            "Find the passages in the indexed documents most relevant to a query. "
            "Returns JSON: source, chunk, score and text for each passage."
        ),
    )

    llm_config = llm or target.config.llm
    if llm_config:

        def ask(
            question: Annotated[str, Field(min_length=1, description="The question to answer.")],
            topK: TopK = None,
            mode: Mode = None,
        ) -> str:
            answer = target.ask(question, {"llm": llm_config, **_retrieval_options(topK, mode)})
            return answer.text

        server.add_tool(
            ask,
            name="ask",
            title="Ask the documents",
            description=(
                "Answer a question using only the indexed documents. "
                "The answer cites passages as [1], [2], ..."
            ),
        )

    def list_sources() -> str:
        docs = target.get_documents() if isinstance(target, DocumentCollection) else [target]
        return json.dumps(
            [
                {
                    "source": d.file_path if is_url(d.file_path) else os.path.basename(d.file_path),
                    "path": d.file_path,
                    "chunks": d.chunk_count,
                }
                for d in docs
            ],
            indent=2,
        )

    server.add_tool(
        list_sources,
        name="list_sources",
        title="List indexed sources",
        description="List the indexed documents and how many passages each has.",
    )
    return server


def serve_mcp(target: McpTarget, llm: Optional[Dict[str, Any]] = None) -> None:
    """Serve over stdio until the client disconnects. stdout carries the protocol."""
    create_mcp_server(target, llm).run("stdio")


def _retrieval_options(top_k: Optional[int], mode: Optional[str]) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    if top_k is not None:
        options["topK"] = top_k
    if mode is not None:
        options["mode"] = mode
    return options
