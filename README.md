# raglite-toolkit

Semantic search, cited answers, an MCP server and agent tools over your documents, in a few lines of Python.

[![PyPI](https://img.shields.io/pypi/v/raglite-toolkit)](https://pypi.org/project/raglite-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/raglite-toolkit)](https://pypi.org/project/raglite-toolkit/)
[![Docs](https://img.shields.io/badge/docs-creatorpiyush.github.io%2Fraglite-blue)](https://creatorpiyush.github.io/raglite/)
[![PR Verification](https://github.com/creatorpiyush/raglite-py/actions/workflows/pr-verify.yml/badge.svg)](https://github.com/creatorpiyush/raglite-py/actions/workflows/pr-verify.yml)
[![License: MIT](https://img.shields.io/pypi/l/raglite-toolkit)](https://github.com/creatorpiyush/raglite-py/blob/main/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/raglite-toolkit)](https://pypi.org/project/raglite-toolkit/)

**[Documentation](https://creatorpiyush.github.io/raglite/)** · [Getting started](https://creatorpiyush.github.io/raglite/getting-started/) · [Providers](https://creatorpiyush.github.io/raglite/providers/) · [MCP server](https://creatorpiyush.github.io/raglite/guides/mcp/) · [Troubleshooting](https://creatorpiyush.github.io/raglite/troubleshooting/) · [Changelog](https://github.com/creatorpiyush/raglite-py/blob/main/CHANGELOG.md)

RAGLite indexes PDFs, Word files, Markdown, text, JSON, folders and web pages, then searches them and answers questions with any LLM, citing the passages it used.

- **Answers with citations**, including the PDF page and Markdown section of each passage
- **Hybrid search**: vector search for meaning, BM25 for exact terms such as error codes, or both
- **MCP server** for Claude Code, Claude Desktop, Cursor and VS Code, and a **function-calling tool** for OpenAI and Anthropic agents
- **Any provider**: OpenAI, Anthropic, Google, Mistral, Cohere, Groq, xAI and Ollama; install only the extras you use
- **Fully local** option: local embeddings and Ollama, no API key
- Content-hash caching, Qdrant and Pinecone stores, streaming, a FastAPI REST API and a CLI

## Install

Requires Python 3.11 or later.

```bash
pip install 'raglite-toolkit[local,anthropic]'   # local embeddings (the default) + Claude answers
pip install 'raglite-toolkit[all]'               # every provider
```

## Quick start

```python
import os
from raglite import Document

doc = Document("./policy.pdf", {
    "llm": {"provider": "anthropic", "apiKey": os.environ["ANTHROPIC_API_KEY"]},
})

doc.build()

hits = doc.search("refund policy", top_k=3)

answer = doc.ask("What is the refund policy?")
print(answer.text)       # "Refunds are issued within 30 days [1]."
print(answer.citations)  # [Citation(n=1, source="policy.pdf", page=12, text="...")]
```

## Learn more

- [Collections](https://creatorpiyush.github.io/raglite/guides/collections/): index folders, files and URLs together
- [Citations](https://creatorpiyush.github.io/raglite/guides/citations/): pages and sections for each cited passage
- [Hybrid search](https://creatorpiyush.github.io/raglite/guides/hybrid-search/): vector, keyword and hybrid retrieval
- [MCP server](https://creatorpiyush.github.io/raglite/guides/mcp/): give your documents to AI coding tools
- [Agent tools](https://creatorpiyush.github.io/raglite/guides/agent-tools/): `as_tool()` for OpenAI and Anthropic function calling
- [Streaming, progress and cancellation](https://creatorpiyush.github.io/raglite/guides/streaming/)
- [Vector stores](https://creatorpiyush.github.io/raglite/guides/vector-stores/): Qdrant, Pinecone or your own
- [REST API](https://creatorpiyush.github.io/raglite/guides/rest-api/) and [CLI](https://creatorpiyush.github.io/raglite/guides/cli/)
- [How caching works](https://creatorpiyush.github.io/raglite/guides/caching/)
- [Options and types](https://creatorpiyush.github.io/raglite/reference/options/)

Choose the **Python** tab on any page; the choice is remembered.

## TypeScript

The same library for TypeScript: [`raglite-toolkit` on npm](https://www.npmjs.com/package/raglite-toolkit) ([source](https://github.com/creatorpiyush/raglite)).

## Contributing

See [CONTRIBUTING.md](https://github.com/creatorpiyush/raglite-py/blob/main/CONTRIBUTING.md). Security issues: [SECURITY.md](https://github.com/creatorpiyush/raglite-py/blob/main/SECURITY.md).

## License

MIT © [Piyush Anand](https://github.com/creatorpiyush)
