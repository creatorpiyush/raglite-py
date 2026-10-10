# raglite-toolkit

> Build semantic search, multi-provider question answering, and REST APIs over your documents, directories, or web URLs in a few lines of Python.

**raglite-toolkit** is a Python port of the [`raglite-toolkit`](https://github.com/creatorpiyush/raglite) TypeScript package with full 1:1 feature parity.

---

## Features

- 📄 **PDF, TXT, JSON, Markdown, DOCX** loaders out of the box
- 📁 **Multi-document, directory, & URL ingestion** — index folders, glob patterns, or web URLs with `DocumentCollection`
- 🤖 **Multi-provider LLMs** — OpenAI, Anthropic (Claude), Google (Gemini), Mistral, Cohere, Groq, xAI, Ollama
- 🔢 **Multi-provider embeddings** — OpenAI, Google, Mistral, Cohere, Voyage, Ollama, or a **local** offline sentence-transformer (no API key needed)
- 📐 **Cosine similarity** scoring with L2-normalized vectors
- 🔎 **Hybrid search** — BM25 keyword search fused with vector search, for exact terms like error codes and SKUs
- 🌏 **Any language** — Chinese, Japanese, Korean and Thai text is chunked and keyword-indexed correctly
- ♻️ **Content-hash cache** — reindexes only when the file actually changes
- 🗂 **Per-document namespacing** — indexes are isolated, two documents never collide
- 🌐 **REST API** via FastAPI with optional **bearer-token auth**
- 📑 **Citations** with page numbers and section headings
- 🤝 **MCP server** and **function-calling tool** for agents
- ⚡ **Streaming** answers
- 🐍 **Python-native** — Pydantic models, type-annotated, fully testable

---

## Install

Requires Python 3.11 or later.

The core install is small. Add an extra for each provider you use; the [Supported Providers](#supported-providers) tables list the extra for each one:

```bash
pip install 'raglite-toolkit[openai,anthropic]'   # OpenAI embeddings + Claude answers
pip install 'raglite-toolkit[local]'              # local offline embeddings (the default), no API key
pip install 'raglite-toolkit[all]'                # every provider
```

If an extra is missing, RAGLite raises a `ConfigError` that names the `pip install` command to run. The `local` extra installs `sentence-transformers` (with PyTorch); the `all-MiniLM-L6-v2` model (~90 MB) is downloaded on first use.

---

## Quick Start

```python
from raglite import Document

doc = Document("./policy.pdf", {
    "embeddings": {"provider": "openai", "apiKey": "sk-..."},
    "llm":        {"provider": "anthropic", "apiKey": "sk-ant-..."},
})

doc.build()                              # chunk → embed → persist

hits = doc.search("refund policy", top_k=3)

answer = doc.ask("What is the refund policy?")
print(answer.text)
```

---

## Index Text You Already Have

No file needed: index a database row, an upload or CMS content with `Document.from_text(id, text)`. The `id` names the index (reuse it to reuse the index) and becomes each chunk's `source`. Changed text is re-indexed on the next `build()`.

```python
doc = Document.from_text("faq-42", faq_text, {"embeddings": {"provider": "openai"}})
doc.build()
```

---

## Multi-Document & Directory Ingestion (`DocumentCollection`)

Index entire directories (`./docs`), glob patterns, web URLs, or mixed file arrays:

```python
from raglite import DocumentCollection

collection = DocumentCollection(["./docs", "https://example.com"], {
    "embeddings": {"provider": "local"},
    "llm": {"provider": "openai", "apiKey": "sk-..."},
})

# Index all documents concurrently
result = collection.build()
print(f"Indexed {result.totalDocuments} document(s), {result.totalChunks} chunk(s).")

# Search across all collection documents simultaneously
hits = collection.search("refund policy", top_k=5)

# Contextual Q&A across the entire collection
answer = collection.ask("What is the refund policy?")
print(answer.text)
```

---

## Hybrid Search (Keyword + Vector)

Vector search matches meaning, but it can miss exact terms such as error codes, SKUs, function names or rare product names. Keyword search (BM25) finds those exactly. `hybrid` runs both and merges the results with Reciprocal Rank Fusion.

```python
doc.search("ERR_4021", mode="hybrid")  # "vector" (default) | "keyword" | "hybrid"
doc.ask("What does ERR_4021 mean?", {"mode": "hybrid"})

# Or set a default once; it applies to search(), ask() and ask_stream().
Document("./runbook.md", {
    "retrieval": {
        "mode": "hybrid",
        "hybrid": {"rrfK": 60, "candidates": 50, "weights": {"vector": 1, "keyword": 1}},
    },
})
```

- In `keyword` and `hybrid` modes, `score` is the fused rank score scaled to 0..1 (1 means ranked first by every retriever). Each result also has `scores` with `vector`, `keyword` and `fused`. Vector mode results are unchanged.
- `scoreThreshold` is still a cosine similarity. In hybrid mode it filters the vector results before fusion; keyword matches are not filtered by it.
- The keyword index is built by `build()` from the chunk texts, with no extra embedding calls, and saved as `<storeDir>/<namespace>/keyword.json`. It works with every vector store. With Qdrant or Pinecone, keep `storeDir` on persistent disk.
- Indexes built before 1.3: the memory and Qdrant stores build the keyword index from the stored chunks on the first keyword or hybrid search. Other stores log a warning and use vector search until you run `build(rebuild=True)`.
- The same options work over HTTP (`"mode"` on `/search` and `/ask`) and in the CLI (`--mode hybrid`).

The tokenizer is the same in the Python and TypeScript SDKs:

| Text | How it is indexed |
|------|-------------------|
| Latin, Cyrillic, Greek, Arabic, Devanagari and other spaced scripts | Words, lowercased and NFKC-normalised (`ﬁ` → `fi`, `ＡＢＣ` → `abc`) |
| Code identifiers | `gpt-4.1`, `snake_case` and `ERR_42` stay whole, and their parts are indexed too |
| Chinese, Japanese, Korean, Thai, Lao, Khmer, Myanmar | Overlapping character pairs (`退款处理` → `退款`, `款处`, `处理`), so no dictionary is needed |

There is no stemming or stopword list, because both are language-specific: in keyword mode `refund` does not match `refunds`. Hybrid mode's vector side covers those cases.

---

## Fully Offline — No API Key Needed

Install `pip install 'raglite-toolkit[local,ollama]'`, then:

```python
from raglite import Document

doc = Document("./manual.txt", {
    "embeddings": {"provider": "local"},
    "llm":        {"provider": "ollama", "model": "llama3.2"},
})

doc.build()
print(doc.ask("How do I reset the device?").text)
```

---

## Citations

`ask()` returns the passages the answer cites, with the page (PDF) and heading path (Markdown) of each:

```python
answer = doc.ask("What is the refund policy?")
answer.text       # "Refunds are issued within 30 days [1]."
answer.citations  # [Citation(n=1, source="policy.pdf", chunk=4, page=12, section="Policies > Refunds", text="...")]
```

- `n` is the number used in the answer. `[2]`, `[1][3]` and `[1, 3]` are all recognised; numbers that match no passage are ignored.
- Search results carry the same `page`, `pageEnd` (when a chunk crosses a page break) and `section` in `metadata`.
- Indexes built before 2.1 have no page or section until you run `build(rebuild=True)`. Answers still get `citations`, without those fields.
- The REST `/ask` response includes `citations`, and `raglite ask` lists the sources under the answer.

## Use from AI Agents (MCP)

`raglite mcp` serves your documents to any MCP client (Claude Code, Claude Desktop, Cursor, VS Code) over stdio:

```bash
pip install 'raglite-toolkit[mcp,local]'
claude mcp add docs -- raglite mcp /absolute/path/to/docs --store-dir /absolute/path/to/docs/.raglite
```

Claude Desktop, Cursor and VS Code use the same command in their MCP settings:

```json
{
  "mcpServers": {
    "docs": {
      "command": "raglite",
      "args": ["mcp", "/absolute/path/to/docs", "--store-dir", "/absolute/path/to/docs/.raglite"]
    }
  }
}
```

- Use absolute paths, and pass `--store-dir`: MCP clients may start the server in any directory, and the index is kept in `./.raglite` by default. If `raglite` is not on the client's `PATH`, use the full path to it (for example `/path/to/.venv/bin/raglite`).
- The first start indexes the documents, which can take longer than a client waits. Run the same command once in a terminal first (stop it with Ctrl+C); later starts reuse the index.

Tools: `search` (`query`, optional `topK` and `mode`), `list_sources`, and `ask` when you pass `--llm-provider`. The tools match the TypeScript SDK's. From code: `serve_mcp(collection)`, or `create_mcp_server(collection)` for other transports.

## Use as an Agent Tool

`as_tool()` gives you a callable plus its tool definition for OpenAI or Anthropic function calling:

```python
tool = doc.as_tool(name="search_policies", description="Search the company policies.", top_k=5)

response = client.messages.create(model=..., tools=[tool.anthropic_tool], messages=[...])  # or tool.openai_tool
# When the model calls it:
result = tool(**tool_use.input)   # JSON string of passages: source, chunk, page, section, score, text
```

## Progress and Cancellation

```python
import threading

cancel = threading.Event()
collection.build(
    on_progress=lambda p: print(f"{p['source']}: {p['embedded']}/{p['total']}"),
    cancel=cancel,   # cancel.set() from another thread stops the build
)
```

Chunks are embedded in batches of 64, with `on_progress` called after each. A cancelled build raises `BuildCancelledError` and keeps the previous index, because the old index is replaced only once embedding finishes.

## Choose Any LLM at Ask-Time

```python
# Pass an inline LLM override to ask()
gpt4 = doc.ask("Summarise this document", options={
    "llm": {"provider": "openai", "model": "gpt-4o", "apiKey": "sk-..."}
})

claude = doc.ask("Summarise this document", options={
    "llm": {"provider": "anthropic", "model": "claude-sonnet-5-5", "apiKey": "sk-ant-..."}
})
```

---

## Streaming Responses

```python
for chunk in doc.ask_stream("Explain section 3 in detail"):
    print(chunk, end="", flush=True)
print()
```

---

## Pluggable Vector Databases

`raglite` supports pluggable vector stores (Memory, Qdrant, Pinecone, or custom subclasses; LanceDB is currently TypeScript-only):

### Memory Store (Default)
```python
doc = Document("./policy.pdf", {
    "vectorStore": {"provider": "memory", "storeDir": ".raglite"}
})
```

### Qdrant Store
By default each document gets its own collection (`raglite_<namespace>`). Set `indexName` to keep several documents in one shared collection; each document's points are tagged with its namespace, so rebuilding one document never affects the others.
```python
doc = Document("./policy.pdf", {
    "vectorStore": {
        "provider": "qdrant",
        "url": "http://localhost:6333",
        "apiKey": "your-key",
        "indexName": "my_collection"
    }
})
```

### Pinecone Store
```python
doc = Document("./policy.pdf", {
    "vectorStore": {
        "provider": "pinecone",
        "url": "https://my-index.svc.pinecone.io",
        "apiKey": "your-key"
    }
})
```

---

## REST API

```python
from raglite import Document

doc = Document("./policy.pdf", {
    "embeddings": {"provider": "local"},
    "llm":        {"provider": "openai", "apiKey": "sk-..."},
})
doc.build()

# Start background FastAPI server on port 8085
doc.serve(port=8085, bearer_token="secret-token")
```

Endpoints:

| Method | Path | Auth required? | Description |
| ------ | ---- | -------------- | ----------- |
| `GET`  | `/health` | ❌ | Liveness + index stats |
| `GET`  | `/info`   | ✅ | Configuration snapshot |
| `POST` | `/search` | ✅ | Semantic search |
| `POST` | `/ask`    | ✅ | Question answering (supports `stream: true`) |

### Example `curl` calls

```bash
# Health check (no auth)
curl http://127.0.0.1:8085/health

# Search
curl -X POST http://127.0.0.1:8085/search \
  -H 'Authorization: Bearer secret-token' \
  -H 'Content-Type: application/json' \
  -d '{"query": "refund policy", "topK": 3}'

# Ask (non-streaming)
curl -X POST http://127.0.0.1:8085/ask \
  -H 'Authorization: Bearer secret-token' \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the refund policy?"}'

# Ask (streaming)
curl -X POST http://127.0.0.1:8085/ask \
  -H 'Authorization: Bearer secret-token' \
  -H 'Content-Type: application/json' \
  -d '{"question": "Summarize the document", "stream": true}'
```

---

## CLI

```bash
# Index a document, directory, or URL
raglite index ./policy.pdf --embed-provider local

# Semantic search
raglite search ./docs "refund policy" --top-k 5
raglite search ./docs "ERR_4021" --mode hybrid

# Ask a question (streaming)
raglite ask ./docs "What is the refund policy?" \
  --llm-provider anthropic --llm-key $ANTHROPIC_API_KEY --stream

# MCP server over stdio, for AI agents
raglite mcp ./docs --llm-provider anthropic

# Serve a REST API
raglite serve https://example.com \
  --llm-provider openai --llm-key $OPENAI_API_KEY \
  --port 8085 --token $RAGLITE_TOKEN
```

---

## Supported Providers

### LLMs

| Provider       | `provider` key | Default model | Extra |
|----------------|----------------|---------------|-------|
| OpenAI         | `openai`       | `gpt-4o-mini` | `[openai]` |
| Anthropic      | `anthropic`    | `claude-sonnet-5-5` | `[anthropic]` |
| Google         | `google`       | `gemini-3.8-flash` | `[google]` |
| Mistral        | `mistral`      | `mistral-large-latest` | `[mistral]` |
| Cohere         | `cohere`       | `command-a-03-2025` | `[cohere]` |
| Groq           | `groq`         | `openai/gpt-oss-120b` | `[groq]` |
| xAI (Grok)     | `xai`          | `grok-4.7` | `[xai]` |
| Ollama (local) | `ollama`       | `llama3.2` | `[ollama]` |

Anthropic requests send no `temperature` unless you set one, because Claude 5 models reject non-default sampling values.

### Embeddings

| Provider       | `provider` key | Default model | Extra |
|----------------|----------------|---------------|-------|
| OpenAI         | `openai`       | `text-embedding-3-small` | `[openai]` |
| Google         | `google`       | `gemini-embedding-2` | `[google]` |
| Mistral        | `mistral`      | `mistral-embed` | `[mistral]` |
| Cohere         | `cohere`       | `embed-english-v3.0` | `[cohere]` |
| Voyage         | `voyage`       | `voyage-3` | `[voyage]` |
| Ollama (local) | `ollama`       | `nomic-embed-text` | `[ollama]` |
| Local (offline)| `local`        | `all-MiniLM-L6-v2` | `[local]` |

The Qdrant and Pinecone stores need no extra.

---

## Configuration Reference

```python
Document("./policy.pdf", {
    # Chunking
    "chunkSize":      500,          # words per chunk; characters for Chinese, Japanese, Thai, ... (default: 500)
    "overlap":        50,           # overlapping words between chunks (default: 50)

    # Retrieval
    "topK":           5,            # default results returned (default: 5)
    "scoreThreshold": 0.0,          # minimum cosine similarity (0..1, default: 0)
    "retrieval":      {"mode": "vector"},  # "vector" | "keyword" | "hybrid" (default: vector)

    # Storage
    "storeDir":       ".raglite",   # where indexes are persisted (default: .raglite)

    # Providers
    "embeddings": {"provider": "local"},
    "llm":        {"provider": "openai", "model": "gpt-4o-mini", "apiKey": "sk-..."},

    # Logging
    "logLevel":   "info",           # "silent" | "info" | "debug" (default: info)
})
```

---

## How Caching Works

Every `build()` call fingerprints the source (file bytes, or the fetched text for URLs) with a **SHA-256 content hash** and persists it alongside the vectors. The cached index is reused only if **all** of the following match the stored index:

| Factor | Triggers rebuild if changed |
|--------|-----------------------------|
| File content | SHA-256 hash differs |
| Chunk size | `chunkSize` changed (when not set, the existing index's value is kept) |
| Overlap | `overlap` changed (when not set, the existing index's value is kept) |
| Embedding provider/model | Provider or model string changed (when no `embeddings` are configured, the existing index's are kept) |
| Index format | Stored index layout changed by a release (rare; ordinary upgrades reuse the index) |

Pass `rebuild=True` to `build()` to force a fresh index regardless.

Each document is stored under `.raglite/<sha256-prefix>/`, so multiple documents in the same project never overwrite each other.

---

## Advanced Usage

### Custom Vector Store

```python
from raglite.vectordb.base import VectorStore

class MyVectorStore(VectorStore):
    # Implement: namespace (property), load, reset, add, search, count,
    #            save_index_metadata, read_index_metadata
    # Optional:  list_chunks() lets hybrid search rebuild a missing keyword index
    #            from your store; keyword_search(query, top_k) replaces the
    #            built-in BM25 index with your own.
    ...
```

### Custom Loader

```python
from raglite.loaders.base import BaseLoader
from raglite.loaders import get_loader

class CsvLoader(BaseLoader):
    def load(self) -> str:
        # read CSV, return string
        ...
```

### Custom Chunker

```python
from raglite.chunking.base import BaseChunker

class SentenceChunker(BaseChunker):
    def split(self, text: str) -> list[str]:
        ...
```

### Direct Embedder Access

```python
from raglite import create_embedder

embedder = create_embedder({"provider": "openai", "apiKey": "sk-..."})
vectors = embedder.embed_documents(["chunk one", "chunk two"])
query_vec = embedder.embed_query("refund policy")
```

---

## Development

```bash
# Clone and set up
git clone https://github.com/creatorpiyush/raglite-py.git
cd raglite-py

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run test suite
pytest

# Run with coverage
pytest --cov=raglite --cov-report=term-missing

# Live call per provider whose API key is set (see the script for the variables)
python scripts/smoke_providers.py

# Run examples
python examples/basic.py
python examples/serve.py
```

### Test Structure

```
tests/
├── unit/
│   ├── test_chunking.py           # RecursiveChunker algorithm
│   ├── test_vectordb.py           # MemoryVectorStore (cosine, persistence, isolation)
│   ├── test_loaders.py            # TxtLoader, MarkdownLoader, JsonLoader
│   ├── test_directory_loader.py   # DirectoryLoader (recursive scanning, glob filtering)
│   ├── test_web_loader.py         # WebLoader (HTML parsing, tag stripping)
│   ├── test_prompt.py             # system/user prompt builders
│   ├── test_errors.py             # exception hierarchy
│   ├── test_config.py             # config defaults and overrides
│   ├── test_hash.py               # SHA-256 file hashing + namespace generation
│   ├── test_retriever.py          # Retriever with mocked embedder
│   └── test_cli.py                # CLI commands and argument parsing
└── integration/
    ├── test_document.py           # build/cache/search lifecycle (mocked embeddings)
    ├── test_collection.py         # DocumentCollection multi-document indexing & FastAPI server
    ├── test_ask.py                # ask/stream with mocked LLM generation
    └── test_api.py                # FastAPI endpoints via TestClient
```

---

## License

MIT © [Piyush Anand](https://github.com/creatorpiyush)
