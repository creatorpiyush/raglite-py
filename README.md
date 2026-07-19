# raglite-toolkit

> Build semantic search, multi-provider question answering, and REST APIs over your documents in a few lines of Python.

**raglite-toolkit** is a Python port of the [`raglite-toolkit`](https://github.com/creatorpiyush/raglite) TypeScript package with full feature parity.

---

## Features

- 📄 **PDF, TXT, JSON, Markdown, DOCX** loaders out of the box
- 🤖 **Multi-provider LLMs** — OpenAI, Anthropic (Claude), Google (Gemini), Mistral, Cohere, Groq, xAI, Ollama
- 🔢 **Multi-provider embeddings** — OpenAI, Google, Mistral, Cohere, Voyage, Ollama, or a **local** offline sentence-transformer (no API key needed)
- 📐 **Cosine similarity** scoring with L2-normalized vectors
- ♻️ **Content-hash cache** — reindexes only when the file actually changes
- 🗂 **Per-document namespacing** — indexes are isolated, two documents never collide
- 🌐 **REST API** via FastAPI with optional **bearer-token auth**
- ⚡ **Streaming** answers
- 🐍 **Python-native** — Pydantic models, type-annotated, fully testable

---

## Install

```bash
pip install raglite-toolkit
```

For **local offline embeddings** (no API key required):

```bash
pip install raglite-toolkit sentence-transformers
```

> `sentence-transformers` is included by default. The `all-MiniLM-L6-v2` model (~90 MB) is downloaded automatically on first use.

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

## Fully Offline — No API Key Needed

```python
from raglite import Document

doc = Document("./policy.pdf", {
    "embeddings": {"provider": "local"},          # sentence-transformers offline
    "llm":        {"provider": "ollama",          # local Ollama instance
                   "model": "llama3.2",
                   "baseURL": "http://localhost:11434/api"},
})

doc.build()
print(doc.ask("What is the refund policy?").text)
```

---

## Pluggable Vector Databases

By default, `raglite` uses a lightweight in-memory vector store that persists indexes locally to JSON files. You can swap it out for a local or cloud-backed vector database by passing `vectorStore` in the options dict.

### Memory Store (Default)

```python
doc = Document("./policy.pdf", {
    "vectorStore": {"provider": "memory", "storeDir": ".raglite"},
})
```

### Qdrant (Local or Cloud)

Supports local Docker instances and Qdrant Cloud clusters.

```python
doc = Document("./policy.pdf", {
    "embeddings": {"provider": "local"},
    "vectorStore": {
        "provider": "qdrant",
        "url": "http://localhost:6333",   # or Qdrant Cloud URL
        "apiKey": "your-qdrant-api-key",  # optional
        "indexName": "my_collection",     # optional
    },
})
```

### Pinecone (Cloud)

Scopes collections using Pinecone's native namespaces.

```python
doc = Document("./policy.pdf", {
    "embeddings": {"provider": "openai", "apiKey": "sk-..."},
    "vectorStore": {
        "provider": "pinecone",
        "url": "https://your-index-host.svc.pinecone.io",
        "apiKey": "your-pinecone-api-key",
    },
})
```

### Custom Vector Store Class

Implement the `VectorStore` ABC and pass an instance directly.

```python
from raglite.vectordb.base import VectorStore, VectorSearchHit
from raglite.types import IndexMetadata, StoredChunk

class MyStore(VectorStore):
    # Implement: load, reset, add, search, count,
    #            save_index_metadata, read_index_metadata
    ...

doc = Document("./policy.pdf", {
    "vectorStore": MyStore(),
})
```



## Choose Any LLM at Ask-Time

```python
# Switch LLMs without rebuilding the index — embeddings are reused
for llm in [
    {"provider": "openai",    "model": "gpt-4o",                   "apiKey": "sk-..."},
    {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022","apiKey": "sk-ant-..."},
    {"provider": "google",    "model": "gemini-2.0-flash",         "apiKey": "AI..."},
    {"provider": "groq",      "model": "llama-3.3-70b-versatile",  "apiKey": "gsk_..."},
]:
    answer = doc.ask("Summarize this document", {"llm": llm})
    print(f"[{llm['provider']}] {answer.text[:120]}")
```

---

## Streaming

```python
for chunk in doc.ask_stream("Explain the introduction"):
    print(chunk, end="", flush=True)
```

---

## REST API

```python
handle = doc.serve({
    "port": 8085,
    "llm":  {"provider": "openai", "apiKey": "sk-..."},
    "bearerToken": "my-secret-token",
})
print(f"Serving on {handle.url}")

# ... later
handle.close()
```

### Endpoints

| Method | Path      | Auth? | Description |
|--------|-----------|-------|-------------|
| `GET`  | `/health` | ❌    | Liveness + index stats |
| `GET`  | `/info`   | ✅    | Configuration snapshot |
| `POST` | `/search` | ✅    | Semantic search |
| `POST` | `/ask`    | ✅    | Question answering |

### Example `curl` calls

```bash
# Health check (no auth)
curl http://127.0.0.1:8085/health

# Search
curl -X POST http://127.0.0.1:8085/search \
  -H 'Authorization: Bearer my-secret-token' \
  -H 'Content-Type: application/json' \
  -d '{"query": "refund policy", "topK": 3}'

# Ask (non-streaming)
curl -X POST http://127.0.0.1:8085/ask \
  -H 'Authorization: Bearer my-secret-token' \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the refund policy?"}'

# Ask (streaming)
curl -X POST http://127.0.0.1:8085/ask \
  -H 'Authorization: Bearer my-secret-token' \
  -H 'Content-Type: application/json' \
  -d '{"question": "Summarize the document", "stream": true}'
```

---

## CLI

```bash
# Index a document
raglite index ./policy.pdf --embed-provider local

# Semantic search
raglite search ./policy.pdf "refund policy" --top-k 5

# Ask a question (streaming)
raglite ask ./policy.pdf "What is the refund policy?" \
  --llm-provider openai --llm-key $OPENAI_API_KEY --stream

# Serve a REST API
raglite serve ./policy.pdf \
  --llm-provider openai --llm-key $OPENAI_API_KEY \
  --port 8085 --token $RAGLITE_TOKEN
```

---

## Supported Providers

### LLMs

| Provider       | `provider` key | Default model |
|----------------|----------------|---------------|
| OpenAI         | `openai`       | `gpt-4o-mini` |
| Anthropic      | `anthropic`    | `claude-3-5-sonnet-20241022` |
| Google         | `google`       | `gemini-2.0-flash` |
| Mistral        | `mistral`      | `mistral-large-latest` |
| Cohere         | `cohere`       | `command-r-plus` |
| Groq           | `groq`         | `llama-3.3-70b-versatile` |
| xAI (Grok)     | `xai`          | `grok-2-latest` |
| Ollama (local) | `ollama`       | `llama3.2` |

### Embeddings

| Provider       | `provider` key | Default model |
|----------------|----------------|---------------|
| OpenAI         | `openai`       | `text-embedding-3-small` |
| Google         | `google`       | `text-embedding-004` |
| Mistral        | `mistral`      | `mistral-embed` |
| Cohere         | `cohere`       | `embed-english-v3.0` |
| Voyage         | `voyage`       | `voyage-3` |
| Ollama (local) | `ollama`       | `nomic-embed-text` |
| Local (offline)| `local`        | `all-MiniLM-L6-v2` |

---

## Configuration Reference

```python
Document("./policy.pdf", {
    # Chunking
    "chunkSize":      500,          # words per chunk (default: 500)
    "overlap":        50,           # overlapping words between chunks (default: 50)

    # Retrieval
    "topK":           5,            # default results returned (default: 5)
    "scoreThreshold": 0.0,          # minimum cosine similarity (0..1, default: 0)

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

Every `build()` call fingerprints the source file with a **SHA-256 content hash** and persists it alongside the vectors. The cached index is reused only if **all** of the following match the stored index:

| Factor | Triggers rebuild if changed |
|--------|-----------------------------|
| File content | SHA-256 hash differs |
| Chunk size | `chunkSize` changed |
| Overlap | `overlap` changed |
| Embedding provider/model | Provider or model string changed |
| Library version | Package version bumped |

Pass `rebuild=True` to `build()` to force a fresh index regardless.

Each document is stored under `.raglite/<sha256-prefix>/`, so multiple documents in the same project never overwrite each other.

---

## Advanced Usage

### Custom Vector Store

```python
from raglite.vectordb.base import VectorStore

class MyVectorStore(VectorStore):
    # Implement: load, reset, add, search, count,
    #            save_index_metadata, read_index_metadata
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
git clone <repo>
cd raglite-py

# Create virtual environment
python3.12 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Run tests (76 tests, ~5s, no network required)
pytest

# Run with coverage
pytest --cov=raglite --cov-report=term-missing

# Run examples (uses local offline embeddings)
python examples/basic.py
python examples/multi_provider.py   # requires API keys in env
python examples/serve.py
```

### Test Structure

```
tests/
├── unit/
│   ├── test_chunking.py   # RecursiveChunker algorithm
│   ├── test_vectordb.py   # MemoryVectorStore (cosine, persistence, isolation)
│   ├── test_loaders.py    # TxtLoader, MarkdownLoader, JsonLoader
│   ├── test_prompt.py     # system/user prompt builders
│   ├── test_errors.py     # exception hierarchy
│   ├── test_config.py     # config defaults and overrides
│   ├── test_hash.py       # SHA-256 file hashing + namespace generation
│   ├── test_retriever.py  # Retriever with mocked embedder
│   └── test_cli.py        # CLI commands and argument parsing
└── integration/
    ├── test_document.py   # build/cache/search lifecycle (mocked embeddings)
    ├── test_ask.py        # ask/stream with mocked LLM generation
    └── test_api.py        # FastAPI endpoints via TestClient
```

---

## License

MIT
