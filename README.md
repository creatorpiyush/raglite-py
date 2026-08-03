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

## Fully Offline — No API Key Needed

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

## Choose Any LLM at Ask-Time

```python
# Pass an inline LLM override to ask()
gpt4 = doc.ask("Summarise this document", options={
    "llm": {"provider": "openai", "model": "gpt-4o", "apiKey": "sk-..."}
})

claude = doc.ask("Summarise this document", options={
    "llm": {"provider": "anthropic", "model": "claude-3-5-sonnet-20241022", "apiKey": "sk-ant-..."}
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

`raglite` supports pluggable vector stores (Memory, Qdrant, Pinecone, LanceDB, or custom subclasses):

### Memory Store (Default)
```python
doc = Document("./policy.pdf", {
    "vectorStore": {"provider": "memory", "storeDir": ".raglite"}
})
```

### Qdrant Store
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

# Ask a question (streaming)
raglite ask ./docs "What is the refund policy?" \
  --llm-provider anthropic --llm-key $ANTHROPIC_API_KEY --stream

# Serve a REST API
raglite serve https://example.com \
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
