# RAGLite Python (`raglite-toolkit`) Architecture

This document details the software architecture, component design, data flow, and APIs of the Python implementation of **RAGLite** (`raglite-toolkit`).

---

## 1. Overview & Core Philosophy

`raglite-toolkit` is a Python-native Retrieval-Augmented Generation library built with Pydantic v2 and FastAPI. It provides zero-boilerplate semantic search, multi-provider LLM response synthesis, and self-hosted REST APIs over local files.

### Key Characteristics
* **Python-Native & Type-Annotated**: Modern Python 3.10+ typing with Pydantic v2 model validation and camelCase alias serialization.
* **1:1 Parity with TypeScript SDK**: Identical function signatures, option dictionaries, data models, and `.raglite/` persistence layout.
* **Pluggable & Extensible**: Modular Abstract Base Classes (ABCs) for Loaders, Chunkers, Embeddings, Vector Databases, and LLMs.
* **Per-Document Isolation**: Namespaced vector collections tied to source document path/identity.
* **SHA-256 Content Caching**: Automatically skips re-embedding if document content has not changed.
* **Offline-First Option**: Local offline embeddings via `sentence-transformers` (`all-MiniLM-L6-v2`) and local LLMs via Ollama.

---

## 2. Directory & Module Structure

```
raglite-py/src/raglite/
├── api/             # FastAPI REST HTTP server & Pydantic schemas
│   ├── __init__.py
│   ├── schemas.py
│   └── server.py
├── chunking/        # Text splitting algorithms
│   ├── __init__.py
│   ├── base.py
│   └── recursive.py
├── core/            # Main facade orchestrator
│   ├── __init__.py
│   └── document.py
├── embeddings/      # Embedding provider factory & adapters
│   ├── __init__.py
│   ├── base.py
│   ├── factory.py
│   ├── local.py      # sentence-transformers (offline)
│   ├── models.py
│   └── remote.py     # OpenAI, Gemini, Mistral, Cohere, Voyage, Ollama
├── llm/             # LLM provider factory & prompt synthesis
│   ├── __init__.py
│   ├── answer.py
│   ├── factory.py
│   ├── models.py
│   └── prompt.py
├── loaders/         # Document parser implementations
│   ├── __init__.py
│   ├── base.py
│   ├── docx.py       # python-docx reader
│   ├── json.py
│   ├── markdown.py
│   ├── pdf.py        # pypdf / pymupdf reader
│   └── txt.py
├── retrieval/       # Retriever & ranking algorithms
│   ├── __init__.py
│   └── retriever.py
├── vectordb/        # Vector Database adapters
│   ├── __init__.py
│   ├── base.py       # VectorStore Abstract Base Class
│   ├── factory.py
│   ├── memory.py     # Local JSON persistence (.raglite/)
│   ├── pinecone.py   # Pinecone Cloud
│   └── qdrant.py     # Qdrant local/cloud
├── cli.py           # Command-line interface tool (Typer/Argparse)
├── config.py        # Environment & default configuration
├── constants.py     # System constants & defaults
├── errors.py        # Custom RAGLite exception definitions
├── types.py         # Pydantic v2 data models & type aliases
└── utils/           # Utility functions (hashing, math, crypto)
```

---

## 3. High-Level System Diagram

```mermaid
graph TD
    subgraph Client Application
        App["Python Application"]
        CLI["raglite CLI"]
        HTTPClient["Curl / HTTPX Client"]
    end

    subgraph Entrypoints
        Doc["Document Facade Class<br/>(src/raglite/core/document.py)"]
        FastAPIServer["FastAPI REST Server<br/>(src/raglite/api/server.py)"]
    end

    subgraph Core Pipeline Modules
        Loaders["Document Loaders<br/>(src/raglite/loaders/)"]
        Chunker["Recursive Splitter<br/>(src/raglite/chunking/)"]
        Embeddings["Embeddings Engine<br/>(src/raglite/embeddings/)"]
        VectorStore["VectorStore Adapter<br/>(src/raglite/vectordb/)"]
        Retriever["Retriever Engine<br/>(src/raglite/retrieval/)"]
        LLM["LLM Synthesis Engine<br/>(src/raglite/llm/)"]
    end

    subgraph Persistence Layer
        JSONDisk["Disk Index Storage<br/>(.raglite/indexes/*.json)"]
        ExternalVDB["External Vector DB<br/>(Qdrant / Pinecone)"]
    end

    App --> Doc
    CLI --> Doc
    HTTPClient --> FastAPIServer
    FastAPIServer --> Doc

    Doc --> Loaders
    Doc --> Chunker
    Doc --> Embeddings
    Doc --> VectorStore
    Doc --> Retriever
    Doc --> LLM

    VectorStore --> JSONDisk
    VectorStore --> ExternalVDB
```

---

## 4. End-to-End Data Pipeline

### 4.1 Ingestion & Indexing
1. `doc.build()` invokes the appropriate `BaseLoader` based on file extension (`.pdf`, `.txt`, `.json`, `.md`, `.docx`).
2. Calculates SHA-256 hash of raw document content.
3. Checks existing `IndexMetadata` in `VectorStore`. The cached index is reused when the index format version, content hash, chunk size, overlap and embedding provider/model all match (URL sources are hashed by their fetched text).
4. If hash differs or force rebuild requested:
   - `RecursiveChunker` splits text into word-based chunks (default 500 words, 50-word overlap).
   - `EmbeddingFactory` generates normalized vectors for each chunk.
   - `VectorStore.add()` saves chunks and `VectorStore.save_index_metadata()` persists index metadata.

### 4.2 Retrieval & Answer Synthesis
1. `doc.search(query, top_k)`:
   - Embeds query text.
   - Executes vector search via `VectorStore.search()`, calculating cosine similarity over L2-normalized vectors.
   - Returns top $K$ `SearchResult` objects.
2. `doc.ask(question, options)`:
   - Runs `search(question)`.
   - Constructs context-augmented system/user prompt via `build_prompt()`.
   - Calls `generate_answer()` or `stream_answer()` with selected LLM adapter (OpenAI, Anthropic, Google, Groq, Ollama, etc.).

---

## 5. REST API Architecture

Built using **FastAPI** framework for async capabilities, automatic OpenAPI docs, and Uvicorn serving.

| Method | Endpoint | Auth Required? | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | No | Liveness status & total chunk count |
| `GET` | `/info` | Yes (Bearer token) | Configuration details & index state |
| `POST` | `/search` | Yes (Bearer token) | Semantic vector search |
| `POST` | `/ask` | Yes (Bearer token) | Context Q&A generation (supports streaming via StreamingResponse) |

---

## 6. Vector Database Abstract Base Class (`src/raglite/vectordb/base.py`)

```python
from abc import ABC, abstractmethod
from typing import List, Optional
from raglite.types import IndexMetadata, StoredChunk
from raglite.vectordb.base import VectorSearchHit

class VectorStore(ABC):
    @property
    @abstractmethod
    def namespace(self) -> str: ...

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def add(self, chunks: List[StoredChunk]) -> None: ...

    @abstractmethod
    def search(self, embedding: List[float], top_k: int) -> List[VectorSearchHit]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def save_index_metadata(self, metadata: IndexMetadata) -> None: ...

    @abstractmethod
    def read_index_metadata(self) -> Optional[IndexMetadata]: ...
```
