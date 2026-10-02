# Changelog

All notable changes to this project will be documented in this file.

## [1.2.2] - Unreleased

### Changed
- **Upgrades keep cached indexes:** The build cache is now keyed on an index format version (`formatVersion` in `IndexMetadata`) instead of the package version, so upgrading RAGLite no longer re-embeds every index. Indexes built by 1.2.1 are reused as-is; indexes from older releases are rebuilt once.

### Fixed
- **Chunker parity with the TypeScript SDK:** Words are now split on exactly the whitespace characters JavaScript treats as whitespace. Previously text containing a byte order mark (U+FEFF), U+0085 or U+001C–U+001F was chunked differently from the TypeScript SDK. Existing indexes are not rebuilt for this; pass `rebuild=True` if your sources contain those characters.
- **Docs:** ARCHITECTURE.md now describes the word-based `RecursiveChunker` (500 words, 50 overlap) and the actual `VectorStore` interface. The README no longer lists LanceDB, which is TypeScript-only.

### Internal
- Added cross-SDK fixtures (`tests/fixtures/shared/`, copied from the TypeScript SDK with `scripts/sync_shared_fixtures.py`) that pin chunking and hashing output. CI fails if the copy is out of date.

## [1.2.1] - 2026-10-02

### Fixed
- **URL sources:** `Document.build()` no longer fails with `LoaderError: File does not exist` for web URLs, so URLs work on their own and inside a `DocumentCollection`.
- **Custom `VectorStore` instances:** Passing a `VectorStore` subclass instance as `{"vectorStore": store}` no longer fails config validation, so the documented custom store usage works.
- **Qdrant shared collections:** When `indexName` is set, documents sharing one Qdrant collection no longer wipe each other. `reset()`, search and index metadata are now scoped to each document's namespace instead of the whole collection. Without `indexName`, behaviour is unchanged.
- **Shared `VectorStore` instances in collections:** `DocumentCollection.build()` now raises `ConfigError` when a single `VectorStore` instance would be shared by more than one document, instead of each document silently resetting the previous one's index. Pass a vector store provider config instead.
- **Web URL re-indexing:** URL sources are now fingerprinted by their fetched content rather than the URL string, so a changed page is re-indexed on the next `build()`.
- **Query embedder after reload:** Searching an existing index in a new process now embeds queries with the provider and model the index was built with, rather than the constructor default. Configured credentials are reused when the provider matches.
- **Collection search errors:** `DocumentCollection.search()` (and `ask`/`ask_stream`) now logs per-document search failures instead of silently dropping them, and raises if every document fails.
- **Web loader User-Agent:** Now reports the actual package version.

### Upgrade notes
- As with every release, cached indexes are rebuilt once on first `build()` because the package version is part of the cache key.
- Existing Qdrant indexes created with `indexName` are rebuilt once. Their old untagged points stay in the collection but are no longer returned by search; drop and recreate the collection to remove them.
- Each `build()` on a URL source now fetches the page to check for changes, even when the cached index is reused.
- When every document in a collection fails to search, the REST server now returns an error response (400 for RAGLite errors, 500 otherwise) instead of empty results.

## [1.2.0] - 2026-08-02

### Added
- **Multi-Document & Directory Ingestion (`DocumentCollection`):**
  - Added `DocumentCollection` class to manage semantic indexing, multi-document retrieval, and Q&A across folders, glob patterns, web URLs, and mixed file lists.
  - Parallel semantic search over collection vector stores with score-based top-$K$ merging and ranking.
  - Contextual Q&A synthesis (`ask` and `ask_stream`) across multi-document collections.
- **Directory Loader (`DirectoryLoader`):**
  - Recursive directory scanner (`recursive=True`) with glob pattern matching (e.g. `./docs/**/*.md`).
  - Auto-detection of supported extensions (`.pdf`, `.txt`, `.md`, `.json`, `.docx`).
  - Detailed error reporting and warning logs for unsupported/empty files.
- **Web Loader (`WebLoader`):**
  - Native loader for fetching HTTP/HTTPS web URLs directly.
  - Automatic HTML cleaning into formatted text/markdown with script, style, and SVG tag stripping.
  - JSON and plain text content-type parsing.
- **CLI & FastAPI REST Server Support:**
  - Upgraded `raglite index`, `search`, `ask`, and `serve` CLI commands to process directories, glob patterns, and URLs.
  - Updated FastAPI REST server to support `DocumentCollection` and single `Document` targets.
- **Unit & Integration Tests:**
  - Added unit and integration tests for `DirectoryLoader`, `WebLoader`, and `DocumentCollection`.

## [1.1.0] - 2026-07-19

### Added
- **Pluggable Vector Databases:** Added support for custom local and cloud vector database backends via a new `vectorStore` config option.
  - **Memory Store (`"memory"`):** Default in-memory store persisting indexes locally to JSON (unchanged behaviour).
  - **Qdrant Store (`"qdrant"`):** Wrapper for Qdrant local/cloud using stdlib `urllib` REST requests. Supports auto-collection creation, API key auth, and custom collection names.
  - **Pinecone Store (`"pinecone"`):** Cloud database support using Pinecone Namespaces and stdlib `urllib` REST requests. Stores index metadata as a reserved `__metadata__` vector.
  - **Custom Adapters:** Pass any class instance implementing the `VectorStore` ABC directly as `vectorStore` in `DocumentOptions`.
- **`VectorStoreProviderConfig` type:** New Pydantic model in `types.py` describing provider, URL, API key, index name, and store directory.
- **Factory:** `create_vector_store(config, namespace)` utility in `vectordb/factory.py` resolving the correct store from config.
- **Examples:**
  - `examples/qdrant_example.py` — full index + search demo with Qdrant.
  - `examples/custom_store_example.py` — implementing and using a custom VectorStore ABC subclass.
- **Automation Scripts:**
  - `scripts/pre-commit.sh` — runs ruff lint, mypy type-check, and pytest before committing.
  - `scripts/pre-release.sh` — cleans builds, runs full verification, and builds distribution packages.
- **GitHub Actions Workflow:** `pr-verify.yml` — automatically runs code style checks and the full test suite on every pull request and push to `main`/`master`.

### Changed
- **`DocumentOptions` / `ResolvedConfig`:** Added optional `vectorStore` field supporting `VectorStoreProviderConfig` or a custom `VectorStore` instance.
- **`Document.__init__`:** Constructor now resolves the appropriate vector store from config, accepting provider configs or custom instances.
