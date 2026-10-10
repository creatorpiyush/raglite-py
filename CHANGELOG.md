# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-10-10

### Breaking changes
- **Provider SDKs are now optional extras.** `pip install raglite-toolkit` no longer installs every provider SDK or `sentence-transformers` (with PyTorch). A fresh install drops from about 1.1 GB to 36 MB. Install the extras you use, for example `pip install 'raglite-toolkit[openai,local]'`, or `[all]` for everything. Extras: `openai`, `anthropic`, `google`, `cohere`, `mistral`, `voyage`, `local`, and `groq`, `xai` and `ollama` (these three install the `openai` client). When one is missing, RAGLite raises a `ConfigError` naming the `pip install` command to run.
- **The default `local` embedding provider needs the `[local]` extra.** A missing `sentence-transformers` now raises `ConfigError` rather than `EmbeddingError`. A model that fails to load raises `EmbeddingError` with the real cause, instead of being reported as a missing package.
- **Google moved from `google-generativeai` to `google-genai`.** Google has deprecated the old package. The `[google]` extra installs `google-genai`. `baseURL` is now honoured for Google.
- **Python 3.11 or later is required.** Python 3.10 reaches end-of-life this month.
- **Default models changed** where the provider has shut the old one down: Anthropic `claude-3-5-sonnet-20241022` → `claude-sonnet-5-5`, Google `gemini-2.0-flash` → `gemini-3.8-flash`, Cohere `command-r-plus` → `command-a-03-2025`, Groq `llama-3.3-70b-versatile` → `openai/gpt-oss-120b`, xAI `grok-2-latest` → `grok-4.7`, and Google embeddings `text-embedding-004` → `gemini-embedding-2`. Pass `model` to keep a different one.
- **Anthropic gets no `temperature` unless you set one.** Claude 5 models reject non-default sampling values, and `anthropic` 1.x removed the argument. A configured `temperature` is sent through `extra_body` for older models that accept it. The default `max_tokens` for Anthropic is now 16000, because thinking tokens count against it.

### Added
- **`Document.from_text(id, text, options)`** indexes text you already have, such as a database row, an upload or CMS content, with no file. The `id` names the index and is each chunk's `source`; changed text is re-indexed on the next `build()`. It uses the same index namespace as the TypeScript SDK's `Document.fromText`.

### Fixed
- **Mistral with `mistralai` 2.x and later:** `Mistral` moved to `mistralai.client`, so `pip install raglite-toolkit` 1.3.0 with a current `mistralai` failed with an `ImportError`. Both layouts now work.
- **Anthropic answers with thinking:** answers join the text blocks of the response rather than reading the first block, which is a thinking block on models that think by default.

### Internal
- CI runs on Python 3.11, 3.12 and 3.13.

### Upgrade notes
- Install the extras you use, for example `pip install 'raglite-toolkit[openai,local]'`.
- Existing indexes are kept. An index built with Google embeddings and the old default model keeps using `text-embedding-004` for queries, which Google has shut down. Rebuild it with `build(rebuild=True)` to move to `gemini-embedding-2`.

## [1.3.0] - 2026-10-03

### Added
- **Hybrid search:** `search()`, `ask()` and `ask_stream()` accept `mode="vector" | "keyword" | "hybrid"` (or `{"mode": ...}` in the options dict; default `"vector"`), on `Document` and `DocumentCollection`. Keyword mode uses BM25; hybrid merges vector and keyword results with Reciprocal Rank Fusion, so exact terms such as `ERR_4021`, SKUs or function names are found even when embeddings miss them. Tune it with `hybrid={"rrfK", "candidates", "weights"}`, or set defaults with the new `retrieval` option. Results in these modes carry `scores` (`vector`, `keyword`, `fused`); vector-mode results are unchanged.
- **Keyword index:** `build()` now also writes a BM25 index (`<storeDir>/<namespace>/keyword.json`) from the chunk texts. It needs no extra embedding calls, works with every vector store, and uses the same file layout as the TypeScript SDK.
- **Multilingual keyword tokenizer (`raglite-v1`):** NFKC and lowercase normalisation, code identifiers kept whole (`gpt-4.1`, `snake_case`), and character bigrams for Chinese, Japanese, Korean, Thai, Lao, Khmer and Myanmar. Exported as `tokenize()`. It produces exactly the same terms as the TypeScript SDK.
- **HTTP and CLI:** optional `mode` on `/search` and `/ask`; `--mode` on `raglite search`, `ask` and `serve`. `/info` reports `retrievalMode`.
- **`VectorStore` extension points (optional):** `list_chunks()` lets keyword and hybrid search rebuild a missing keyword index from the store, and `keyword_search(query, top_k)` replaces the built-in BM25 index. The memory and Qdrant stores implement `list_chunks()`. Existing custom stores need no changes.

### Changed
- **Chunking of text without spaces:** Chinese, Japanese, Thai, Lao, Khmer and Myanmar text is now chunked by character instead of becoming one giant "word". Previously a document in these scripts became a single chunk of any length, which could exceed embedding model limits. `chunkSize` and `overlap` count characters for these scripts and words for everything else.

### Fixed
- **Custom chunk sizes are kept:** `build()` without `chunk_size` or `overlap` (in the call or the constructor) now reuses the existing index's values instead of the defaults. Previously `raglite search` or `raglite ask` after `raglite index --chunk-size N` silently re-embedded the whole index at the default 500 words. Defaults still apply to a new index, and explicit values still trigger a rebuild when they differ.
- **Embedding provider is kept:** with no `embeddings` configured, `build()` now keeps the existing index's provider and model (reusing configured credentials when the provider matches) instead of switching to the local default. The CLI no longer assumes `--embed-provider local` when no `--embed-*` flag is given, so `raglite search` and `raglite ask` reuse an index built with `--embed-provider openai` instead of re-embedding it locally. New indexes still default to local embeddings.
- **`build(options)` embeddings dict:** `build({"embeddings": {...}})` now accepts a plain dict, as the constructor does.
- **`raglite serve`:** no longer crashes with `TypeError` when given `--llm-provider` or `--token`.
- **`DocumentCollection.serve()`:** no longer fails with `ImportError`, so serving a directory works. It now also falls back to the collection's configured `llm`.
- **`ask()` options:** an explicit `scoreThreshold` of `0` is no longer replaced by the configured default.

### Upgrade notes
- The index format version is now 2. Indexes whose source contains no Chinese, Japanese, Thai, Lao, Khmer or Myanmar text are upgraded in place on the next `build()`, without re-embedding (the source file is read once to check). Indexes of sources that do contain such text are rebuilt once.
- Indexes built before 1.3.0 have no keyword index. The memory and Qdrant stores build it from the stored chunks on the first keyword or hybrid search. Pinecone and custom stores without `list_chunks()` log a warning and use vector search until you run `build(rebuild=True)`.
- With Qdrant or Pinecone, the keyword index lives on local disk under `storeDir`. Keep `storeDir` on persistent storage when you use keyword or hybrid search.

## [1.2.2] - 2026-10-02

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
