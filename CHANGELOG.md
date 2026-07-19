# Changelog

All notable changes to this project will be documented in this file.

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
