import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

from ..config import DocumentOptions, ResolvedConfig, resolve_config
from ..errors import ConfigError, RagLiteError
from ..llm import generate_answer, stream_answer
from ..loaders import DirectoryLoader, is_supported_file, is_url
from ..types import AnswerResult, SearchResult
from ..utils.logger import Logger, create_logger
from ..vectordb import VectorStore
from .document import Document


@dataclass
class CollectionBuildResult:
    totalDocuments: int
    totalChunks: int
    cachedDocuments: int
    newDocuments: int
    errors: List[Dict[str, str]] = field(default_factory=list)


class DocumentCollection:
    """Manages semantic indexing, multi-document retrieval, and question-answering across
    multiple files, directories, glob patterns, and web URLs."""

    def __init__(
        self,
        sources: Union[str, Path, List[Union[str, Path]]] = None,
        options: Optional[Union[DocumentOptions, Dict[str, Any]]] = None,
    ):
        self.options = options or {}
        self.config: ResolvedConfig = resolve_config(options)
        self.logger: Logger = create_logger(self.config.logLevel)
        self.sources: List[str] = []
        self.documents: Dict[str, Document] = {}
        self.ready: bool = False

        if sources is not None:
            raw = sources if isinstance(sources, list) else [sources]
            for src in raw:
                src_str = str(src).strip()
                if src_str:
                    self.sources.append(src_str)

    def add_source(self, source: Union[str, Path]) -> None:
        """Add a file path, directory path, glob pattern, or web URL to the collection."""
        src_str = str(source).strip()
        if src_str:
            self.sources.append(src_str)
            self.ready = False

    def build(self, options: Optional[Dict[str, Any]] = None, rebuild: bool = False) -> CollectionBuildResult:
        """Build (or reuse) semantic indexes for all documents in the collection."""
        opts = options or {}
        should_rebuild = rebuild or bool(opts.get("rebuild", False))

        file_list: List[str] = []
        errors: List[Dict[str, str]] = []

        for src in self.sources:
            if is_url(src):
                file_list.append(src)
                continue

            resolved = os.path.abspath(src)
            if not os.path.exists(resolved):
                msg = f"Source path does not exist: {src}"
                self.logger.warning(msg)
                errors.append({"source": src, "error": msg})
                continue

            if os.path.isdir(resolved):
                dir_loader = DirectoryLoader(resolved)
                try:
                    res = dir_loader.load_files()
                    for item in res.loaded:
                        file_list.append(item.file_path)
                    for err in res.errors:
                        errors.append({"source": err.file_path, "error": err.error})
                except Exception as err:
                    err_msg = str(err)
                    self.logger.error(f'Error scanning directory "{src}": {err_msg}')
                    errors.append({"source": src, "error": err_msg})
            elif os.path.isfile(resolved):
                if is_supported_file(resolved):
                    file_list.append(resolved)
                else:
                    msg = f"Unsupported file type: {src}"
                    self.logger.warning(msg)
                    errors.append({"source": src, "error": msg})

        # A VectorStore instance has a single namespace, so every document would
        # share it and each build() would reset the previous document's index.
        raw_store = self.options.get("vectorStore") if isinstance(self.options, dict) else None
        if isinstance(raw_store, VectorStore) and len(set(file_list)) > 1:
            raise ConfigError(
                "A VectorStore instance cannot be shared by multiple documents in a "
                "DocumentCollection. Pass a vector store provider config "
                '(e.g. {"provider": "qdrant", ...}) instead.'
            )

        total_chunks = 0
        cached_docs = 0
        new_docs = 0

        for file_path in file_list:
            try:
                doc = self.documents.get(file_path)
                if doc is None:
                    doc = Document(file_path, self.options)
                    self.documents[file_path] = doc

                res = doc.build(options, rebuild=should_rebuild)
                chunk_count = res.get("chunkCount", 0)
                total_chunks += chunk_count

                if res.get("cached", False):
                    cached_docs += 1
                else:
                    new_docs += 1
            except Exception as err:
                err_msg = str(err)
                self.logger.error(f'Failed to index document "{file_path}": {err_msg}')
                errors.append({"source": file_path, "error": err_msg})

        self.ready = True
        self.logger.info(
            f"Collection index ready ({len(self.documents)} documents, {total_chunks} total chunks, {len(errors)} error(s))."
        )

        return CollectionBuildResult(
            totalDocuments=len(self.documents),
            totalChunks=total_chunks,
            cachedDocuments=cached_docs,
            newDocuments=new_docs,
            errors=errors,
        )

    def search(
        self,
        query: str,
        options: Optional[Dict[str, Any]] = None,
        *,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Semantic search across all documents in the collection."""
        self._ensure_ready()
        if not self.documents:
            return []

        opts = options or {}
        tk = top_k if top_k is not None else opts.get("topK", opts.get("top_k", self.config.topK))
        st = score_threshold if score_threshold is not None else opts.get("scoreThreshold", opts.get("score_threshold", self.config.scoreThreshold))

        all_hits: List[SearchResult] = []
        failures: List[Exception] = []
        for doc in self.documents.values():
            try:
                hits = doc.search(query, top_k=tk * 2, score_threshold=st)
                all_hits.extend(hits)
            except Exception as err:
                failures.append(err)
                self.logger.warning(f'Search failed for document "{doc.file_path}": {err}')

        # One broken document should not hide results from the others, but if
        # every document failed the caller needs the error, not an empty list.
        if len(failures) == len(self.documents):
            raise failures[0]

        all_hits.sort(key=lambda h: h.score, reverse=True)
        return all_hits[:tk]

    def ask(self, question: str, options: Optional[Dict[str, Any]] = None) -> AnswerResult:
        """Ask a question across the entire collection."""
        opts = options or {}
        llm_config = opts.get("llm") or self.config.llm
        if not llm_config:
            raise RagLiteError(
                "No LLM provider configured. Pass one to `ask(options={'llm': ...})` or `DocumentCollection(..., options={'llm': ...})`."
            )

        context = self.search(
            question,
            top_k=opts.get("topK", opts.get("top_k", self.config.topK)),
            score_threshold=opts.get("scoreThreshold", opts.get("score_threshold", self.config.scoreThreshold)),
        )
        if not context:
            raise RagLiteError("No relevant context found in document collection to answer question.")

        return generate_answer(
            llm_config=llm_config,
            question=question,
            context=context,
            include_citations=opts.get("includeCitations", opts.get("include_citations", True)),
            system_hint=opts.get("systemHint", opts.get("system_hint")),
        )

    def ask_stream(self, question: str, options: Optional[Dict[str, Any]] = None) -> Generator[str, None, None]:
        """Stream LLM response over retrieved collection context."""
        opts = options or {}
        llm_config = opts.get("llm") or self.config.llm
        if not llm_config:
            raise RagLiteError(
                "No LLM provider configured. Pass one to `ask_stream(options={'llm': ...})` or `DocumentCollection(..., options={'llm': ...})`."
            )

        context = self.search(
            question,
            top_k=opts.get("topK", opts.get("top_k", self.config.topK)),
            score_threshold=opts.get("scoreThreshold", opts.get("score_threshold", self.config.scoreThreshold)),
        )
        if not context:
            raise RagLiteError("No relevant context found in document collection to answer question.")

        yield from stream_answer(
            llm_config=llm_config,
            question=question,
            context=context,
            include_citations=opts.get("includeCitations", opts.get("include_citations", True)),
            system_hint=opts.get("systemHint", opts.get("system_hint")),
        )

    def serve(
        self,
        host: str = "127.0.0.1",
        port: int = 8085,
        bearer_token: Optional[str] = None,
        llm: Optional[Any] = None,
    ) -> None:
        """Serve a FastAPI REST API server over the document collection."""
        import uvicorn

        from ..api.server import create_app

        self._ensure_ready()
        opts = {
            "host": host,
            "port": port,
            "bearer_token": bearer_token,
            "llm": llm,
        }
        app = create_app(self, opts)
        uvicorn.run(app, host=host, port=port)

    def get_documents(self) -> List[Document]:
        """Access underlying Document instances."""
        return list(self.documents.values())

    @property
    def chunk_count(self) -> int:
        """Total chunks indexed across all documents in the collection."""
        return sum(doc.chunk_count for doc in self.documents.values())

    @property
    def resolved_config(self) -> ResolvedConfig:
        """Underlying resolved configuration."""
        return self.config

    def _ensure_ready(self) -> None:
        if not self.ready:
            self.build()
