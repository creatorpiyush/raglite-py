import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Union

from ..chunking import RecursiveChunker
from ..config import DocumentOptions, ResolvedConfig, resolve_config
from ..constants import (
    DEFAULT_STORE_DIRNAME,
    INDEX_FORMAT_VERSION,
    LEGACY_FORMAT_1_VERSIONS,
    PACKAGE_VERSION,
)
from ..embeddings import Embedder, create_embedder
from ..embeddings.models import DEFAULT_EMBEDDING_MODELS, is_retired_embedding_model
from ..errors import FileNotIndexedError, LoaderError, RagLiteError
from ..llm import generate_answer, stream_answer
from ..loaders import get_loader, is_url
from ..retrieval import Retriever
from ..retrieval.fusion import RankedList, reciprocal_rank_fusion
from ..retrieval.keyword_index import (
    KeywordHit,
    KeywordIndex,
    keyword_file_matches,
    read_keyword_file,
    write_keyword_file,
)
from ..retrieval.plan import RetrievalPlan, resolve_retrieval_plan
from ..text.scripts import has_unspaced_text
from ..types import (
    AnswerResult,
    ChunkMetadata,
    EmbeddingProviderConfig,
    IndexMetadata,
    SearchResult,
    StoredChunk,
)
from ..utils.hash import hash_file, hash_string, namespace_from_path
from ..utils.logger import create_logger
from ..vectordb import MemoryVectorStore, VectorStore, create_vector_store


@dataclass
class RetrievalCandidates:
    """Candidate lists of one document, before fusion (internal)."""

    vector: List[SearchResult]
    # None when the document has no keyword index (a warning has been logged).
    keyword: Optional[List[KeywordHit]]


class Document:
    @classmethod
    def from_text(
        cls,
        id: str,
        text: str,
        options: Optional[Union[DocumentOptions, Dict[str, Any]]] = None,
    ) -> "Document":
        """Index text you already have (a DB row, an upload, CMS content) without a file.

        ``id`` names the index: reuse the same id to reuse it, and changed text is
        re-indexed on the next build(). ``id`` is also the chunks' ``source``.
        """
        return cls(id, options, _text=text)

    def __init__(
        self,
        file_path: str,
        options: Optional[Union[DocumentOptions, Dict[str, Any]]] = None,
        *,
        _text: Optional[str] = None,
    ):
        # Set by from_text(): indexed instead of reading file_path.
        self._inline_text = _text
        if _text is not None or is_url(file_path):
            self.file_path = file_path
        else:
            self.file_path = os.path.abspath(file_path)
        self.config: ResolvedConfig = resolve_config(options)
        self.logger = create_logger(self.config.logLevel)
        self.namespace = namespace_from_path(
            f"text:{file_path}" if _text is not None else self.file_path
        )

        # Resolve vector store — options may provide a VectorStore instance directly,
        # a VectorStoreProviderConfig dict/object, or fall back to the in-memory default.
        raw_opts = options if isinstance(options, dict) else {}
        raw_store = raw_opts.get("vectorStore") if raw_opts else None
        if isinstance(raw_store, VectorStore):
            self.store: VectorStore = raw_store
        elif self.config.vectorStore is not None:
            self.store = create_vector_store(self.config.vectorStore, self.namespace)
        else:
            self.store = MemoryVectorStore(self.config.storeDir, self.namespace)

        # Sidecar file holding the BM25 keyword index; see retrieval/keyword_index.py.
        base_dir = (
            self.config.storeDir
            if isinstance(raw_store, VectorStore)
            else _keyword_base_dir(self.config)
        )
        self.keyword_path = os.path.join(base_dir, self.namespace, "keyword.json")

        # Chunking the caller set in the constructor; unset values follow the existing index.
        self._chunking = _explicit_chunking(options)
        # Embeddings the caller set in the constructor; if unset, an existing index keeps its own.
        self._explicit_embeddings = (
            options.embeddings is not None
            if isinstance(options, DocumentOptions)
            else isinstance(options, dict) and options.get("embeddings") is not None
        )

        self.embedder: Optional[Embedder] = None
        self.ready = False
        self._keyword_index: Optional[KeywordIndex] = None
        self._keyword_unavailable_warned = False

    def build(
        self,
        options: Optional[Dict[str, Any]] = None,
        *,
        chunk_size: Optional[int] = None,
        overlap: Optional[int] = None,
        embeddings: Optional[Any] = None,
        rebuild: Optional[bool] = None,
    ) -> Dict[str, Any]:
        opts = options or {}

        c_size = chunk_size
        if c_size is None:
            c_size = opts.get("chunkSize")
        if c_size is None:
            c_size = opts.get("chunk_size")
        if c_size is None:
            c_size = self._chunking["chunkSize"]

        c_overlap = overlap
        if c_overlap is None:
            c_overlap = opts.get("overlap")
        if c_overlap is None:
            c_overlap = self._chunking["overlap"]

        embed_config = embeddings
        if embed_config is None:
            embed_config = opts.get("embeddings")
        if embed_config is None and self._explicit_embeddings:
            embed_config = self.config.embeddings
        if isinstance(embed_config, dict):
            embed_config = EmbeddingProviderConfig.model_validate(embed_config)

        should_rebuild = rebuild
        if should_rebuild is None:
            should_rebuild = opts.get("rebuild", False)

        if (
            self._inline_text is None
            and not is_url(self.file_path)
            and not os.path.exists(self.file_path)
        ):
            raise LoaderError(f"File does not exist: {self.file_path}")

        self.store.load()
        existing = self.store.read_index_metadata()
        # Chunking nobody asked for keeps the existing index's values, so a
        # plain build() (for example from `raglite search`) does not re-embed
        # an index that was built with a custom chunk size.
        if c_size is None:
            c_size = existing.chunkSize if existing is not None else self.config.chunkSize
        if c_overlap is None:
            c_overlap = existing.overlap if existing is not None else self.config.overlap
        retired = existing is not None and _uses_retired_model(existing)
        requested_embeddings = embed_config
        # Likewise an unconfigured embedding provider keeps the existing
        # index's provider and model rather than switching it to the local
        # default, unless the provider has shut that model down.
        if embed_config is None:
            embed_config = (
                _query_embeddings_config(existing, self.config.embeddings)
                if existing is not None
                else self.config.embeddings
            )
            if retired:
                # Inherited a shut-down model: fall back to the provider's current default.
                embed_config = embed_config.model_copy(update={"model": None})
        # Web pages change without their URL changing, so fingerprint the
        # fetched content rather than the URL.
        text: Optional[str] = self._inline_text
        if text is None and is_url(self.file_path):
            text = get_loader(self.file_path).load()
        source_hash = hash_string(text) if text is not None else hash_file(self.file_path)

        # Re-embed an index built with a shut-down model unless that model was asked for explicitly.
        keep_retired = (
            retired
            and existing is not None
            and requested_embeddings is not None
            and requested_embeddings.model == existing.embeddingModel
        )
        reusable = (
            not should_rebuild
            and existing is not None
            and (not retired or keep_retired)
            and self._cache_key_matches(existing, source_hash, c_size, c_overlap, embed_config)
        )
        if existing is not None and retired and not keep_retired and not should_rebuild:
            new_model = embed_config.model or DEFAULT_EMBEDDING_MODELS[embed_config.provider]
            self.logger.warning(
                f'Index was built with {existing.embeddingProvider} "{existing.embeddingModel}", '
                f'which the provider has shut down. Re-embedding with "{new_model}".'
            )
        if (
            reusable
            and existing is not None
            and index_format_version(existing) != INDEX_FORMAT_VERSION
        ):
            # Format 2 only changed how unspaced scripts are chunked, so a
            # format-1 index of a source without such text is still exact:
            # upgrade it in place instead of paying to re-embed it.
            if text is None:
                text = get_loader(self.file_path).load()
            reusable = index_format_version(existing) == 1 and not has_unspaced_text(text)
            if reusable:
                existing = existing.model_copy(update={"formatVersion": INDEX_FORMAT_VERSION})
                self.store.save_index_metadata(existing)

        if reusable and existing is not None:
            self.logger.info(
                f"Reusing cached index ({existing.chunkCount} chunks)."
            )
            self.ready = True
            if self.embedder is None:
                self.embedder = create_embedder(embed_config)
            return {
                "chunkCount": existing.chunkCount,
                "cached": True,
                "embeddingProvider": existing.embeddingProvider,
                "embeddingModel": existing.embeddingModel,
                "dimensions": existing.embeddingDimensions,
            }

        self.logger.info("Building new index...")
        self._keyword_index = None
        self.store.reset()
        self.store.load()

        if text is None:
            text = get_loader(self.file_path).load()
        if not text:
            raise LoaderError(
                f"Loader returned empty text for {self.file_path}"
            )

        chunker = RecursiveChunker(c_size, c_overlap)
        chunks = chunker.split(text)
        if len(chunks) == 0:
            raise RagLiteError(f"No chunks produced from {self.file_path}")

        self.logger.info(f"Produced {len(chunks)} chunk(s). Embedding...")

        embedder = create_embedder(embed_config)
        vectors = embedder.embed_documents(chunks)
        if len(vectors) != len(chunks):
            raise RagLiteError(
                f"Embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )

        source = (
            self.file_path
            if self._inline_text is not None or is_url(self.file_path)
            else os.path.basename(self.file_path)
        )
        stored_chunks = []
        for index, text_chunk in enumerate(chunks):
            metadata = ChunkMetadata(
                source=source,
                chunk=index + 1,
                totalChunks=len(chunks),
            )
            chunk_id = f"{self.namespace}_{(index + 1):06d}"
            stored_chunks.append(
                StoredChunk(
                    id=chunk_id,
                    text=text_chunk,
                    embedding=vectors[index],
                    metadata=metadata,
                )
            )

        self.store.add(stored_chunks)

        dimensions = embedder.dimensions or (
            len(vectors[0]) if vectors else 0
        )
        from datetime import timezone
        index_metadata = IndexMetadata(
            version=PACKAGE_VERSION,
            formatVersion=INDEX_FORMAT_VERSION,
            source=self.file_path,
            sourceHash=source_hash,
            chunkSize=c_size,
            overlap=c_overlap,
            embeddingProvider=embedder.provider,
            embeddingModel=embedder.model,
            embeddingDimensions=dimensions,
            chunkCount=len(chunks),
            createdAt=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        )
        self.store.save_index_metadata(index_metadata)
        self._save_keyword_index(KeywordIndex.build(stored_chunks), index_metadata)

        self.embedder = embedder
        self.ready = True
        self.logger.info(f"Index ready ({len(chunks)} chunks).")

        return {
            "chunkCount": len(chunks),
            "cached": False,
            "embeddingProvider": embedder.provider,
            "embeddingModel": embedder.model,
            "dimensions": embedder.dimensions,
        }

    def search(
        self,
        query: str,
        options: Optional[Dict[str, Any]] = None,
        *,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        mode: Optional[str] = None,
        hybrid: Optional[Any] = None,
    ) -> List[SearchResult]:
        """Search the indexed document. ``mode`` picks vector (default),
        keyword or hybrid retrieval."""
        plan = self.retrieval_plan(
            options, top_k=top_k, score_threshold=score_threshold, mode=mode, hybrid=hybrid
        )
        if plan.mode == "vector":
            self._ensure_ready()
            return self._vector_search(query, plan.top_k, plan.score_threshold)
        candidates = self.retrieve_candidates(query, plan)
        if candidates.keyword is None:
            return candidates.vector[: plan.top_k]
        return reciprocal_rank_fusion(
            fusion_lists(plan, candidates.vector, candidates.keyword), plan.rrf_k, plan.top_k
        )

    def retrieval_plan(
        self,
        options: Optional[Dict[str, Any]] = None,
        *,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        mode: Optional[str] = None,
        hybrid: Optional[Any] = None,
    ) -> RetrievalPlan:
        """Resolve search options against this document's defaults (internal)."""
        opts = options or {}

        tk = top_k
        if tk is None:
            tk = opts.get("topK")
        if tk is None:
            tk = opts.get("top_k")
        if tk is None:
            tk = self.config.topK

        st = score_threshold
        if st is None:
            st = opts.get("scoreThreshold")
        if st is None:
            st = opts.get("score_threshold")
        if st is None:
            st = self.config.scoreThreshold

        return resolve_retrieval_plan(
            self.config.retrieval,
            mode if mode is not None else opts.get("mode"),
            hybrid if hybrid is not None else opts.get("hybrid"),
            tk,
            st,
        )

    def retrieve_candidates(self, query: str, plan: RetrievalPlan) -> RetrievalCandidates:
        """The vector and keyword candidate lists for a keyword or hybrid search,
        before fusion (internal). DocumentCollection fuses these across
        documents. Without a keyword index the vector list is returned for
        every mode, so the caller can fall back to vector search."""
        self._ensure_ready()
        keyword = self._keyword_search(query, plan.candidates)
        need_vector = keyword is None or (plan.mode == "hybrid" and plan.vector_weight > 0)
        vector = (
            self._vector_search(
                query,
                plan.top_k if keyword is None else plan.candidates,
                plan.score_threshold,
            )
            if need_vector
            else []
        )
        return RetrievalCandidates(vector=vector, keyword=keyword)

    def ask(
        self, question: str, options: Optional[Dict[str, Any]] = None
    ) -> AnswerResult:
        opts = options or {}
        llm_config = opts.get("llm") or self.config.llm
        if not llm_config:
            raise RagLiteError(
                "No LLM provider configured. Pass one to `ask({ llm: ... })` or `new Document(path, { llm: ... })`."
            )

        context = self.search(question, search_options_of(opts))

        generate_opts = dict(opts)
        generate_opts["llm"] = llm_config
        generate_opts["question"] = question
        generate_opts["context"] = context

        return generate_answer(generate_opts)

    def ask_stream(
        self, question: str, options: Optional[Dict[str, Any]] = None
    ) -> Generator[str, None, None]:
        opts = options or {}
        llm_config = opts.get("llm") or self.config.llm
        if not llm_config:
            raise RagLiteError(
                "No LLM provider configured. Pass one to `ask_stream({ llm: ... })` or `new Document(path, { llm: ... })`."
            )

        context = self.search(question, search_options_of(opts))

        generate_opts = dict(opts)
        generate_opts["llm"] = llm_config
        generate_opts["question"] = question
        generate_opts["context"] = context

        yield from stream_answer(generate_opts)

    # TypeScript compatibility aliases
    def askStream(
        self, question: str, options: Optional[Dict[str, Any]] = None
    ) -> Generator[str, None, None]:
        return self.ask_stream(question, options)

    @property
    def chunk_count(self) -> int:
        return self.store.count()

    @property
    def chunkCount(self) -> int:
        return self.chunk_count

    @property
    def store_namespace(self) -> str:
        return self.namespace

    @property
    def storeNamespace(self) -> str:
        return self.store_namespace

    @property
    def resolved_config(self) -> ResolvedConfig:
        return self.config

    @property
    def resolvedConfig(self) -> ResolvedConfig:
        return self.resolved_config

    @property
    def vector_store(self) -> VectorStore:
        return self.store

    @property
    def vectorStore(self) -> VectorStore:
        return self.vector_store

    def serve(
        self,
        options: Optional[Dict[str, Any]] = None,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        bearer_token: Optional[str] = None,
    ) -> Any:
        self._ensure_ready()
        opts = options or {}
        merged_opts = dict(opts)
        if "llm" not in merged_opts and self.config.llm:
            merged_opts["llm"] = self.config.llm

        from ..api.server import create_server

        return create_server(
            self,
            merged_opts,
            host=host,
            port=port,
            bearer_token=bearer_token,
        )

    def _vector_search(self, query: str, top_k: int, score_threshold: float) -> List[SearchResult]:
        assert self.embedder is not None  # set by build() or _ensure_ready()
        retriever = Retriever(self.embedder, self.store)
        return retriever.retrieve(query, top_k=top_k, score_threshold=score_threshold)

    def _keyword_search(self, query: str, top_k: int) -> Optional[List[KeywordHit]]:
        """BM25 (or the store's native) keyword search; None when no keyword index exists."""
        native = self.store.keyword_search(query, top_k)
        if native is not None:
            return [
                KeywordHit(id=h.id, text=h.text, metadata=h.metadata, score=h.score)
                for h in native
            ]
        index = self._load_keyword_index()
        return index.search(query, top_k) if index is not None else None

    def _load_keyword_index(self) -> Optional[KeywordIndex]:
        if self._keyword_index is not None:
            return self._keyword_index
        metadata = self.store.read_index_metadata()
        if metadata is None:
            return None

        data = read_keyword_file(self.keyword_path)
        if data is not None and keyword_file_matches(data, metadata):
            try:
                self._keyword_index = KeywordIndex.from_file(data)
                return self._keyword_index
            except Exception:
                pass

        # Indexes built before 1.3, or a sidecar left behind on another
        # machine: rebuild from the stored chunk texts, which needs no
        # embedding calls.
        chunks = self.store.list_chunks()
        if chunks is not None and len(chunks) == metadata.chunkCount:
            self.logger.info(f"Building keyword index from {len(chunks)} stored chunk(s).")
            index = KeywordIndex.build(chunks)
            self._save_keyword_index(index, metadata)
            return index

        if not self._keyword_unavailable_warned:
            self._keyword_unavailable_warned = True
            self.logger.warn(
                f'No keyword index for "{self.file_path}" (expected {self.keyword_path}); '
                "using vector search. Run build(rebuild=True) to enable keyword and hybrid search."
            )
        return None

    def _save_keyword_index(self, index: KeywordIndex, metadata: IndexMetadata) -> None:
        self._keyword_index = index
        try:
            write_keyword_file(self.keyword_path, index.to_file(metadata))
        except Exception as err:
            # The vector index is already saved; keyword search still works in
            # this process and will be rebuilt or reported the next time it is
            # needed.
            self.logger.warn(f"Could not save keyword index to {self.keyword_path}: {err}")

    def _ensure_ready(self) -> None:
        if self.ready and self.embedder is not None:
            return

        self.store.load()
        existing = self.store.read_index_metadata()
        if not existing:
            raise FileNotIndexedError(
                f'No RagLite index found for "{self.file_path}". Call build() first.'
            )

        if _uses_retired_model(existing):
            raise RagLiteError(
                f'The index for "{self.file_path}" was built with {existing.embeddingProvider} '
                f'"{existing.embeddingModel}", which the provider has shut down. '
                "Call build() to re-index it with the current default model."
            )
        if self.embedder is None:
            self.embedder = create_embedder(
                _query_embeddings_config(existing, self.config.embeddings)
            )
        self.ready = True

    def _cache_key_matches(
        self,
        existing: IndexMetadata,
        source_hash: str,
        chunk_size: int,
        overlap: int,
        embeddings_config: Any,
    ) -> bool:
        """Everything in the cache key except the index format, which build() checks separately."""
        if existing.sourceHash != source_hash:
            return False
        if existing.chunkSize != chunk_size:
            return False
        if existing.overlap != overlap:
            return False
        if existing.embeddingProvider != embeddings_config.provider:
            return False

        req_model = embeddings_config.model
        if req_model is not None and req_model != existing.embeddingModel:
            return False
        return True


def index_format_version(metadata: IndexMetadata) -> Optional[int]:
    """Indexes written before ``formatVersion`` existed are identified by package version."""
    if metadata.formatVersion is not None:
        return metadata.formatVersion
    return 1 if metadata.version in LEGACY_FORMAT_1_VERSIONS else None


def _query_embeddings_config(
    existing: IndexMetadata, configured: EmbeddingProviderConfig
) -> EmbeddingProviderConfig:
    """Queries must be embedded with the same provider and model as the index,
    which may differ from the constructor default when ``build(embeddings=...)``
    overrode it. Credentials from the configured provider are reused when it
    matches; otherwise the provider falls back to its environment variables."""
    same_provider = configured.provider == existing.embeddingProvider
    return EmbeddingProviderConfig(
        provider=existing.embeddingProvider,
        model=existing.embeddingModel,
        apiKey=configured.apiKey if same_provider else None,
        baseURL=configured.baseURL if same_provider else None,
    )


def fusion_lists(
    plan: RetrievalPlan, vector: List[SearchResult], keyword: List[KeywordHit]
) -> List[RankedList]:
    """The candidate lists to fuse for a keyword or hybrid plan."""
    if plan.mode == "keyword":
        return [RankedList("keyword", 1, keyword)]
    return [
        RankedList("vector", plan.vector_weight, vector),
        RankedList("keyword", plan.keyword_weight, keyword),
    ]


_SEARCH_KEYS = ("topK", "top_k", "scoreThreshold", "score_threshold", "mode", "hybrid")


def search_options_of(options: Dict[str, Any]) -> Dict[str, Any]:
    """The retrieval-related subset of ask() options."""
    return {k: options[k] for k in _SEARCH_KEYS if options.get(k) is not None}


def _keyword_base_dir(config: ResolvedConfig) -> str:
    """Where the keyword sidecar lives: next to the vector index for the
    memory store (mirroring create_vector_store's default), otherwise under
    ``storeDir``."""
    vs = config.vectorStore
    if vs is None:
        return config.storeDir
    if vs.store_dir:
        return vs.store_dir
    return DEFAULT_STORE_DIRNAME if vs.provider == "memory" else config.storeDir


def _explicit_chunking(options: Any) -> Dict[str, Optional[int]]:
    """chunkSize/overlap as given to the constructor, None where unset."""
    if isinstance(options, DocumentOptions):
        return {"chunkSize": options.chunkSize, "overlap": options.overlap}
    opts = options if isinstance(options, dict) else {}
    size = opts.get("chunkSize", opts.get("chunk_size"))
    return {"chunkSize": size, "overlap": opts.get("overlap")}


def _uses_retired_model(metadata: IndexMetadata) -> bool:
    return is_retired_embedding_model(metadata.embeddingProvider, metadata.embeddingModel)
