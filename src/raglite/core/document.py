from datetime import datetime
import os
from typing import Optional, Union, Dict, Any, Generator

from ..config import DocumentOptions, ResolvedConfig, resolve_config
from ..constants import PACKAGE_VERSION
from ..errors import LoaderError, RagLiteError, FileNotIndexedError
from ..loaders import get_loader
from ..chunking import RecursiveChunker
from ..embeddings import create_embedder, Embedder
from ..vectordb import MemoryVectorStore
from ..retrieval import Retriever
from ..llm import generate_answer, stream_answer
from ..types import StoredChunk, ChunkMetadata, IndexMetadata, AnswerResult
from ..utils.hash import hash_file, namespace_from_path
from ..utils.logger import create_logger


class Document:
    def __init__(
        self,
        file_path: str,
        options: Optional[Union[DocumentOptions, Dict[str, Any]]] = None,
    ):
        self.file_path = os.path.abspath(file_path)
        self.config: ResolvedConfig = resolve_config(options)
        self.logger = create_logger(self.config.logLevel)
        self.namespace = namespace_from_path(self.file_path)
        self.store = MemoryVectorStore(self.config.storeDir, self.namespace)
        self.embedder: Optional[Embedder] = None
        self.ready = False

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
            c_size = self.config.chunkSize

        c_overlap = overlap
        if c_overlap is None:
            c_overlap = opts.get("overlap")
        if c_overlap is None:
            c_overlap = self.config.overlap

        embed_config = embeddings
        if embed_config is None:
            embed_config = opts.get("embeddings") or self.config.embeddings

        should_rebuild = rebuild
        if should_rebuild is None:
            should_rebuild = opts.get("rebuild", False)

        if not os.path.exists(self.file_path):
            raise LoaderError(f"File does not exist: {self.file_path}")

        self.store.load()
        existing = self.store.read_index_metadata()
        source_hash = hash_file(self.file_path)

        if (
            not should_rebuild
            and existing
            and self._cache_still_valid(
                existing, source_hash, c_size, c_overlap, embed_config
            )
        ):
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
        self.store.reset()
        self.store.load()

        loader = get_loader(self.file_path)
        text = loader.load()
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

        source = os.path.basename(self.file_path)
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
        metadata = IndexMetadata(
            version=PACKAGE_VERSION,
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
        self.store.save_index_metadata(metadata)

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
    ) -> list:
        self._ensure_ready()
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

        retriever = Retriever(self.embedder, self.store)
        return retriever.retrieve(query, top_k=tk, score_threshold=st)

    def ask(
        self, question: str, options: Optional[Dict[str, Any]] = None
    ) -> AnswerResult:
        opts = options or {}
        llm_config = opts.get("llm") or self.config.llm
        if not llm_config:
            raise RagLiteError(
                "No LLM provider configured. Pass one to `ask({ llm: ... })` or `new Document(path, { llm: ... })`."
            )

        tk = opts.get("topK") or opts.get("top_k") or self.config.topK
        st = (
            opts.get("scoreThreshold")
            or opts.get("score_threshold")
            or self.config.scoreThreshold
        )

        context = self.search(question, top_k=tk, score_threshold=st)

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

        tk = opts.get("topK") or opts.get("top_k") or self.config.topK
        st = (
            opts.get("scoreThreshold")
            or opts.get("score_threshold")
            or self.config.scoreThreshold
        )

        context = self.search(question, top_k=tk, score_threshold=st)

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
    def vector_store(self) -> MemoryVectorStore:
        return self.store

    @property
    def vectorStore(self) -> MemoryVectorStore:
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

    def _ensure_ready(self) -> None:
        if self.ready and self.embedder is not None:
            return

        self.store.load()
        existing = self.store.read_index_metadata()
        if not existing:
            raise FileNotIndexedError(
                f'No RagLite index found for "{self.file_path}". Call build() first.'
            )

        embed_config = self.config.embeddings
        if embed_config.model is None:
            from ..types import EmbeddingProviderConfig

            embed_config = EmbeddingProviderConfig(
                provider=embed_config.provider,
                model=existing.embeddingModel,
                apiKey=embed_config.apiKey,
                baseURL=embed_config.baseURL,
            )

        if self.embedder is None:
            self.embedder = create_embedder(embed_config)
        self.ready = True

    def _cache_still_valid(
        self,
        existing: IndexMetadata,
        source_hash: str,
        chunk_size: int,
        overlap: int,
        embeddings_config: Any,
    ) -> bool:
        if existing.version != PACKAGE_VERSION:
            return False
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
