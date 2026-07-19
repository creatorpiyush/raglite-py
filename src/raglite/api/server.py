import threading
import time
from typing import Any, Dict, Optional

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import ValidationError

from ..constants import DEFAULT_HOST, DEFAULT_PORT, PACKAGE_VERSION
from ..embeddings.models import DEFAULT_EMBEDDING_MODELS
from ..errors import RagLiteError
from .schemas import AskRequest, SearchRequest

security = HTTPBearer(auto_error=False)


class ServerHandle:
    def __init__(
        self,
        server_thread: threading.Thread,
        uvicorn_server: uvicorn.Server,
        url: str,
    ):
        self.server_thread = server_thread
        self.uvicorn_server = uvicorn_server
        self.url = url

    def close(self) -> None:
        """Stop the background API server."""
        self.uvicorn_server.should_exit = True
        self.server_thread.join()


def build_app(document: Any, options: Dict[str, Any]) -> FastAPI:
    app = FastAPI(title="RAGLite REST API")

    bearer_token = options.get("bearerToken") or options.get("bearer_token")

    if options.get("requestLogging"):

        @app.middleware("http")
        async def log_requests(request: Request, call_next):
            start = time.time()
            response = await call_next(request)
            duration = int((time.time() - start) * 1000)
            print(
                f"[raglite] {request.method} {request.url.path} -> {response.status_code} ({duration}ms)"
            )
            return response

    def authorize(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    ):
        if bearer_token:
            if not credentials or credentials.credentials != bearer_token:
                raise HTTPException(status_code=401, detail="Unauthorized")
        return True

    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        return JSONResponse(
            status_code=400,
            content={"error": "ValidationError", "details": exc.errors()},
        )

    @app.exception_handler(RagLiteError)
    async def raglite_exception_handler(request: Request, exc: RagLiteError):
        return JSONResponse(
            status_code=400,
            content={"error": exc.__class__.__name__, "message": str(exc)},
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        print("[raglite] unhandled error", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "InternalServerError"},
        )

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": PACKAGE_VERSION,
            "chunks": document.chunk_count,
            "namespace": document.store_namespace,
        }

    dependencies = [Depends(authorize)] if bearer_token else []

    @app.get("/info", dependencies=dependencies)
    async def info():
        cfg = document.resolved_config
        llm_config = options.get("llm") or cfg.llm
        return {
            "version": PACKAGE_VERSION,
            "chunkSize": cfg.chunkSize,
            "overlap": cfg.overlap,
            "topK": cfg.topK,
            "embeddings": {
                "provider": cfg.embeddings.provider,
                "model": cfg.embeddings.model
                or DEFAULT_EMBEDDING_MODELS.get(cfg.embeddings.provider),
            },
            "llmProvider": llm_config.provider if llm_config else None,
            "chunks": document.chunk_count,
        }

    @app.post("/search", dependencies=dependencies)
    async def search(req: SearchRequest):
        results = document.search(
            req.query,
            top_k=req.topK,
            score_threshold=req.scoreThreshold,
        )
        return {
            "results": [r.model_dump(by_alias=True) for r in results]
        }

    ask_provider = options.get("llm")

    if ask_provider:

        @app.post("/ask", dependencies=dependencies)
        async def ask(req: AskRequest):
            ask_options = {
                "llm": ask_provider,
                "topK": req.topK,
                "scoreThreshold": req.scoreThreshold,
                "includeCitations": req.includeCitations,
            }

            if req.stream:

                def event_generator():
                    for chunk in document.ask_stream(req.question, ask_options):
                        yield chunk

                return StreamingResponse(
                    event_generator(), media_type="text/plain"
                )

            answer = document.ask(req.question, ask_options)
            return answer.model_dump(by_alias=True)

    else:

        @app.post("/ask", dependencies=dependencies)
        async def ask(req: AskRequest):
            return JSONResponse(
                status_code=503,
                content={
                    "error": "AskDisabled",
                    "message": "No LLM provider configured on the server.",
                },
            )

    return app


def create_server(
    document: Any,
    options: Dict[str, Any],
    *,
    host: Optional[str] = None,
    port: Optional[int] = None,
    bearer_token: Optional[str] = None,
) -> ServerHandle:
    srv_host = host or options.get("host") or DEFAULT_HOST
    srv_port = port or options.get("port") or DEFAULT_PORT

    # Create server options clone
    server_options = dict(options)
    if bearer_token:
        server_options["bearerToken"] = bearer_token

    app = build_app(document, server_options)

    config = uvicorn.Config(
        app=app, host=srv_host, port=srv_port, log_level="warning"
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run)
    thread.daemon = True
    thread.start()

    url = f"http://{srv_host}:{srv_port}"

    # Wait up to 2 seconds for server to boot
    retries = 20
    while retries > 0 and not server.started:
        time.sleep(0.1)
        retries -= 1

    return ServerHandle(server_thread=thread, uvicorn_server=server, url=url)
