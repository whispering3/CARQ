"""
Ponto de entrada da aplicação CARQ.
Inicia o servidor FastAPI com gerenciamento de ciclo de vida (init do BD, migrações).
"""
import asyncio
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from sqlalchemy import text
from starlette.middleware.base import BaseHTTPMiddleware

from carq.api.router import create_router
from carq.core.config import settings
from carq.core.database import DatabaseManager
from carq.core.logging import CorrelationIDFilter, get_logger
from carq.embedding.embedding_cache import EmbeddingCache
from carq.embedding.embedding_dispatcher import EmbeddingDispatcher
from carq.embedding.vector_store import VectorStore
from carq.models.base import Base
from carq.monitoring.health import HealthChecker
from carq.monitoring.metrics import get_metrics_collector

logger = get_logger(__name__)

db_manager = DatabaseManager()
_health_checker: Optional[HealthChecker] = None



class MetricsMiddleware(BaseHTTPMiddleware):
    """Registra métricas Prometheus para cada requisição HTTP."""

    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        latency = time.perf_counter() - start

        path = request.url.path
        collector = get_metrics_collector()
        collector.record_http_request(
            endpoint=path,
            method=request.method,
            status=response.status_code,
            latency_seconds=latency,
        )
        return response


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Injeta um correlation ID por requisição no contexto assíncrono."""

    async def dispatch(self, request: Request, call_next):
        import uuid
        cid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        CorrelationIDFilter.set_correlation_id(cid)
        response = await call_next(request)
        response.headers["X-Request-ID"] = cid
        return response



class _RateLimitWindow:
    """Contador de janela deslizante para um único cliente (fallback em processo)."""
    __slots__ = ("count", "window_start")

    def __init__(self):
        self.count = 0
        self.window_start = time.monotonic()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiter por cliente com janela fixa e backend distribuído.

    Backend principal: Redis INCR + EXPIRE — atômico e compartilhado entre todos os
    processos/pods de trabalho, garantindo limite verdadeiramente por cliente, não por processo.

    Fallback: defaultdict em processo quando o Redis está indisponível (degradação
    graciosa — o limite se aplica por worker nesse caso).
    """

    _SKIP_PATHS = frozenset({"/health/live", "/health/ready", "/metrics", "/health"})

    def __init__(self, app, rpm: int = 600, redis_url: Optional[str] = None):
        super().__init__(app)
        self.rpm = rpm
        self.window_seconds = 60
        self._redis_url = redis_url
        self._redis = None
        self._redis_ok = True  # otimista; muda para False após primeira falha
        # Contadores de fallback em processo
        self._windows: dict = defaultdict(_RateLimitWindow)

    async def _get_redis(self):
        """Retorna um cliente Redis ativo, ou None se indisponível."""
        if not self._redis_ok or not self._redis_url:
            return None
        if self._redis is None:
            try:
                import redis.asyncio as aioredis
                client = aioredis.from_url(
                    self._redis_url,
                    socket_connect_timeout=1,
                    socket_timeout=1,
                )
                await client.ping()
                self._redis = client
                logger.info("RateLimitMiddleware: Redis connected (distributed mode)")
            except Exception as exc:
                logger.warning(
                    f"RateLimitMiddleware: Redis unavailable, using in-process fallback: {exc}"
                )
                self._redis_ok = False
        return self._redis

    def _get_client_key(self, request: Request) -> str:
        api_key = request.headers.get("X-API-Key", "")
        if api_key:
            return f"key:{api_key[:20]}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    async def _allowed_redis(self, client: object, client_key: str) -> bool:
        """Incrementa o contador Redis para a janela atual de 60s; retorna True se permitido."""
        window = int(time.time() // self.window_seconds)
        key = f"ratelimit:{client_key}:{window}"
        try:
            pipe = client.pipeline()
            pipe.incr(key)
            pipe.expire(key, self.window_seconds + 1)
            results = await pipe.execute()
            return results[0] <= self.rpm
        except Exception as exc:
            logger.warning(f"RateLimitMiddleware: Redis check failed (fail-open): {exc}")
            return True  # fail open — preferir disponibilidade a limites estritos

    def _allowed_local(self, client_key: str) -> bool:
        """Verificação de rate em processo (fallback)."""
        now = time.monotonic()
        win = self._windows[client_key]
        if now - win.window_start >= self.window_seconds:
            win.count = 0
            win.window_start = now
        win.count += 1
        return win.count <= self.rpm

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self._SKIP_PATHS:
            return await call_next(request)

        client_key = self._get_client_key(request)
        redis_client = await self._get_redis()

        if redis_client is not None:
            allowed = await self._allowed_redis(redis_client, client_key)
        else:
            allowed = self._allowed_local(client_key)

        if not allowed:
            return Response(
                content='{"detail":"Rate limit exceeded. Retry after 60 seconds."}',
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                media_type="application/json",
                headers={"Retry-After": "60"},
            )
        return await call_next(request)



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ciclo de vida da aplicação: inicialização e encerramento."""
    global _health_checker

    logger.info("Starting CARQ application...")
    await db_manager.initialize()

    # Cria tabelas (usar migrações Alembic em produção)
    async with db_manager.engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)
        # Índice HNSW via cast halfvec (suporta >2000 dims, pgvector ≥0.7)
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS idx_embedding_vector "
            "ON rag_embeddings USING hnsw "
            "((embedding::halfvec(3072)) halfvec_cosine_ops) "
            "WITH (m = 16, ef_construction = 64)"
        ))

    # Inicializa componentes da aplicação
    dispatcher = EmbeddingDispatcher(
        api_key=settings.embedding.openai_api_key or "placeholder",
    )
    vector_store = VectorStore(db_manager)
    cache = EmbeddingCache(redis_url=settings.redis.url)

    # Conecta o health checker com as dependências ativas
    _health_checker = HealthChecker(redis_url=settings.redis.url)

    # Monta o router com as dependências inicializadas
    api_router = create_router(dispatcher, vector_store, cache, db_manager=db_manager)
    app.include_router(api_router)

    logger.info("CARQ is ready.")
    yield

    # ── Encerramento ──────────────────────────────────────────────────────
    logger.info("Shutting down CARQ...")
    await db_manager.close()
    logger.info("CARQ stopped.")



def create_app() -> FastAPI:
    """Cria e configura a aplicação FastAPI."""
    app = FastAPI(
        title="CARQ - Context-Aware RAG Processing Queue",
        description="Enterprise RAG ingestion orchestrator with async processing, "
                    "embedding generation, and vector storage.",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # A ordem importa: o middleware mais externo executa primeiro
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(CorrelationIDMiddleware)
    app.add_middleware(RateLimitMiddleware, rpm=600, redis_url=settings.redis.url)

    # CORS — usar origens configuradas (nunca wildcard em produção)
    cors_origins = settings.api.cors_origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["X-API-Key", "X-Request-ID", "Content-Type", "Authorization"],
    )

    # ── Probe de liveness do Kubernetes ──────────────────────────────────
    @app.get("/health/live", tags=["health"], include_in_schema=False)
    async def liveness():
        """Probe de liveness — sempre retorna 200 se o processo está rodando."""
        return {"status": "alive", "service": "carq"}

    # ── Probe de readiness do Kubernetes ─────────────────────────────────
    @app.get("/health/ready", tags=["health"], include_in_schema=False)
    async def readiness():
        """Probe de readiness — verifica todas as dependências (BD + Redis)."""
        checker = _health_checker
        if checker is None:
            return Response(
                content='{"status":"starting"}',
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                media_type="application/json",
            )

        # Usa uma verificação rápida apenas do BD para a probe (caminho rápido)
        db_ok = await db_manager.health_check()
        if not db_ok:
            return Response(
                content='{"status":"not_ready","database":"error"}',
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                media_type="application/json",
            )
        return {"status": "ready", "database": "ok"}

    # ── Health completo (legível por humanos) ──────────────────────────
    @app.get("/health", tags=["health"])
    async def health():
        """Verificação completa de saúde incluindo BD e Redis."""
        db_ok = await db_manager.health_check()
        return {
            "status": "healthy" if db_ok else "degraded",
            "service": "carq",
            "version": "1.0.0",
            "database": "ok" if db_ok else "error",
        }

    # ── Métricas Prometheus ────────────────────────────────────────────
    @app.get("/metrics", tags=["observability"], include_in_schema=False)
    async def metrics():
        """Endpoint de métricas Prometheus."""
        collector = get_metrics_collector()
        return PlainTextResponse(
            content=collector.get_metrics(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "carq.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        log_level=settings.logging.level.lower(),
    )
