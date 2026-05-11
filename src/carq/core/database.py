"""Pool de conexões e gerenciamento de sessões do banco de dados com SQLAlchemy 2.0 assíncrono."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool, AsyncAdaptedQueuePool

from carq.core.config import settings
from carq.core.exceptions import ConnectionPoolError, DatabaseError
from carq.core.logging import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """Gerencia conexões e sessões do banco de dados."""

    def __init__(self):
        self._engine = None
        self._session_factory = None
        self._initialized = False

    async def initialize(self) -> None:
        """Inicializa o engine do banco e cria o pool de conexões."""
        if self._initialized:
            return

        try:
            db_config = settings.database

            if settings.environment == "testing":
                poolclass = NullPool
            else:
                poolclass = AsyncAdaptedQueuePool

            # Cria engine assíncrono (NullPool não aceita args de dimensionamento de pool)
            pool_kwargs = {}
            if poolclass is not NullPool:
                pool_kwargs = {
                    "pool_size": db_config.pool_size,
                    "max_overflow": db_config.max_overflow,
                    "pool_timeout": db_config.pool_timeout,
                    "pool_recycle": 3600,
                    "pool_pre_ping": True,
                }

            self._engine = create_async_engine(
                db_config.url,
                echo=db_config.echo,
                poolclass=poolclass,
                **pool_kwargs,
            )

            self._setup_connection_handlers()

            self._session_factory = async_sessionmaker(
                self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )

            await self._test_connection()

            logger.info("Database initialized successfully", extra={
                "host": db_config.host,
                "port": db_config.port,
                "database": db_config.database,
            })

            self._initialized = True

        except Exception as e:
            logger.error("Failed to initialize database", extra={
                "error": str(e),
            })
            raise DatabaseError(f"Database initialization failed: {str(e)}")

    async def _test_connection(self) -> None:
        try:
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
                logger.info("Database connection test successful")
        except Exception as e:
            raise DatabaseError(f"Database connection test failed: {str(e)}")

    def _setup_connection_handlers(self) -> None:
        """Configura os handlers de eventos do SQLAlchemy para gerenciamento de conexão."""

        @event.listens_for(self._engine.sync_engine, "engine_disposed")
        def receive_engine_disposed(engine):
            logger.warning("Engine do banco de dados descartado")

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Gerenciador de contexto que retorna uma sessão do banco de dados."""
        if not self._initialized:
            raise DatabaseError("Banco de dados não inicializado. Chame initialize() primeiro.")

        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", extra={
                "error": str(e),
            })
            raise
        finally:
            await session.close()

    async def close(self) -> None:
        """Fecha o engine do banco e libera recursos."""
        if self._engine:
            await self._engine.dispose()
            self._initialized = False
            logger.info("Database connections closed")

    async def health_check(self, timeout_seconds: float = 5.0) -> bool:
        """Verifica a saúde do banco de dados com timeout.

        CORREÇÃO CRÍTICA: timeout evita que o health check trave a probe de liveness.
        """
        try:
            async with asyncio.timeout(timeout_seconds):
                async with self.get_session() as session:
                    await session.execute(text("SELECT 1"))
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "Database health check timeout",
                extra={"timeout_seconds": timeout_seconds},
            )
            return False
        except Exception as e:
            logger.error(
                "Database health check failed",
                extra={"error": str(e)},
            )
            return False

    async def get_pool_status(self) -> dict:
        """Retorna o status do pool de conexões usando a API pública do SQLAlchemy."""
        if not self._engine:
            return {"status": "not_initialized"}

        pool = self._engine.pool
        return {
            "pool_type": pool.__class__.__name__,
            "size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
        }

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def engine(self):
        if not self._initialized:
            raise DatabaseError("Banco de dados não inicializado")
        return self._engine


# Instância global do gerenciador de banco de dados
db = DatabaseManager()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Injeção de dependência para sessão do banco de dados no FastAPI."""
    async with db.get_session() as session:
        yield session
