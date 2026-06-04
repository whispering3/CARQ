"""Vector store assíncrono com pgvector para busca semântica por similaridade cosseno."""

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from sqlalchemy import func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import select

from carq.core.exceptions import ProcessingError
from carq.models.models import Embedding

logger = logging.getLogger(__name__)


class VectorStoreError(ProcessingError):
    """Levantada quando operações do vector store falham."""

    pass


@dataclass
class SearchResult:
    """Resultado de busca semântica."""

    chunk_id: str
    text: str
    embedding: list[float]
    similarity_score: float
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class EmbeddingRecord:
    """Registro de embedding armazenado."""

    chunk_id: str
    text: str
    embedding: list[float]
    model: str
    tokens_used: int
    cost_usd: float
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class VectorStore:
    """Vector store assíncrono utilizando a extensão pgvector."""

    def __init__(self, session: Any):
        """Inicializa o vector store.

        Args:
            session: AsyncSession (para testes / uso por requisição) ou
                     DatabaseManager com método get_session() (para instâncias
                     de produção de longa duração).
        """
        # Suporta tanto AsyncSession diretamente quanto DatabaseManager (fábrica de sessões)
        from carq.core.database import DatabaseManager
        if isinstance(session, DatabaseManager):
            self._db_manager = session
            self.session = None
        else:
            self._db_manager = None
            self.session = session
        self.logger = logging.getLogger(__name__)

    @asynccontextmanager
    async def _get_session(self):
        """Gerenciador de contexto que fornece uma AsyncSession ativa."""
        if self._db_manager is not None:
            async with self._db_manager.get_session() as sess:
                yield sess
        else:
            yield self.session

    async def insert(
        self,
        chunk_id: str,
        text: str,
        embedding: list[float],
        model: str,
        tokens_used: int,
        cost_usd: float,
        metadata: Optional[dict] = None,
        _session: Optional[AsyncSession] = None,
    ) -> str:
        """Insere embedding no vector store."""
        async def _do_insert(session):
            db_embedding = Embedding(
                chunk_id=chunk_id,
                text=text,
                embedding=embedding,
                model=model,
                tokens_used=tokens_used,
                cost_usd=cost_usd,
                attributes=metadata or {},
            )
            session.add(db_embedding)
            await session.flush()
            self.logger.debug(
                f"Inserted embedding for chunk {chunk_id}",
                extra={"chunk_id": chunk_id, "model": model, "embedding_id": db_embedding.id},
            )
            return str(db_embedding.id)

        try:
            if _session is not None:
                return await _do_insert(_session)
            async with self._get_session() as session:
                return await _do_insert(session)
        except VectorStoreError:
            raise
        except Exception as e:
            self.logger.error(
                f"Failed to insert embedding: {e}",
                extra={"chunk_id": chunk_id, "error": str(e)},
            )
            raise VectorStoreError(f"Failed to insert embedding: {e}") from e

    async def insert_batch(
        self,
        records: list[EmbeddingRecord],
    ) -> list[str]:
        """Insere lote de embeddings usando savepoint para atomicidade tudo-ou-nada."""
        try:
            async with self._get_session() as session:
                embedding_ids = []
                async with session.begin_nested():
                    for record in records:
                        eid = await self.insert(
                            chunk_id=record.chunk_id,
                            text=record.text,
                            embedding=record.embedding,
                            model=record.model,
                            tokens_used=record.tokens_used,
                            cost_usd=record.cost_usd,
                            metadata=record.metadata,
                            _session=session,
                        )
                        embedding_ids.append(eid)
                await session.commit()
                self.logger.info(
                    f"Inserted batch of {len(records)} embeddings",
                    extra={"batch_size": len(records)},
                )
                return embedding_ids
        except Exception as e:
            self.logger.error(
                f"Batch insert failed: {e}",
                extra={"batch_size": len(records), "error_type": type(e).__name__},
            )
            raise VectorStoreError(f"Batch insert failed: {e}") from e

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
    ) -> list[SearchResult]:
        """Busca semântica usando similaridade cosseno."""
        try:
            # pgvector usa o operador <=> para distância cosseno
            # pgvector cosine distance is 0 for identical normalized vectors and 1 - cosine similarity.
            query_str = """
            SELECT
                e.id,
                e.chunk_id,
                e.text,
                e.embedding,
                (1 - (e.embedding <=> :query_embedding::vector)) as similarity_score,
                e.attributes
            FROM rag_embeddings e
            WHERE (1 - (e.embedding <=> :query_embedding::vector)) >= :threshold
            ORDER BY similarity_score DESC
            LIMIT :limit
            """
            async with self._get_session() as session:
                result = await session.execute(
                    text(query_str),
                    {"query_embedding": query_embedding, "threshold": similarity_threshold, "limit": limit},
                )
                rows = result.fetchall()

            results = []
            for row in rows:
                similarity_score = max(0.0, min(1.0, float(row[4])))
                results.append(
                    SearchResult(
                        chunk_id=row[1],
                        text=row[2],
                        embedding=row[3],
                        similarity_score=similarity_score,
                        metadata=row[5],
                    )
                )

            self.logger.debug(
                f"Search returned {len(results)} results",
                extra={"results": len(results), "threshold": similarity_threshold},
            )
            return results

        except VectorStoreError:
            raise
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise VectorStoreError(f"Search failed: {e}") from e

    async def update(
        self,
        chunk_id: str,
        embedding: list[float],
        metadata: Optional[dict] = None,
    ) -> bool:
        """Atualiza embedding existente pelo ID do chunk."""
        try:
            async with self._get_session() as session:
                stmt = select(Embedding).where(Embedding.chunk_id == chunk_id)
                result = await session.execute(stmt)
                db_embedding = result.scalar_one_or_none()

                if not db_embedding:
                    self.logger.warning(
                        f"Embedding not found for chunk {chunk_id}",
                        extra={"chunk_id": chunk_id},
                    )
                    return False

                db_embedding.embedding = embedding
                if metadata:
                    db_embedding.attributes.update(metadata)
                await session.commit()

            self.logger.debug(f"Updated embedding for chunk {chunk_id}", extra={"chunk_id": chunk_id})
            return True

        except Exception as e:
            self.logger.error(f"Update failed: {e}")
            raise VectorStoreError(f"Update failed: {e}") from e

    async def delete(self, chunk_id: str) -> bool:
        """Remove embedding pelo ID do chunk."""
        try:
            async with self._get_session() as session:
                stmt = select(Embedding).where(Embedding.chunk_id == chunk_id)
                result = await session.execute(stmt)
                db_embedding = result.scalar_one_or_none()

                if not db_embedding:
                    return False

                await session.delete(db_embedding)
                await session.commit()

            self.logger.debug(f"Deleted embedding for chunk {chunk_id}", extra={"chunk_id": chunk_id})
            return True

        except Exception as e:
            self.logger.error(f"Delete failed: {e}")
            raise VectorStoreError(f"Delete failed: {e}") from e

    async def get_by_chunk_id(self, chunk_id: str) -> Optional[EmbeddingRecord]:
        """Busca embedding pelo ID do chunk. Retorna None se não encontrado."""
        try:
            async with self._get_session() as session:
                stmt = select(Embedding).where(Embedding.chunk_id == chunk_id)
                result = await session.execute(stmt)
                db_embedding = result.scalar_one_or_none()

            if not db_embedding:
                return None

            return EmbeddingRecord(
                chunk_id=db_embedding.chunk_id,
                text=db_embedding.text,
                embedding=db_embedding.embedding,
                model=db_embedding.model,
                tokens_used=db_embedding.tokens_used,
                cost_usd=db_embedding.cost_usd,
                metadata=db_embedding.attributes,
            )

        except Exception as e:
            self.logger.error(f"Get failed: {e}")
            raise VectorStoreError(f"Get failed: {e}") from e

    async def count(self) -> int:
        try:
            async with self._get_session() as session:
                stmt = select(func.count(Embedding.id))
                result = await session.execute(stmt)
                count = result.scalar()
            return count or 0

        except Exception as e:
            self.logger.error(f"Count failed: {e}")
            raise VectorStoreError(f"Count failed: {e}") from e

    async def delete_all(self) -> int:
        """Remove todos os embeddings com um único DELETE em massa (sem N+1)."""
        from sqlalchemy import delete

        try:
            async with self._get_session() as session:
                # Conta antes de remover para retornar o número removido
                count_result = await session.execute(select(func.count(Embedding.id)))
                count = count_result.scalar() or 0

                await session.execute(delete(Embedding))
                await session.commit()

            self.logger.warning("Deleted all embeddings", extra={"count": count})
            return count

        except Exception as e:
            self.logger.error(f"Delete all failed: {e}")
            raise VectorStoreError(f"Delete all failed: {e}") from e
