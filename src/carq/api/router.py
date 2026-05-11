"""Router REST API com endpoints de embeddings, busca e ingestão de documentos."""

import asyncio
import hashlib
import logging
import time
import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Header, Query
from sqlalchemy import select

from carq.api.models import (
    EmbedRequest,
    EmbedResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    BatchEmbedRequest,
    BatchEmbedResponse,
    StatsResponse,
    ErrorResponse,
    IngestRequest,
    IngestResponse,
    DocumentStatusResponse,
)
from carq.api.auth import verify_api_key
from carq.core.database import DatabaseManager
from carq.embedding.embedding_dispatcher import (
    EmbeddingDispatcher,
    EmbeddingRequest,
    EmbeddingModel as DispatcherModel,
    EmbeddingError,
)
from carq.embedding.vector_store import VectorStore, VectorStoreError
from carq.embedding.embedding_cache import EmbeddingCache, CacheError

logger = logging.getLogger(__name__)


class EmbeddingAPI:
    """Handler da API de embeddings com dependências."""

    def __init__(
        self,
        dispatcher: EmbeddingDispatcher,
        vector_store: VectorStore,
        cache: EmbeddingCache,
    ):
        self.dispatcher = dispatcher
        self.vector_store = vector_store
        self.cache = cache
        self.logger = logging.getLogger(__name__)

    async def embed_single(
        self,
        request: EmbedRequest,
        api_key: str,
    ) -> EmbedResponse:
        """Gera embedding para um único texto."""
        try:
            # Tenta o cache primeiro
            cached = await self.cache.get(request.text)

            if cached:
                result = EmbedResponse(
                    text=request.text,
                    embedding=cached,
                    model=request.model.value,
                    tokens_used=0,  # Não rastreado para cache
                    cost_usd=0.0,  # Sem custo para cache hit
                    cached=True,
                )
                self.logger.debug(
                    f"Cache hit for embed request",
                    extra={"cached": True},
                )
                return result

            # Gera o embedding
            embed_request = EmbeddingRequest(
                text=request.text,
                model=DispatcherModel(request.model.value),
            )

            embed_result = await self.dispatcher.embed(embed_request)

            # Cacheia o resultado
            try:
                await self.cache.set(request.text, embed_result.embedding)
            except CacheError as e:
                self.logger.warning(f"Falha ao cachear embedding: {e}")

            return EmbedResponse(
                text=embed_result.text,
                embedding=embed_result.embedding,
                model=embed_result.model,
                tokens_used=embed_result.tokens_used,
                cost_usd=embed_result.cost_usd,
                cached=False,
            )

        except EmbeddingError as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Embedding generation failed: {str(e)}",
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in embed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error",
            )

    async def embed_batch(
        self,
        request: BatchEmbedRequest,
        api_key: str,
    ) -> BatchEmbedResponse:
        """Gera embeddings em lote."""
        start_time = time.time()

        try:
            # Converte para requisições do dispatcher
            embed_requests = [
                EmbeddingRequest(
                    text=text,
                    model=DispatcherModel(request.model.value),
                )
                for text in request.texts
            ]

            # Gera os embeddings
            results = await self.dispatcher.embed_batch(embed_requests)

            # Cacheia os resultados
            for result in results:
                try:
                    await self.cache.set(result.text, result.embedding)
                except CacheError:
                    pass  # Continua mesmo se o cache falhar

            # Converte as respostas
            embed_responses = [
                EmbedResponse(
                    text=r.text,
                    embedding=r.embedding,
                    model=r.model,
                    tokens_used=r.tokens_used,
                    cost_usd=r.cost_usd,
                    cached=False,
                )
                for r in results
            ]

            latency = (time.time() - start_time) * 1000  # ms

            return BatchEmbedResponse(
                embeddings=embed_responses,
                count=len(embed_responses),
                total_tokens=sum(r.tokens_used for r in results),
                total_cost_usd=sum(r.cost_usd for r in results),
                batch_latency_ms=latency,
            )

        except EmbeddingError as e:
            self.logger.error(f"Batch embedding failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Batch embedding failed: {str(e)}",
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in batch embed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error",
            )

    async def search(
        self,
        request: SearchRequest,
        api_key: str,
    ) -> SearchResponse:
        """Realiza busca semântica nos embeddings armazenados."""
        start_time = time.time()

        try:
            # Gera o embedding da query
            embed_request = EmbeddingRequest(
                text=request.query,
                model=DispatcherModel(request.model.value),
            )

            embed_result = await self.dispatcher.embed(embed_request)

            # Busca no vector store
            search_results = await self.vector_store.search(
                query_embedding=embed_result.embedding,
                limit=request.limit,
                similarity_threshold=request.similarity_threshold,
            )

            # Converte os resultados
            results = [
                SearchResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    similarity_score=r.similarity_score,
                    metadata=r.metadata,
                )
                for r in search_results
            ]

            latency = (time.time() - start_time) * 1000  # ms

            return SearchResponse(
                query=request.query,
                results=results,
                count=len(results),
                search_latency_ms=latency,
            )

        except EmbeddingError as e:
            self.logger.error(f"Search embedding failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {str(e)}",
            )
        except VectorStoreError as e:
            self.logger.error(f"Vector store search failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Search failed: {str(e)}",
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in search: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error",
            )

    async def get_stats(
        self,
        api_key: str,
    ) -> StatsResponse:
        """Retorna as estatísticas de uso da API."""
        try:
            # Obtém métricas do dispatcher
            dispatcher_metrics = self.dispatcher.get_metrics()

            # Obtém estatísticas do cache
            cache_stats = await self.cache.get_stats()

            # Obtém contagem do vector store
            vector_count = await self.vector_store.count()

            return StatsResponse(
                embeddings_generated=dispatcher_metrics["total_embeddings"],
                total_tokens_used=dispatcher_metrics["total_tokens"],
                total_cost_usd=dispatcher_metrics["total_cost_usd"],
                avg_cost_per_embedding=dispatcher_metrics["avg_cost_per_embedding"],
                cache_hit_rate=cache_stats.hit_rate,
                cached_embeddings=cache_stats.size,
                stored_vectors=vector_count,
            )

        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve statistics",
            )


def create_router(
    dispatcher: EmbeddingDispatcher,
    vector_store: VectorStore,
    cache: EmbeddingCache,
    db_manager: Optional[DatabaseManager] = None,
) -> APIRouter:
    """Cria e configura o router da API."""
    router = APIRouter(prefix="/api/v1", tags=["embeddings"])
    api = EmbeddingAPI(dispatcher, vector_store, cache)

    @router.post(
        "/embed",
        response_model=EmbedResponse,
        summary="Gerar embedding único",
        responses={
            200: {"description": "Embedding generated successfully"},
            400: {"model": ErrorResponse, "description": "Invalid request"},
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            500: {"model": ErrorResponse, "description": "Server error"},
        },
    )
    async def embed(
        request: EmbedRequest,
        api_key: Annotated[str, Depends(verify_api_key)],
    ) -> EmbedResponse:
        """Gera embedding para um único texto."""
        return await api.embed_single(request, api_key)

    @router.post(
        "/embed-batch",
        response_model=BatchEmbedResponse,
        summary="Gerar embeddings em lote",
        responses={
            200: {"description": "Embeddings generated successfully"},
            400: {"model": ErrorResponse, "description": "Invalid request"},
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            500: {"model": ErrorResponse, "description": "Server error"},
        },
    )
    async def embed_batch(
        request: BatchEmbedRequest,
        api_key: Annotated[str, Depends(verify_api_key)],
    ) -> BatchEmbedResponse:
        """Gera embeddings para múltiplos textos."""
        return await api.embed_batch(request, api_key)

    @router.post(
        "/search",
        response_model=SearchResponse,
        summary="Busca semântica",
        responses={
            200: {"description": "Search results returned"},
            400: {"model": ErrorResponse, "description": "Invalid request"},
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            500: {"model": ErrorResponse, "description": "Server error"},
        },
    )
    async def search(
        request: SearchRequest,
        api_key: Annotated[str, Depends(verify_api_key)],
    ) -> SearchResponse:
        """Realiza busca semântica nos embeddings armazenados."""
        return await api.search(request, api_key)

    @router.get(
        "/stats",
        response_model=StatsResponse,
        summary="Obter estatísticas da API",
        responses={
            200: {"description": "Statistics retrieved"},
            401: {"model": ErrorResponse, "description": "Unauthorized"},
            500: {"model": ErrorResponse, "description": "Server error"},
        },
    )
    async def stats(
        api_key: Annotated[str, Depends(verify_api_key)],
    ) -> StatsResponse:
        """Retorna estatísticas de uso da API."""
        return await api.get_stats(api_key)

    @router.get(
        "/health",
        summary="Verificação de saúde",
        responses={200: {"description": "Serviço saudável"}},
    )
    async def health():
        """Endpoint de verificação de saúde."""
        return {"status": "healthy"}

    @router.post(
        "/documents",
        response_model=IngestResponse,
        status_code=status.HTTP_202_ACCEPTED,
        summary="Ingerir documento no pipeline RAG",
        tags=["documents"],
        responses={
            202: {"description": "Documento aceito para processamento"},
            400: {"model": ErrorResponse, "description": "Requisição inválida"},
            401: {"model": ErrorResponse, "description": "Não autorizado"},
            500: {"model": ErrorResponse, "description": "Erro no servidor"},
        },
    )
    async def ingest_document(
        request: IngestRequest,
        api_key: Annotated[str, Depends(verify_api_key)],
    ) -> IngestResponse:
        """Submete um documento ao pipeline de processamento RAG (processamento assíncrono)."""
        if db_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Document ingestion not available (database not configured)",
            )

        from carq.models.models import Document, DocumentStatus, ProcessingTask, TaskType, TaskStatus

        # Deriva hash do conteúdo para deduplicação
        content_for_hash = (request.content or request.source_uri).encode("utf-8")
        content_hash = hashlib.sha256(content_for_hash).digest()

        try:
            async with db_manager.get_session() as session:
                # Cria o registro do documento
                doc = Document(
                    source_uri=request.source_uri,
                    content_hash=content_hash,
                    status=DocumentStatus.PENDING,
                    document_type=request.document_type.value,
                    attributes=request.attributes or {},
                )
                session.add(doc)
                await session.flush()  # obtém doc.id antes de criar a tarefa

                # Cria a tarefa de processamento inicial
                task_type = (
                    TaskType.PARSE_PDF
                    if request.document_type.value == "pdf"
                    else TaskType.EMBED_CHUNK
                )
                task = ProcessingTask(
                    document_id=doc.id,
                    task_type=task_type,
                    status=TaskStatus.PENDING,
                    priority=request.priority,
                    attributes={
                        "source_uri": request.source_uri,
                        "document_type": request.document_type.value,
                        **({"content": request.content} if request.content else {}),
                    },
                )
                session.add(task)
                await session.commit()

                logger.info(
                    f"Document accepted for ingestion",
                    extra={"document_id": str(doc.id), "task_id": str(task.id)},
                )

                return IngestResponse(
                    document_id=str(doc.id),
                    status=DocumentStatus.PENDING.value,
                    message="Document accepted for processing",
                    task_id=str(task.id),
                )

        except Exception as exc:
            logger.error(f"Failed to create document: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to accept document for processing",
            )

    @router.get(
        "/documents/{document_id}",
        response_model=DocumentStatusResponse,
        summary="Obter status de processamento do documento",
        tags=["documents"],
        responses={
            200: {"description": "Status do documento"},
            401: {"model": ErrorResponse, "description": "Não autorizado"},
            404: {"model": ErrorResponse, "description": "Documento não encontrado"},
            500: {"model": ErrorResponse, "description": "Erro no servidor"},
        },
    )
    async def get_document_status(
        document_id: str,
        api_key: Annotated[str, Depends(verify_api_key)],
    ) -> DocumentStatusResponse:
        """Retorna o status atual de processamento de um documento."""
        if db_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not configured",
            )

        from carq.models.models import Document

        try:
            doc_uuid = uuid.UUID(document_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid document ID format",
            )

        try:
            async with db_manager.get_session() as session:
                from sqlalchemy import select
                result = await session.execute(
                    select(Document).where(Document.id == doc_uuid)
                )
                doc = result.scalar_one_or_none()

                if doc is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Document {document_id} not found",
                    )

                return DocumentStatusResponse(
                    document_id=str(doc.id),
                    status=doc.status.value if hasattr(doc.status, "value") else str(doc.status),
                    document_type=doc.document_type,
                    source_uri=doc.source_uri,
                    chunk_count=doc.chunk_count,
                    embedding_count=doc.embedding_count,
                    progress_percentage=doc.progress_percentage,
                    error_message=doc.error_message,
                )
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Failed to fetch document status: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve document status",
            )

    @router.get(
        "/documents",
        summary="Listar documentos",
        tags=["documents"],
    )
    async def list_documents(
        api_key: Annotated[str, Depends(verify_api_key)],
        limit: int = 20,
        offset: int = 0,
        status_filter: Optional[str] = Query(default=None, alias="status"),
    ):
        """Lista documentos ingeridos com paginação."""
        if db_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not configured",
            )

        from carq.models.models import Document

        try:
            async with db_manager.get_session() as session:
                stmt = select(Document).order_by(Document.created_at.desc()).limit(min(max(limit, 1), 100)).offset(max(offset, 0))
                if status_filter:
                    stmt = stmt.where(Document.status == status_filter)
                result = await session.execute(stmt)
                docs = result.scalars().all()

                return {
                    "documents": [
                        {
                            "id": str(d.id),
                            "source_uri": d.source_uri,
                            "status": d.status.value if hasattr(d.status, "value") else str(d.status),
                            "document_type": d.document_type,
                            "created_at": d.created_at.isoformat() if d.created_at else None,
                            "updated_at": d.updated_at.isoformat() if d.updated_at else None,
                        }
                        for d in docs
                    ],
                    "count": len(docs),
                    "limit": min(max(limit, 1), 100),
                    "offset": max(offset, 0),
                }
        except Exception as exc:
            logger.error(f"Failed to list documents: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to list documents",
            )

    @router.post(
        "/documents/{document_id}/retry",
        summary="Retentar processamento do documento",
        tags=["documents"],
    )
    async def retry_document(
        document_id: str,
        api_key: Annotated[str, Depends(verify_api_key)],
    ):
        """Recoloca na fila as tarefas de processamento de um documento."""
        if db_manager is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Database not configured",
            )

        from carq.models.models import Document, ProcessingTask, TaskStatus, DocumentStatus

        try:
            doc_uuid = uuid.UUID(document_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid document ID format",
            )

        try:
            async with db_manager.get_session() as session:
                doc_result = await session.execute(select(Document).where(Document.id == doc_uuid))
                doc = doc_result.scalar_one_or_none()
                if doc is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Document {document_id} not found",
                    )

                task_result = await session.execute(
                    select(ProcessingTask).where(ProcessingTask.document_id == doc_uuid).order_by(ProcessingTask.created_at.desc())
                )
                tasks = task_result.scalars().all()
                if not tasks:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"No tasks found for document {document_id}",
                    )

                for task in tasks:
                    task.status = TaskStatus.PENDING
                    task.worker_id = None
                    task.error_message = None
                    if (task.attributes or {}).get("dlq"):
                        attrs = task.attributes or {}
                        attrs.pop("dlq", None)
                        attrs.pop("dlq_at", None)
                        task.attributes = attrs

                doc.status = DocumentStatus.PENDING
                await session.commit()

                return {
                    "document_id": document_id,
                    "status": "queued",
                    "tasks_requeued": len(tasks),
                }
        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"Failed to retry document: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retry document",
            )

    return router
