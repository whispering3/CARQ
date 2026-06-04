"""Executable CARQ worker process.

Consumes tasks from PostgreSQL and runs the ingestion pipeline:
document source -> chunks -> embeddings -> vector store.
"""

import asyncio
import hashlib
import signal
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx
from sqlalchemy import select

from carq.core.config import settings
from carq.core.database import DatabaseManager
from carq.core.logging import get_logger
from carq.embedding.embedding_dispatcher import (
    EmbeddingDispatcher,
    EmbeddingModel,
    EmbeddingRequest,
)
from carq.embedding.vector_store import VectorStore
from carq.models.models import (
    Chunk,
    ChunkStatus,
    Document,
    DocumentStatus,
    ProcessingTask,
    TaskType,
)
from carq.pdf.pdf_parser import PDFParser
from carq.pdf.semantic_chunker import SemanticChunker
from carq.queue.queue_manager import QueueManager
from carq.worker.task_coordinator import CoordinatorConfig, TaskCoordinator
from carq.worker.worker_pool import WorkerConfig, WorkerPool

logger = get_logger(__name__)


class PipelineHandlers:
    """Task handlers for the durable PostgreSQL-backed queue."""

    def __init__(self, db_manager: DatabaseManager):
        if not settings.embedding.openai_api_key:
            raise RuntimeError("CARQ_EMBEDDING_OPENAI_API_KEY is required for carq-worker")
        self.db_manager = db_manager
        self.dispatcher = EmbeddingDispatcher(api_key=settings.embedding.openai_api_key)
        self.vector_store = VectorStore(db_manager)
        self.pdf_parser = PDFParser(max_workers=settings.worker.pdf_parser_workers)
        self.chunker = SemanticChunker()
        self.embedding_model = EmbeddingModel(settings.embedding.openai_model)

    async def parse_pdf(self, task: ProcessingTask) -> None:
        attrs = task.attributes or {}
        source_uri = attrs.get("source_uri")
        if not source_uri:
            raise ValueError("parse_pdf task missing source_uri")

        file_path = self._resolve_local_file(source_uri)
        pages, metadata = await self.pdf_parser.parse(file_path)
        content = "\n\n".join(page.text for page in pages if page.text.strip())
        document_attrs = {
            "pdf": {
                "filename": metadata.filename,
                "num_pages": metadata.num_pages,
                "file_size_bytes": metadata.file_size_bytes,
                "content_hash": metadata.content_hash,
            }
        }
        await self._chunk_document(task.document_id, content, document_attrs)

    async def chunk_document(self, task: ProcessingTask) -> None:
        attrs = task.attributes or {}
        source_uri = attrs.get("source_uri")
        content = attrs.get("content")

        if not content:
            content = await self._load_text_source(source_uri)

        await self._chunk_document(task.document_id, content, {})

    async def embed_chunk(self, task: ProcessingTask) -> None:
        attrs = task.attributes or {}
        raw_chunk_id = attrs.get("chunk_id")
        if not raw_chunk_id:
            raise ValueError("embed_chunk task missing chunk_id")

        chunk_id = uuid.UUID(str(raw_chunk_id))
        async with self.db_manager.get_session() as session:
            chunk = await session.get(Chunk, chunk_id)
            if chunk is None:
                raise ValueError(f"Chunk {chunk_id} not found")

            chunk.status = ChunkStatus.EMBEDDING
            result = await self.dispatcher.embed(
                EmbeddingRequest(text=chunk.content, model=self.embedding_model, chunk_id=str(chunk.id))
            )
            await self.vector_store.insert(
                chunk_id=str(chunk.id),
                text=chunk.content,
                embedding=result.embedding,
                model=result.model,
                tokens_used=result.tokens_used,
                cost_usd=result.cost_usd,
                metadata={"task_id": str(task.id), **(chunk.attributes or {})},
                _session=session,
            )
            chunk.status = ChunkStatus.DONE
            await self._refresh_document_progress(session, chunk.document_id)

    async def _chunk_document(
        self,
        document_id: uuid.UUID,
        content: str,
        extra_attributes: dict,
    ) -> None:
        if not content or not content.strip():
            raise ValueError("Document content is empty")

        chunks = self.chunker.chunk(content)
        if not chunks:
            raise ValueError("Chunker produced no chunks")

        async with self.db_manager.get_session() as session:
            document = await session.get(Document, document_id)
            if document is None:
                raise ValueError(f"Document {document_id} not found")

            document.status = DocumentStatus.CHUNKING
            document.attributes = {**(document.attributes or {}), **extra_attributes}
            await session.flush()

            queue = QueueManager(session)
            for chunk_data in chunks:
                chunk = Chunk(
                    document_id=document_id,
                    chunk_index=chunk_data.chunk_num,
                    content=chunk_data.text,
                    content_hash=hashlib.sha256(chunk_data.text.encode("utf-8")).digest(),
                    status=ChunkStatus.PENDING,
                    tokens=chunk_data.tokens_estimate,
                    attributes={
                        "start_char": chunk_data.start_char,
                        "end_char": chunk_data.end_char,
                        "page_num": chunk_data.page_num,
                        "context": chunk_data.context,
                    },
                )
                session.add(chunk)
                await session.flush()
                await queue.enqueue_task(
                    document_id=document_id,
                    task_type=TaskType.EMBED_CHUNK,
                    priority=0,
                    metadata={"chunk_id": str(chunk.id)},
                )

            document.status = DocumentStatus.EMBEDDING

    async def _refresh_document_progress(self, session, document_id: uuid.UUID) -> None:
        result = await session.execute(select(Chunk.status).where(Chunk.document_id == document_id))
        statuses = [row[0] for row in result.all()]
        document = await session.get(Document, document_id)
        if document is None:
            return
        if statuses and all(status == ChunkStatus.DONE for status in statuses):
            document.status = DocumentStatus.DONE
            document.completed_at = datetime.now(timezone.utc)
        else:
            document.status = DocumentStatus.EMBEDDING

    async def _load_text_source(self, source_uri: str | None) -> str:
        if not source_uri:
            raise ValueError("Text document missing content and source_uri")

        parsed = urlparse(source_uri)
        if parsed.scheme in {"http", "https"}:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(source_uri)
                response.raise_for_status()
                return response.text

        return self._resolve_local_file(source_uri).read_text(encoding="utf-8")

    @staticmethod
    def _resolve_local_file(source_uri: str) -> Path:
        parsed = urlparse(source_uri)
        if parsed.scheme == "file":
            path = Path(unquote(parsed.path))
        elif parsed.scheme == "":
            path = Path(source_uri)
        else:
            raise ValueError(f"Unsupported document source for worker: {source_uri}")

        if not path.exists():
            raise FileNotFoundError(f"Document source not found: {path}")
        return path


async def run_worker() -> None:
    db_manager = DatabaseManager()
    await db_manager.initialize()

    handlers = PipelineHandlers(db_manager)
    worker_id = f"carq-worker-{uuid.uuid4()}"
    pool = WorkerPool(
        config=WorkerConfig(
            worker_id=worker_id,
            max_workers=max(
                settings.worker.pdf_parser_workers,
                settings.worker.chunk_workers,
                settings.worker.embedding_workers,
                settings.worker.vector_insert_workers,
            ),
            task_timeout=settings.worker.task_timeout,
            max_retries=settings.worker.retry_max_attempts,
        )
    )
    pool.register_handler(TaskType.PARSE_PDF, handlers.parse_pdf)
    pool.register_handler(TaskType.CHUNK_DOCUMENT, handlers.chunk_document)
    pool.register_handler(TaskType.EMBED_CHUNK, handlers.embed_chunk)

    coordinator = TaskCoordinator(
        db_manager=db_manager,
        config=CoordinatorConfig(stuck_task_timeout=settings.worker.task_timeout),
    )
    coordinator.register_worker_pool(pool)

    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signame in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signame, None)
        if signum is not None:
            try:
                loop.add_signal_handler(signum, stop_event.set)
            except NotImplementedError:
                signal.signal(signum, lambda *_: stop_event.set())

    logger.info("Starting CARQ worker", extra={"worker_id": worker_id})
    await coordinator.start()
    try:
        await stop_event.wait()
    finally:
        await coordinator.stop()
        await db_manager.close()


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
