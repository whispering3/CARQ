"""Modelos ORM SQLAlchemy para o CARQ: Document, Chunk, Embedding, ProcessingTask, TaskDeadletter."""

import enum
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    ForeignKey,
    Index,
    String,
    Text,
    Integer,
    LargeBinary,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from pgvector.sqlalchemy import Vector
from sqlalchemy.orm import Mapped, mapped_column, relationship

from carq.models.base import Base, DialectJSON


class DocumentStatus(str, enum.Enum):
    """Status de processamento do documento."""

    PENDING = "pending"  # Na fila para processamento
    CHUNKING = "chunking"  # Sendo dividido em chunks
    EMBEDDING = "embedding"  # Chunks sendo incorporados
    DONE = "done"  # Processado com sucesso
    FAILED = "failed"  # Processamento falhou
    ARCHIVED = "archived"  # Marcado para exclusão


class ChunkStatus(str, enum.Enum):
    """Status de processamento do chunk."""

    PENDING = "pending"  # Aguardando incorporação
    EMBEDDING = "embedding"  # Sendo incorporado
    DONE = "done"  # Vetor inserido
    FAILED = "failed"  # Incorporação falhou


class TaskType(str, enum.Enum):
    """Tipo de tarefa de processamento."""

    PARSE_PDF = "parse_pdf"
    CHUNK_DOCUMENT = "chunk_document"
    EMBED_CHUNK = "embed_chunk"
    INSERT_VECTOR = "insert_vector"


class TaskStatus(str, enum.Enum):
    """Status da tarefa."""

    PENDING = "pending"  # Aguardando worker
    PROCESSING = "processing"  # Em processamento
    DONE = "done"  # Concluído com sucesso
    FAILED = "failed"  # Processamento falhou
    RETRYING = "retrying"  # Agendado para retry




class Document(Base):
    """Entidade raiz do documento (PDF, URL, texto puro)."""

    __tablename__ = "rag_documents"

    source_uri: Mapped[str] = mapped_column(String(2048), nullable=False, index=True)
    content_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, index=True)
    status: Mapped[DocumentStatus] = mapped_column(
        String(32),
        nullable=False,
        default=DocumentStatus.PENDING,
        index=True,
    )
    document_type: Mapped[str] = mapped_column(String(50), default="pdf")
    attributes: Mapped[dict] = mapped_column(DialectJSON(), default={}, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)


    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk",
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
    )
    tasks: Mapped[list["ProcessingTask"]] = relationship(
        "ProcessingTask",
        back_populates="document",
        cascade="all, delete-orphan",
    )
    deadletter: Mapped[list["TaskDeadletter"]] = relationship(
        "TaskDeadletter",
        back_populates="document",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("idx_doc_status_created", status, "created_at"),
        Index("idx_doc_uri_hash", source_uri, content_hash, unique=True),
    )

    def __init__(self, **kwargs):
        kwargs.setdefault('status', DocumentStatus.PENDING)
        kwargs.setdefault('document_type', 'pdf')
        kwargs.setdefault('attributes', {})
        super().__init__(**kwargs)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks) if self.chunks else 0

    @property
    def embedding_count(self) -> int:
        if not self.chunks:
            return 0
        return sum(1 for chunk in self.chunks if chunk.embedding is not None)

    @property
    def progress_percentage(self) -> float:
        """Progresso como percentual (0-100)."""
        if not self.chunks:
            return 0.0
        return (self.embedding_count / self.chunk_count) * 100




class Chunk(Base):
    """Chunk de documento (segmento de documento)."""

    __tablename__ = "rag_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("rag_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, index=True)
    status: Mapped[ChunkStatus] = mapped_column(
        String(32),
        nullable=False,
        default=ChunkStatus.PENDING,
        index=True,
    )
    tokens: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    attributes: Mapped[dict] = mapped_column(DialectJSON(), default={}, nullable=False)


    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
        lazy="joined",
    )
    embedding: Mapped[Optional["Embedding"]] = relationship(
        "Embedding",
        back_populates="chunk",
        uselist=False,
        cascade="all, delete-orphan",
        lazy="joined",
    )

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_index"),
        Index("idx_chunk_status", status),
        Index("idx_chunk_hash", content_hash),
    )

    def __init__(self, **kwargs):
        kwargs.setdefault('status', ChunkStatus.PENDING)
        kwargs.setdefault('attributes', {})
        super().__init__(**kwargs)




class Embedding(Base):
    """Embedding vetorial para o chunk."""

    __tablename__ = "rag_embeddings"

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("rag_chunks.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(3072),
        nullable=False,
    )
    model: Mapped[str] = mapped_column(
        String(100),
        default="text-embedding-3-large",
    )
    tokens_used: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(default=0.0)
    attributes: Mapped[dict] = mapped_column(DialectJSON(), default={}, nullable=False)


    chunk: Mapped["Chunk"] = relationship(
        "Chunk",
        back_populates="embedding",
        lazy="joined",
    )

    __table_args__ = ({"comment": "Vector index created via migration (see main.py lifespan)"},)




class ProcessingTask(Base):
    """Tarefa de processamento assíncrono."""

    __tablename__ = "processing_tasks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("rag_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    task_type: Mapped[TaskType] = mapped_column(String(50), nullable=False)
    status: Mapped[TaskStatus] = mapped_column(
        String(32),
        nullable=False,
        default=TaskStatus.PENDING,
        index=True,
    )
    priority: Mapped[int] = mapped_column(Integer, default=0, index=True)
    attempt_count: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    attributes: Mapped[dict] = mapped_column(DialectJSON(), default={}, nullable=False)
    completed_at: Mapped[Optional[datetime]] = mapped_column(nullable=True)
    worker_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)


    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="tasks",
        lazy="joined",
    )

    __table_args__ = (
        Index("idx_task_status_priority", status, priority),
        Index("idx_task_created", "created_at"),
        Index("idx_task_worker", worker_id),
    )

    def __init__(self, **kwargs):
        kwargs.setdefault('status', TaskStatus.PENDING)
        kwargs.setdefault('priority', 0)
        kwargs.setdefault('attempt_count', 0)
        kwargs.setdefault('max_attempts', 3)
        kwargs.setdefault('attributes', {})
        super().__init__(**kwargs)

    @property
    def is_retryable(self) -> bool:
        return (
            self.status == TaskStatus.FAILED
            and self.attempt_count < self.max_attempts
        )

    @property
    def can_be_claimed(self) -> bool:
        return self.status == TaskStatus.PENDING and self.attempt_count < self.max_attempts




class TaskDeadletter(Base):
    """Tarefa que excedeu o limite de tentativas."""

    __tablename__ = "task_deadletter"

    task_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("rag_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    error_details: Mapped[dict] = mapped_column(DialectJSON(), default={}, nullable=False)


    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="deadletter",
        lazy="joined",
    )

    __table_args__ = (
        Index("idx_dlq_task_id", task_id),
        Index("idx_dlq_created", "created_at"),
    )
