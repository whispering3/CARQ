"""
Unit tests for models.
"""

import pytest
from uuid import uuid4

from carq.models.models import (
    Document,
    DocumentStatus,
    Chunk,
    ChunkStatus,
    ProcessingTask,
    TaskStatus,
    TaskType,
)


class TestDocumentModel:
    """Test Document model."""

    def test_document_creation(self):
        """Test document can be created."""
        doc = Document(
            source_uri="s3://bucket/document.pdf",
            content_hash=b"hash123",
            status=DocumentStatus.PENDING,
            attributes={"category": "technical"},
        )

        assert doc.source_uri == "s3://bucket/document.pdf"
        assert doc.status == DocumentStatus.PENDING
        assert doc.attributes["category"] == "technical"

    def test_document_defaults(self):
        """Test document has proper defaults."""
        doc = Document(
            source_uri="s3://bucket/doc.pdf",
            content_hash=b"hash",
        )

        assert doc.status == DocumentStatus.PENDING
        assert doc.document_type == "pdf"
        assert doc.attributes == {}

    def test_document_to_dict(self):
        """Test document.to_dict()."""
        doc = Document(
            source_uri="s3://bucket/doc.pdf",
            content_hash=b"hash",
        )

        doc_dict = doc.to_dict()
        assert doc_dict["source_uri"] == "s3://bucket/doc.pdf"
        assert "id" in doc_dict
        assert "created_at" in doc_dict


class TestChunkModel:
    """Test Chunk model."""

    def test_chunk_creation(self):
        """Test chunk can be created."""
        doc_id = uuid4()
        chunk = Chunk(
            document_id=doc_id,
            chunk_index=0,
            content="This is chunk content",
            content_hash=b"chunkhash",
            status=ChunkStatus.PENDING,
        )

        assert chunk.chunk_index == 0
        assert chunk.content == "This is chunk content"
        assert chunk.status == ChunkStatus.PENDING

    def test_chunk_defaults(self):
        """Test chunk has proper defaults."""
        chunk = Chunk(
            document_id=uuid4(),
            chunk_index=0,
            content="content",
            content_hash=b"hash",
        )

        assert chunk.status == ChunkStatus.PENDING
        assert chunk.attributes == {}
        assert chunk.tokens is None


class TestProcessingTaskModel:
    """Test ProcessingTask model."""

    def test_task_creation(self):
        """Test task can be created."""
        doc_id = uuid4()
        task = ProcessingTask(
            document_id=doc_id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.PENDING,
            priority=5,
        )

        assert task.task_type == TaskType.PARSE_PDF
        assert task.status == TaskStatus.PENDING
        assert task.priority == 5

    def test_task_is_retryable(self):
        """Test is_retryable property."""
        doc_id = uuid4()
        task = ProcessingTask(
            document_id=doc_id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.FAILED,
            attempt_count=1,
            max_attempts=3,
        )

        assert task.is_retryable is True

        # Max attempts reached
        task.attempt_count = 3
        assert task.is_retryable is False

    def test_task_can_be_claimed(self):
        """Test can_be_claimed property."""
        doc_id = uuid4()
        task = ProcessingTask(
            document_id=doc_id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.PENDING,
        )

        assert task.can_be_claimed is True

        # Change status
        task.status = TaskStatus.PROCESSING
        assert task.can_be_claimed is False


@pytest.mark.asyncio
class TestModelsAsync:
    """Async tests for models."""

    async def test_document_relationship_lazy_loading(self):
        """Test relationships work correctly."""
        # This would require a database session
        # Covered by integration tests
        pass
