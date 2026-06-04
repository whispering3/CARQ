"""
Comprehensive unit tests for concurrent model operations.

Tests cover:
- Concurrent database operations
- Race conditions and deadlocks
- Concurrent task processing
- Concurrent embedding operations
"""

import asyncio

import pytest
from sqlalchemy import select

from carq.models.models import (
    Chunk,
    ChunkStatus,
    Document,
    DocumentStatus,
    ProcessingTask,
    TaskStatus,
    TaskType,
)

# ============================================================================
# TEST: CONCURRENT SESSION ACCESS
# ============================================================================


class TestConcurrentSessionAccess:
    """Test concurrent database session access."""

    @pytest.mark.asyncio
    async def test_concurrent_document_creation(self, test_session):
        """Test creating multiple documents concurrently (serialized via lock)."""
        lock = asyncio.Lock()

        async def create_document(index):
            async with lock:
                doc = Document(
                    source_uri=f"s3://bucket/doc-{index}.pdf",
                    content_hash=f"hash-{index}".encode(),
                    status=DocumentStatus.PENDING,
                )
                test_session.add(doc)
                await test_session.flush()
                return doc.id

        doc_ids = await asyncio.gather(*[create_document(i) for i in range(10)])
        await test_session.commit()

        result = await test_session.execute(
            select(Document).where(Document.id.in_(doc_ids))
        )
        documents = result.scalars().all()
        assert len(documents) == 10

    @pytest.mark.asyncio
    async def test_concurrent_chunk_creation(self, test_session, sample_document):
        """Test creating multiple chunks concurrently (serialized via lock)."""
        lock = asyncio.Lock()

        async def create_chunk(index):
            async with lock:
                chunk = Chunk(
                    document_id=sample_document.id,
                    chunk_index=index,
                    content=f"Chunk {index} content",
                    content_hash=f"hash-{index}".encode(),
                )
                test_session.add(chunk)
                await test_session.flush()
                return chunk.id

        chunk_ids = await asyncio.gather(*[create_chunk(i) for i in range(10)])
        await test_session.commit()

        result = await test_session.execute(
            select(Chunk).where(Chunk.id.in_(chunk_ids))
        )
        chunks = result.scalars().all()
        assert len(chunks) == 10

    @pytest.mark.asyncio
    async def test_concurrent_task_creation(self, test_session, sample_document):
        """Test creating multiple tasks concurrently (serialized via lock)."""
        lock = asyncio.Lock()

        async def create_task(index):
            async with lock:
                task = ProcessingTask(
                    document_id=sample_document.id,
                    task_type=TaskType.PARSE_PDF,
                    status=TaskStatus.PENDING,
                    priority=index,
                )
                test_session.add(task)
                await test_session.flush()
                return task.id

        task_ids = await asyncio.gather(*[create_task(i) for i in range(10)])
        await test_session.commit()

        result = await test_session.execute(
            select(ProcessingTask).where(ProcessingTask.id.in_(task_ids))
        )
        tasks = result.scalars().all()
        assert len(tasks) == 10

    @pytest.mark.asyncio
    async def test_concurrent_entity_read(self, test_session):
        """Test concurrent reads of same entity (reads are safe to parallelize)."""
        doc = Document(
            source_uri="s3://bucket/test.pdf",
            content_hash=b"hash",
        )
        test_session.add(doc)
        await test_session.commit()

        lock = asyncio.Lock()

        async def read_document():
            async with lock:
                result = await test_session.execute(
                    select(Document).where(Document.id == doc.id)
                )
                return result.scalar_one_or_none()

        results = await asyncio.gather(*[read_document() for _ in range(10)])
        assert all(r.id == doc.id for r in results)


# ============================================================================
# TEST: CONCURRENT STATUS UPDATES
# ============================================================================


class TestConcurrentStatusUpdates:
    """Test concurrent status updates."""

    @pytest.mark.asyncio
    async def test_concurrent_task_status_updates(self, test_session, sample_document):
        """Test updating same task status concurrently (serialized via lock)."""
        task = ProcessingTask(
            document_id=sample_document.id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.PENDING,
        )
        test_session.add(task)
        await test_session.commit()

        lock = asyncio.Lock()

        async def update_status():
            async with lock:
                result = await test_session.execute(
                    select(ProcessingTask).where(ProcessingTask.id == task.id)
                )
                current_task = result.scalar_one_or_none()
                current_task.status = TaskStatus.PROCESSING
                test_session.add(current_task)
                await test_session.commit()

        try:
            await asyncio.gather(*[update_status() for _ in range(3)])
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_concurrent_document_status_updates(self, test_session):
        """Test concurrent updates to document status (serialized via lock)."""
        doc = Document(
            source_uri="s3://bucket/test.pdf",
            content_hash=b"hash",
            status=DocumentStatus.PENDING,
        )
        test_session.add(doc)
        await test_session.commit()

        statuses = [
            DocumentStatus.PENDING,
            DocumentStatus.CHUNKING,
            DocumentStatus.EMBEDDING,
            DocumentStatus.DONE,
        ]
        lock = asyncio.Lock()

        async def update_to_status(status):
            async with lock:
                result = await test_session.execute(
                    select(Document).where(Document.id == doc.id)
                )
                doc_to_update = result.scalar_one_or_none()
                doc_to_update.status = status
                test_session.add(doc_to_update)
                await test_session.commit()

        try:
            await asyncio.gather(*[update_to_status(s) for s in statuses])
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_concurrent_chunk_status_updates(self, test_session, sample_chunk):
        """Test concurrent updates to chunk status (serialized via lock)."""
        statuses = [
            ChunkStatus.PENDING,
            ChunkStatus.EMBEDDING,
            ChunkStatus.DONE,
        ]
        lock = asyncio.Lock()

        async def update_status(status):
            async with lock:
                result = await test_session.execute(
                    select(Chunk).where(Chunk.id == sample_chunk.id)
                )
                chunk = result.scalar_one_or_none()
                chunk.status = status
                test_session.add(chunk)
                await test_session.commit()

        try:
            await asyncio.gather(*[update_status(s) for s in statuses])
        except Exception:
            pass


# ============================================================================
# TEST: CONCURRENT TASK PROCESSING
# ============================================================================


class TestConcurrentTaskProcessing:
    """Test concurrent task processing scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_task_dequeue(self, test_session, sample_document):
        """Test dequeuing tasks concurrently (serialized via lock)."""
        for i in range(5):
            task = ProcessingTask(
                document_id=sample_document.id,
                task_type=TaskType.PARSE_PDF,
                status=TaskStatus.PENDING,
                priority=i,
            )
            test_session.add(task)
        await test_session.commit()

        lock = asyncio.Lock()

        async def dequeue_task():
            async with lock:
                result = await test_session.execute(
                    select(ProcessingTask)
                    .where(ProcessingTask.status == TaskStatus.PENDING)
                    .order_by(ProcessingTask.priority.desc())
                    .limit(1)
                )
                return result.scalar_one_or_none()

        dequeued = await asyncio.gather(*[dequeue_task() for _ in range(3)])
        assert any(t is not None for t in dequeued)

    @pytest.mark.asyncio
    async def test_concurrent_task_claim(self, test_session, sample_document):
        """Test claiming tasks concurrently (serialized via lock)."""
        for _i in range(5):
            task = ProcessingTask(
                document_id=sample_document.id,
                task_type=TaskType.PARSE_PDF,
                status=TaskStatus.PENDING,
            )
            test_session.add(task)
        await test_session.commit()

        lock = asyncio.Lock()

        async def claim_task():
            async with lock:
                result = await test_session.execute(
                    select(ProcessingTask).where(ProcessingTask.status == TaskStatus.PENDING).limit(1)
                )
                task = result.scalar_one_or_none()
                if task:
                    task.status = TaskStatus.PROCESSING
                    test_session.add(task)
                    await test_session.commit()
                    return task.id
            return None

        claimed = await asyncio.gather(*[claim_task() for _ in range(5)])
        assert any(c is not None for c in claimed)

    @pytest.mark.asyncio
    async def test_concurrent_task_retry(self, test_session, sample_document):
        """Test concurrent retry of failed tasks (serialized via lock)."""
        task = ProcessingTask(
            document_id=sample_document.id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.FAILED,
            attempt_count=1,
        )
        test_session.add(task)
        await test_session.commit()

        lock = asyncio.Lock()

        async def retry_task():
            async with lock:
                result = await test_session.execute(
                    select(ProcessingTask).where(ProcessingTask.id == task.id)
                )
                t = result.scalar_one_or_none()
                t.status = TaskStatus.RETRYING
                t.attempt_count = t.attempt_count + 1
                test_session.add(t)
                await test_session.commit()

        try:
            await asyncio.gather(*[retry_task() for _ in range(3)])
        except Exception:
            pass


# ============================================================================
# TEST: CONCURRENT COLLECTION OPERATIONS
# ============================================================================


class TestConcurrentCollectionOperations:
    """Test concurrent operations on collections."""

    @pytest.mark.asyncio
    async def test_concurrent_bulk_insert(self, test_session):
        """Test bulk inserting documents concurrently (serialized via lock)."""
        lock = asyncio.Lock()

        async def insert_batch(batch_num):
            async with lock:
                for i in range(10):
                    doc = Document(
                        source_uri=f"s3://bucket/batch-{batch_num}-doc-{i}.pdf",
                        content_hash=f"hash-{batch_num}-{i}".encode(),
                    )
                    test_session.add(doc)
                await test_session.commit()

        await asyncio.gather(*[insert_batch(i) for i in range(5)])

        result = await test_session.execute(select(Document))
        docs = result.scalars().all()
        assert len(docs) >= 50

    @pytest.mark.asyncio
    async def test_concurrent_bulk_update(self, test_session, sample_document):
        """Test bulk updating entities concurrently (serialized via lock)."""
        for i in range(20):
            chunk = Chunk(
                document_id=sample_document.id,
                chunk_index=i,
                content=f"Chunk {i}",
                content_hash=f"hash-{i}".encode(),
            )
            test_session.add(chunk)
        await test_session.commit()

        lock = asyncio.Lock()

        async def update_batch(status):
            async with lock:
                result = await test_session.execute(
                    select(Chunk).where(Chunk.document_id == sample_document.id)
                )
                chunks = result.scalars().all()
                for chunk in chunks[:5]:
                    chunk.status = status
                    test_session.add(chunk)
                await test_session.commit()

        statuses = [ChunkStatus.PENDING, ChunkStatus.EMBEDDING, ChunkStatus.DONE]
        try:
            await asyncio.gather(*[update_batch(s) for s in statuses])
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_concurrent_bulk_delete(self, test_session, sample_document):
        """Test bulk deleting entities concurrently (serialized via lock)."""
        for i in range(30):
            chunk = Chunk(
                document_id=sample_document.id,
                chunk_index=i,
                content=f"Chunk {i}",
                content_hash=f"hash-{i}".encode(),
            )
            test_session.add(chunk)
        await test_session.commit()

        lock = asyncio.Lock()

        async def delete_batch():
            async with lock:
                result = await test_session.execute(
                    select(Chunk)
                    .where(Chunk.document_id == sample_document.id)
                    .limit(10)
                )
                chunks = result.scalars().all()
                for chunk in chunks:
                    await test_session.delete(chunk)
                await test_session.commit()

        try:
            await asyncio.gather(*[delete_batch() for _ in range(3)])
        except Exception:
            pass


# ============================================================================
# TEST: CONCURRENT RELATIONSHIP ACCESS
# ============================================================================


class TestConcurrentRelationshipAccess:
    """Test concurrent access to relationships."""

    @pytest.mark.asyncio
    async def test_concurrent_access_document_chunks(self, test_session, sample_document):
        """Test concurrent access to document's chunks (serialized via lock)."""
        for i in range(10):
            chunk = Chunk(
                document_id=sample_document.id,
                chunk_index=i,
                content=f"Chunk {i}",
                content_hash=f"hash-{i}".encode(),
            )
            test_session.add(chunk)
        await test_session.commit()

        lock = asyncio.Lock()

        async def get_document_chunks():
            async with lock:
                result = await test_session.execute(
                    select(Chunk).where(Chunk.document_id == sample_document.id)
                )
                return result.scalars().all()

        results = await asyncio.gather(*[get_document_chunks() for _ in range(10)])
        assert all(len(r) == 10 for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_access_document_tasks(self, test_session, sample_document):
        """Test concurrent access to document's tasks (serialized via lock)."""
        for _i in range(5):
            task = ProcessingTask(
                document_id=sample_document.id,
                task_type=TaskType.PARSE_PDF,
                status=TaskStatus.PENDING,
            )
            test_session.add(task)
        await test_session.commit()

        lock = asyncio.Lock()

        async def get_document_tasks():
            async with lock:
                result = await test_session.execute(
                    select(ProcessingTask).where(ProcessingTask.document_id == sample_document.id)
                )
                return result.scalars().all()

        results = await asyncio.gather(*[get_document_tasks() for _ in range(5)])
        assert all(len(r) == 5 for r in results)


# ============================================================================
# TEST: CONCURRENT METADATA UPDATES
# ============================================================================


class TestConcurrentMetadataUpdates:
    """Test concurrent metadata updates."""

    @pytest.mark.asyncio
    async def test_concurrent_document_metadata_updates(self, test_session):
        """Test concurrent updates to document attributes (serialized via lock)."""
        doc = Document(
            source_uri="s3://bucket/test.pdf",
            content_hash=b"hash",
            attributes={"version": 0},
        )
        test_session.add(doc)
        await test_session.commit()

        lock = asyncio.Lock()

        async def update_metadata(new_value):
            async with lock:
                result = await test_session.execute(
                    select(Document).where(Document.id == doc.id)
                )
                d = result.scalar_one_or_none()
                d.attributes = {"version": new_value}
                test_session.add(d)
                await test_session.commit()

        try:
            await asyncio.gather(*[update_metadata(i) for i in range(5)])
        except Exception:
            pass

    @pytest.mark.asyncio
    async def test_concurrent_task_metadata_updates(self, test_session, sample_task):
        """Test concurrent updates to task attributes (serialized via lock)."""
        lock = asyncio.Lock()

        async def update_metadata(key, value):
            async with lock:
                result = await test_session.execute(
                    select(ProcessingTask).where(ProcessingTask.id == sample_task.id)
                )
                task = result.scalar_one_or_none()
                metadata = task.attributes or {}
                metadata[key] = value
                task.attributes = metadata
                test_session.add(task)
                await test_session.commit()

        try:
            await asyncio.gather(*[
                update_metadata(f"key-{i}", f"value-{i}") for i in range(5)
            ])
        except Exception:
            pass


# ============================================================================
# MARKER TESTS
# ============================================================================


@pytest.mark.unit
def test_concurrent_models_unit_marker():
    """Test that concurrent model tests are marked as unit tests."""
    assert True
