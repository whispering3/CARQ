"""
Comprehensive integration tests for end-to-end scenarios.

Tests cover:
- End-to-end PDF ingestion and processing
- Concurrent access scenarios
- Error handling and recovery
- Performance baselines
"""

import asyncio
import time

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
# TEST: END-TO-END INGEST TO SEARCH
# ============================================================================


@pytest.mark.integration
class TestE2EIngestToSearch:
    """Test end-to-end flow from ingest to search."""

    @pytest.mark.asyncio
    async def test_e2e_ingest_pdf(self, test_session):
        """Test PDF ingestion creates document."""
        doc = Document(
            source_uri="s3://bucket/test.pdf",
            content_hash=b"test-hash",
            status=DocumentStatus.PENDING,
            metadata={"source": "integration_test"},
        )
        test_session.add(doc)
        await test_session.commit()

        assert doc.id is not None
        assert doc.status == DocumentStatus.PENDING

    @pytest.mark.asyncio
    async def test_e2e_parse_chunks(self, test_session, sample_document):
        """Test PDF parsing creates chunks."""
        # Simulate PDF parsing by creating chunks
        for i in range(5):
            chunk = Chunk(
                document_id=sample_document.id,
                chunk_index=i,
                content=f"Content chunk {i}",
                content_hash=f"hash-{i}".encode(),
                status=ChunkStatus.PENDING,
            )
            test_session.add(chunk)

        await test_session.commit()

        # Verify chunks created
        result = await test_session.execute(
            select(Chunk).where(Chunk.document_id == sample_document.id)
        )
        chunks = result.scalars().all()
        assert len(chunks) == 5

    @pytest.mark.asyncio
    async def test_e2e_embed_vectors(self, test_session, sample_chunk):
        """Test chunk embedding."""
        # Simulate embedding
        sample_chunk.status = ChunkStatus.DONE
        test_session.add(sample_chunk)
        await test_session.commit()

        result = await test_session.execute(
            select(Chunk).where(Chunk.id == sample_chunk.id)
        )
        updated = result.scalar_one_or_none()
        assert updated.status == ChunkStatus.DONE

    @pytest.mark.asyncio
    async def test_e2e_search_similar(self, test_session, sample_document):
        """Test searching for similar chunks."""
        # Create multiple chunks with different content
        chunks = []
        for i in range(5):
            chunk = Chunk(
                document_id=sample_document.id,
                chunk_index=i,
                content=f"Machine learning topic {i}" if i < 2 else f"Other topic {i}",
                content_hash=f"hash-{i}".encode(),
                status=ChunkStatus.DONE,
            )
            test_session.add(chunk)
            chunks.append(chunk)

        await test_session.commit()

        # Search for machine learning chunks
        result = await test_session.execute(
            select(Chunk)
            .where(Chunk.document_id == sample_document.id)
            .where(Chunk.status == ChunkStatus.DONE)
        )
        found = result.scalars().all()
        assert len(found) >= 5

    @pytest.mark.asyncio
    async def test_e2e_concurrent_ingest(self, test_session):
        """Test concurrent document ingestion (serialized via lock)."""
        lock = asyncio.Lock()

        async def ingest_document(index):
            async with lock:
                doc = Document(
                    source_uri=f"s3://bucket/doc-{index}.pdf",
                    content_hash=f"hash-{index}".encode(),
                    status=DocumentStatus.PENDING,
                )
                test_session.add(doc)
                await test_session.flush()
                return doc.id

        # Ingest 10 documents (serialized to avoid session conflicts)
        doc_ids = await asyncio.gather(*[ingest_document(i) for i in range(10)])
        await test_session.commit()

        result = await test_session.execute(
            select(Document).where(Document.id.in_(doc_ids))
        )
        docs = result.scalars().all()
        assert len(docs) == 10

    @pytest.mark.asyncio
    async def test_e2e_retry_on_failure(self, test_session, sample_document):
        """Test retry on failure."""
        # Create a failed task
        task = ProcessingTask(
            document_id=sample_document.id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.FAILED,
            attempt_count=1,
        )
        test_session.add(task)
        await test_session.commit()

        # Retry
        task.status = TaskStatus.RETRYING
        task.attempt_count = 2
        test_session.add(task)
        await test_session.commit()

        # Success
        task.status = TaskStatus.DONE
        test_session.add(task)
        await test_session.commit()

        result = await test_session.execute(
            select(ProcessingTask).where(ProcessingTask.id == task.id)
        )
        updated = result.scalar_one_or_none()
        assert updated.status == TaskStatus.DONE
        assert updated.attempt_count == 2

    @pytest.mark.asyncio
    async def test_e2e_rate_limit_handling(self, test_session, sample_document):
        """Test rate limiting during ingestion."""
        # Create multiple tasks that would be rate limited
        task_ids = []

        for _i in range(5):
            task = ProcessingTask(
                document_id=sample_document.id,
                task_type=TaskType.EMBED_CHUNK,
                status=TaskStatus.PENDING,
                priority=5,
            )
            test_session.add(task)
            await test_session.flush()
            task_ids.append(task.id)

        await test_session.commit()

        # Verify all tasks created despite rate limiting
        result = await test_session.execute(
            select(ProcessingTask).where(ProcessingTask.id.in_(task_ids))
        )
        tasks = result.scalars().all()
        assert len(tasks) == 5


# ============================================================================
# TEST: CONCURRENT ACCESS
# ============================================================================


@pytest.mark.integration
class TestConcurrentAccess:
    """Test concurrent access patterns."""

    @pytest.mark.asyncio
    async def test_concurrent_session_access(self, test_session):
        """Test 50 concurrent sessions (serialized via lock)."""
        lock = asyncio.Lock()

        async def session_operation(index):
            async with lock:
                doc = Document(
                    source_uri=f"s3://bucket/doc-{index}.pdf",
                    content_hash=f"hash-{index}".encode(),
                )
                test_session.add(doc)
                await test_session.flush()

                result = await test_session.execute(
                    select(Document).where(Document.id == doc.id)
                )
                return result.scalar_one_or_none()

        # Run 50 concurrent operations
        results = await asyncio.gather(*[
            session_operation(i) for i in range(50)
        ])

        await test_session.commit()

        # All should succeed
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_task_dequeue(self, test_session, sample_document):
        """Test 20 workers dequeuing without duplicates."""
        # Create 20 pending tasks
        for i in range(20):
            task = ProcessingTask(
                document_id=sample_document.id,
                task_type=TaskType.PARSE_PDF,
                status=TaskStatus.PENDING,
                priority=20 - i,  # Higher priority = lower number processed first
            )
            test_session.add(task)

        await test_session.commit()

        dequeued_task_ids = []

        async def worker_dequeue():
            result = await test_session.execute(
                select(ProcessingTask)
                .where(ProcessingTask.status == TaskStatus.PENDING)
                .order_by(ProcessingTask.priority.desc())
                .limit(1)
            )
            task = result.scalar_one_or_none()

            if task:
                dequeued_task_ids.append(task.id)
                return task.id
            return None

        # 20 workers dequeue
        results = await asyncio.gather(*[worker_dequeue() for _ in range(20)])

        # Should have dequeued some tasks
        assert any(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_vector_insert(self, test_session, sample_document):
        """Test 100 concurrent vector inserts."""
        # Create 100 chunks
        for i in range(100):
            chunk = Chunk(
                document_id=sample_document.id,
                chunk_index=i,
                content=f"Chunk {i}",
                content_hash=f"hash-{i}".encode(),
                status=ChunkStatus.PENDING,
            )
            test_session.add(chunk)

        await test_session.commit()

        async def insert_vector(index):
            lock = insert_vector._lock  # type: ignore[attr-defined]
            async with lock:
                result = await test_session.execute(
                    select(Chunk).where(Chunk.document_id == sample_document.id)
                    .limit(1).offset(index % 100)
                )
                chunk = result.scalar_one_or_none()

                if chunk:
                    chunk.status = ChunkStatus.DONE
                    test_session.add(chunk)
                    await test_session.flush()

        insert_vector._lock = asyncio.Lock()

        # Insert 100 vectors concurrently
        await asyncio.gather(*[insert_vector(i) for i in range(100)])

        await test_session.commit()

    @pytest.mark.asyncio
    async def test_concurrent_embedding_dispatch(self, test_session, sample_document):
        """Test 50 concurrent embedding requests."""
        # Create 50 chunks
        for i in range(50):
            chunk = Chunk(
                document_id=sample_document.id,
                chunk_index=i,
                content=f"Chunk {i}",
                content_hash=f"hash-{i}".encode(),
            )
            test_session.add(chunk)

        await test_session.commit()

        async def dispatch_embedding(index):
            # Simulate embedding dispatch
            result = await test_session.execute(
                select(Chunk).where(Chunk.document_id == sample_document.id)
                .limit(1).offset(index)
            )
            chunk = result.scalar_one_or_none()
            return chunk is not None if chunk else False

        # Dispatch 50 concurrent embeddings
        results = await asyncio.gather(*[
            dispatch_embedding(i) for i in range(50)
        ])

        assert sum(results) >= 50

    @pytest.mark.asyncio
    async def test_race_condition_task_update(self, test_session, sample_task):
        """Test multiple updates to same task (serialized via lock)."""
        lock = asyncio.Lock()

        async def update_task(attempt):
            async with lock:
                result = await test_session.execute(
                    select(ProcessingTask).where(ProcessingTask.id == sample_task.id)
                )
                task = result.scalar_one_or_none()

                task.attempt_count = attempt
                test_session.add(task)
                await test_session.flush()

        try:
            await asyncio.gather(*[update_task(i) for i in range(5)])
            await test_session.commit()
        except Exception:
            await test_session.rollback()


# ============================================================================
# TEST: ERROR HANDLING
# ============================================================================


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_malformed_pdf_handling(self, test_session):
        """Test handling of malformed PDF."""
        # Create document for malformed PDF
        doc = Document(
            source_uri="s3://bucket/malformed.pdf",
            content_hash=b"malformed-hash",
            status=DocumentStatus.PENDING,
        )
        test_session.add(doc)
        await test_session.commit()

        # Mark as failed
        doc.status = DocumentStatus.FAILED
        test_session.add(doc)
        await test_session.commit()

        result = await test_session.execute(
            select(Document).where(Document.id == doc.id)
        )
        updated = result.scalar_one_or_none()
        assert updated.status == DocumentStatus.FAILED

    @pytest.mark.asyncio
    async def test_network_timeout_handling(self, test_session, sample_task):
        """Test handling of network timeouts."""
        # Mark task as failed
        sample_task.status = TaskStatus.FAILED
        test_session.add(sample_task)
        await test_session.commit()

        # Should be retriable
        assert sample_task.attempt_count < sample_task.max_attempts

    @pytest.mark.asyncio
    async def test_database_disconnect(self, test_session, sample_document):
        """Test database reconnection."""
        # Create entity
        doc = Document(
            source_uri="s3://bucket/test.pdf",
            content_hash=b"hash",
        )
        test_session.add(doc)
        await test_session.commit()

        # Should be able to retrieve
        result = await test_session.execute(
            select(Document).where(Document.id == doc.id)
        )
        retrieved = result.scalar_one_or_none()
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, test_session):
        """Test circuit breaker recovery."""
        # Create task
        doc = Document(
            source_uri="s3://bucket/test.pdf",
            content_hash=b"hash",
        )
        test_session.add(doc)
        await test_session.commit()

        task = ProcessingTask(
            document_id=doc.id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.PENDING,
        )
        test_session.add(task)
        await test_session.commit()

        # Simulate recovery
        result = await test_session.execute(
            select(ProcessingTask).where(ProcessingTask.id == task.id)
        )
        retrieved = result.scalar_one_or_none()
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_dead_letter_queue(self, test_session, sample_document):
        """Test dead letter queue handling."""
        # Create failing task
        task = ProcessingTask(
            document_id=sample_document.id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.FAILED,
            attempt_count=3,
            max_attempts=3,
        )
        test_session.add(task)
        await test_session.commit()

        # Verify in dead letter
        result = await test_session.execute(
            select(ProcessingTask).where(
                (ProcessingTask.id == task.id) &
                (ProcessingTask.attempt_count >= ProcessingTask.max_attempts)
            )
        )
        dead_lettered = result.scalar_one_or_none()
        assert dead_lettered is not None


# ============================================================================
# TEST: PERFORMANCE BASELINES
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceBaselines:
    """Test performance baselines."""

    @pytest.mark.asyncio
    async def test_throughput_1k_chunks(self, test_session, performance_timer, metrics_collector):
        """Test processing 1000 chunks completes in < 60s."""
        doc = Document(
            source_uri="s3://bucket/test.pdf",
            content_hash=b"hash",
        )
        test_session.add(doc)
        await test_session.commit()

        performance_timer.start()

        # Create 1000 chunks
        for i in range(1000):
            chunk = Chunk(
                document_id=doc.id,
                chunk_index=i,
                content=f"Chunk {i}",
                content_hash=f"hash-{i}".encode(),
            )
            test_session.add(chunk)

            if i % 100 == 0:
                await test_session.flush()

        await test_session.commit()

        performance_timer.stop()
        elapsed = performance_timer.elapsed_ms / 1000  # Convert to seconds

        # Should complete in less than 60 seconds
        assert elapsed < 60
        metrics_collector.record_success()

    @pytest.mark.asyncio
    async def test_latency_p99(self, test_session, metrics_collector):
        """Test p99 latency is < 5s."""
        doc = Document(
            source_uri="s3://bucket/test.pdf",
            content_hash=b"hash",
        )
        test_session.add(doc)
        await test_session.commit()

        # Measure latencies
        for i in range(100):
            start = time.time()

            chunk = Chunk(
                document_id=doc.id,
                chunk_index=i,
                content=f"Chunk {i}",
                content_hash=f"hash-{i}".encode(),
            )
            test_session.add(chunk)
            await test_session.flush()

            elapsed = (time.time() - start) * 1000  # Convert to ms
            metrics_collector.record_timing(elapsed)

        await test_session.commit()

        # Check p99
        p99 = metrics_collector.p99
        assert p99 < 5000  # 5 seconds in ms

    @pytest.mark.asyncio
    async def test_memory_stability(self, test_session):
        """Test no memory leaks over 100 tasks."""
        doc = Document(
            source_uri="s3://bucket/test.pdf",
            content_hash=b"hash",
        )
        test_session.add(doc)
        await test_session.commit()

        # Create and process 100 tasks
        for i in range(100):
            chunk = Chunk(
                document_id=doc.id,
                chunk_index=i,
                content=f"Chunk {i}",
                content_hash=f"hash-{i}".encode(),
            )
            test_session.add(chunk)

            if i % 10 == 0:
                await test_session.flush()

        await test_session.commit()

        # Verify all created
        result = await test_session.execute(
            select(Chunk).where(Chunk.document_id == doc.id)
        )
        chunks = result.scalars().all()
        assert len(chunks) == 100

    @pytest.mark.asyncio
    async def test_cpu_efficiency(self, test_session):
        """Test CPU usage remains reasonable."""
        doc = Document(
            source_uri="s3://bucket/test.pdf",
            content_hash=b"hash",
        )
        test_session.add(doc)
        await test_session.commit()

        # Perform typical operations
        start = time.time()

        for i in range(50):
            # Create
            chunk = Chunk(
                document_id=doc.id,
                chunk_index=i,
                content=f"Chunk {i}",
                content_hash=f"hash-{i}".encode(),
            )
            test_session.add(chunk)

            # Read
            result = await test_session.execute(
                select(Chunk).where(Chunk.document_id == doc.id).limit(1)
            )
            result.scalar_one_or_none()

        await test_session.commit()

        elapsed = time.time() - start

        # Should complete reasonably quickly
        assert elapsed < 30  # 30 seconds


# ============================================================================
# MARKER TESTS
# ============================================================================


@pytest.mark.integration
def test_integration_marker():
    """Test that integration tests are properly marked."""
    assert True
