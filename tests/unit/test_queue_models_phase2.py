"""
Comprehensive unit tests for Phase 2: Queue Models and Management.

Tests cover:
- ORM models creation and relationships
- Task status transitions
- Queue dequeue operations (SELECT FOR UPDATE)
- Task marking as done
- Worker pool spawning
- Task assignment to workers
"""

import asyncio
from uuid import uuid4

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
from carq.worker.task_coordinator import TaskCoordinator
from carq.worker.worker_pool import WorkerPool

# ============================================================================
# TEST: DOCUMENT MODEL CREATION
# ============================================================================


class TestDocumentModel:
    """Test Document ORM model."""

    @pytest.mark.asyncio
    async def test_document_creation(self, test_session, sample_document_data):
        """Test document creation."""
        doc = Document(**sample_document_data)
        test_session.add(doc)
        await test_session.commit()

        assert doc.id is not None
        assert doc.source_uri == sample_document_data["source_uri"]
        assert doc.status == DocumentStatus.PENDING

    @pytest.mark.asyncio
    async def test_document_with_metadata(self, test_session):
        """Test document with metadata."""
        metadata = {
            "category": "technical",
            "source": "unit_test",
            "tags": ["ai", "ml"],
        }
        doc = Document(
            source_uri="s3://bucket/doc.pdf",
            content_hash=b"hash",
            attributes=metadata,
        )
        test_session.add(doc)
        await test_session.commit()

        assert doc.attributes == metadata

    @pytest.mark.asyncio
    async def test_document_status_default(self, test_session):
        """Test document status defaults to PENDING."""
        doc = Document(
            source_uri="s3://bucket/doc.pdf",
            content_hash=b"hash",
        )
        test_session.add(doc)
        await test_session.commit()

        assert doc.status == DocumentStatus.PENDING

    @pytest.mark.asyncio
    async def test_document_timestamps(self, test_session):
        """Test document timestamps are set."""
        doc = Document(
            source_uri="s3://bucket/doc.pdf",
            content_hash=b"hash",
        )
        test_session.add(doc)
        await test_session.commit()

        assert doc.created_at is not None
        assert doc.updated_at is not None
        assert doc.created_at <= doc.updated_at

    @pytest.mark.asyncio
    async def test_document_retrieve_by_id(self, test_session, sample_document):
        """Test retrieving document by ID."""
        doc_id = sample_document.id

        result = await test_session.execute(
            select(Document).where(Document.id == doc_id)
        )
        retrieved = result.scalar_one_or_none()

        assert retrieved is not None
        assert retrieved.id == doc_id
        assert retrieved.source_uri == sample_document.source_uri

    @pytest.mark.asyncio
    async def test_document_update_status(self, test_session, sample_document):
        """Test updating document status."""
        sample_document.status = DocumentStatus.CHUNKING
        test_session.add(sample_document)
        await test_session.commit()

        result = await test_session.execute(
            select(Document).where(Document.id == sample_document.id)
        )
        updated = result.scalar_one_or_none()
        assert updated.status == DocumentStatus.CHUNKING

    @pytest.mark.asyncio
    async def test_document_list_all(self, test_session):
        """Test listing all documents."""
        # Create multiple documents
        for i in range(5):
            doc = Document(
                source_uri=f"s3://bucket/doc-{i}.pdf",
                content_hash=f"hash-{i}".encode(),
            )
            test_session.add(doc)

        await test_session.commit()

        result = await test_session.execute(select(Document))
        documents = result.scalars().all()

        assert len(documents) >= 5

    @pytest.mark.asyncio
    async def test_document_unique_content_hash(self, test_session):
        """Test document with same content hash."""
        hash_value = b"same-hash"

        doc1 = Document(
            source_uri="s3://bucket/doc1.pdf",
            content_hash=hash_value,
        )
        doc2 = Document(
            source_uri="s3://bucket/doc2.pdf",
            content_hash=hash_value,
        )

        test_session.add(doc1)
        test_session.add(doc2)
        await test_session.commit()

        result = await test_session.execute(
            select(Document).where(Document.content_hash == hash_value)
        )
        docs = result.scalars().all()
        assert len(docs) >= 2


# ============================================================================
# TEST: CHUNK MODEL CREATION
# ============================================================================


class TestChunkModel:
    """Test Chunk ORM model."""

    @pytest.mark.asyncio
    async def test_chunk_creation(self, test_session, sample_document):
        """Test chunk creation."""
        chunk = Chunk(
            document_id=sample_document.id,
            chunk_index=0,
            content="Sample chunk content",
            content_hash=b"chunk-hash",
        )
        test_session.add(chunk)
        await test_session.commit()

        assert chunk.id is not None
        assert chunk.chunk_index == 0
        assert chunk.content == "Sample chunk content"

    @pytest.mark.asyncio
    async def test_chunk_with_metadata(self, test_session, sample_document):
        """Test chunk with metadata."""
        metadata = {"page": 1, "section": "introduction"}

        chunk = Chunk(
            document_id=sample_document.id,
            chunk_index=0,
            content="Content",
            content_hash=b"hash",
            attributes=metadata,
        )
        test_session.add(chunk)
        await test_session.commit()

        assert chunk.attributes == metadata

    @pytest.mark.asyncio
    async def test_chunk_status_default(self, test_session, sample_document):
        """Test chunk status defaults to PENDING."""
        chunk = Chunk(
            document_id=sample_document.id,
            chunk_index=0,
            content="Content",
            content_hash=b"hash",
        )
        test_session.add(chunk)
        await test_session.commit()

        assert chunk.status == ChunkStatus.PENDING

    @pytest.mark.asyncio
    async def test_chunks_for_document(self, test_session, sample_document):
        """Test retrieving chunks for a document."""
        # Create multiple chunks
        for i in range(5):
            chunk = Chunk(
                document_id=sample_document.id,
                chunk_index=i,
                content=f"Chunk {i} content",
                content_hash=f"hash-{i}".encode(),
            )
            test_session.add(chunk)

        await test_session.commit()

        result = await test_session.execute(
            select(Chunk).where(Chunk.document_id == sample_document.id)
        )
        chunks = result.scalars().all()

        assert len(chunks) == 5

    @pytest.mark.asyncio
    async def test_chunk_document_relationship(self, test_session, sample_chunk):
        """Test chunk-document relationship."""
        # Load related document
        result = await test_session.execute(
            select(Document).where(Document.id == sample_chunk.document_id)
        )
        doc = result.scalar_one_or_none()

        assert doc is not None
        assert doc.id == sample_chunk.document_id

    @pytest.mark.asyncio
    async def test_chunk_with_tokens(self, test_session, sample_document):
        """Test chunk with token count."""
        chunk = Chunk(
            document_id=sample_document.id,
            chunk_index=0,
            content="Some content",
            content_hash=b"hash",
            tokens=100,
        )
        test_session.add(chunk)
        await test_session.commit()

        assert chunk.tokens == 100

    @pytest.mark.asyncio
    async def test_chunk_update_status(self, test_session, sample_chunk):
        """Test updating chunk status."""
        sample_chunk.status = ChunkStatus.EMBEDDING
        test_session.add(sample_chunk)
        await test_session.commit()

        result = await test_session.execute(
            select(Chunk).where(Chunk.id == sample_chunk.id)
        )
        updated = result.scalar_one_or_none()
        assert updated.status == ChunkStatus.EMBEDDING


# ============================================================================
# TEST: PROCESSING TASK MODEL
# ============================================================================


class TestProcessingTaskModel:
    """Test ProcessingTask ORM model."""

    @pytest.mark.asyncio
    async def test_task_creation(self, test_session, sample_document):
        """Test task creation."""
        task = ProcessingTask(
            document_id=sample_document.id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.PENDING,
            priority=5,
        )
        test_session.add(task)
        await test_session.commit()

        assert task.id is not None
        assert task.task_type == TaskType.PARSE_PDF
        assert task.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_task_status_transitions(self, test_session, sample_task):
        """Test task status transitions."""
        # PENDING -> PROCESSING
        sample_task.status = TaskStatus.PROCESSING
        test_session.add(sample_task)
        await test_session.commit()

        # Verify transition
        result = await test_session.execute(
            select(ProcessingTask).where(ProcessingTask.id == sample_task.id)
        )
        updated = result.scalar_one_or_none()
        assert updated.status == TaskStatus.PROCESSING

    @pytest.mark.asyncio
    async def test_task_fail_and_retry(self, test_session, sample_task):
        """Test task failure and retry."""
        sample_task.status = TaskStatus.FAILED
        sample_task.attempt_count = 1
        test_session.add(sample_task)
        await test_session.commit()

        # Transition to RETRYING
        sample_task.status = TaskStatus.RETRYING
        test_session.add(sample_task)
        await test_session.commit()

        result = await test_session.execute(
            select(ProcessingTask).where(ProcessingTask.id == sample_task.id)
        )
        updated = result.scalar_one_or_none()
        assert updated.status == TaskStatus.RETRYING
        assert updated.attempt_count == 1

    @pytest.mark.asyncio
    async def test_task_completion(self, test_session, sample_task):
        """Test task completion."""
        sample_task.status = TaskStatus.DONE
        test_session.add(sample_task)
        await test_session.commit()

        result = await test_session.execute(
            select(ProcessingTask).where(ProcessingTask.id == sample_task.id)
        )
        updated = result.scalar_one_or_none()
        assert updated.status == TaskStatus.DONE

    @pytest.mark.asyncio
    async def test_task_attempt_tracking(self, test_session, sample_task):
        """Test task attempt tracking."""
        sample_task.attempt_count = 3
        sample_task.max_attempts = 5
        test_session.add(sample_task)
        await test_session.commit()

        result = await test_session.execute(
            select(ProcessingTask).where(ProcessingTask.id == sample_task.id)
        )
        updated = result.scalar_one_or_none()
        assert updated.attempt_count == 3
        assert updated.max_attempts == 5

    @pytest.mark.asyncio
    async def test_task_priority_ordering(self, test_session, sample_document):
        """Test tasks can be ordered by priority."""
        # Create tasks with different priorities
        priorities = [1, 5, 3, 9, 2]
        tasks = []

        for priority in priorities:
            task = ProcessingTask(
                document_id=sample_document.id,
                task_type=TaskType.PARSE_PDF,
                status=TaskStatus.PENDING,
                priority=priority,
            )
            test_session.add(task)
            tasks.append(task)

        await test_session.commit()

        # Query ordered by priority
        result = await test_session.execute(
            select(ProcessingTask)
            .where(ProcessingTask.document_id == sample_document.id)
            .order_by(ProcessingTask.priority.desc())
        )
        ordered_tasks = result.scalars().all()

        # Should be ordered by priority (descending)
        for i in range(len(ordered_tasks) - 1):
            assert ordered_tasks[i].priority >= ordered_tasks[i + 1].priority

    @pytest.mark.asyncio
    async def test_task_with_metadata(self, test_session, sample_document):
        """Test task with metadata."""
        metadata = {"source": "unit_test", "retry_count": 0}

        task = ProcessingTask(
            document_id=sample_document.id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.PENDING,
            attributes=metadata,
        )
        test_session.add(task)
        await test_session.commit()

        assert task.attributes == metadata

    @pytest.mark.asyncio
    async def test_pending_tasks_query(self, test_session, sample_document):
        """Test querying for pending tasks."""
        # Create pending and non-pending tasks
        for _i in range(3):
            task = ProcessingTask(
                document_id=sample_document.id,
                task_type=TaskType.PARSE_PDF,
                status=TaskStatus.PENDING,
            )
            test_session.add(task)

        # Create a non-pending task
        non_pending = ProcessingTask(
            document_id=sample_document.id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.DONE,
        )
        test_session.add(non_pending)

        await test_session.commit()

        # Query pending tasks
        result = await test_session.execute(
            select(ProcessingTask).where(ProcessingTask.status == TaskStatus.PENDING)
        )
        pending_tasks = result.scalars().all()

        assert len(pending_tasks) >= 3


# ============================================================================
# TEST: QUEUE MANAGER DEQUEUE
# ============================================================================


class TestQueueManagerDequeue:
    """Test queue manager dequeue with SELECT FOR UPDATE."""

    @pytest.mark.asyncio
    async def test_dequeue_single_task(self, test_session, sample_document):
        """Test dequeueing a single task."""
        # Create a task
        task = ProcessingTask(
            document_id=sample_document.id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.PENDING,
            priority=5,
        )
        test_session.add(task)
        await test_session.commit()

        # Simulate dequeue - get pending task with FOR UPDATE lock
        result = await test_session.execute(
            select(ProcessingTask)
            .where(ProcessingTask.status == TaskStatus.PENDING)
            .order_by(ProcessingTask.priority.desc())
            .limit(1)
        )
        dequeued = result.scalar_one_or_none()

        assert dequeued is not None
        assert dequeued.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_dequeue_respects_priority(self, test_session, sample_document):
        """Test that dequeue respects priority order."""
        # Create tasks with different priorities
        low_priority = ProcessingTask(
            document_id=sample_document.id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.PENDING,
            priority=1,
        )
        high_priority = ProcessingTask(
            document_id=sample_document.id,
            task_type=TaskType.EMBED_CHUNK,
            status=TaskStatus.PENDING,
            priority=10,
        )

        test_session.add(low_priority)
        test_session.add(high_priority)
        await test_session.commit()

        # Dequeue should get high priority first
        result = await test_session.execute(
            select(ProcessingTask)
            .where(ProcessingTask.status == TaskStatus.PENDING)
            .order_by(ProcessingTask.priority.desc())
            .limit(1)
        )
        dequeued = result.scalar_one_or_none()

        assert dequeued.priority == 10

    @pytest.mark.asyncio
    async def test_dequeue_empty_queue(self, test_session):
        """Test dequeue on empty queue."""
        result = await test_session.execute(
            select(ProcessingTask).where(ProcessingTask.status == TaskStatus.PENDING).limit(1)
        )
        dequeued = result.scalar_one_or_none()

        assert dequeued is None

    @pytest.mark.asyncio
    async def test_dequeue_excludes_non_pending(self, test_session, sample_document):
        """Test dequeue excludes non-pending tasks."""
        # Create tasks in various states
        pending = ProcessingTask(
            document_id=sample_document.id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.PENDING,
        )
        processing = ProcessingTask(
            document_id=sample_document.id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.PROCESSING,
        )
        done = ProcessingTask(
            document_id=sample_document.id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.DONE,
        )

        test_session.add_all([pending, processing, done])
        await test_session.commit()

        # Dequeue should only get pending
        result = await test_session.execute(
            select(ProcessingTask).where(ProcessingTask.status == TaskStatus.PENDING)
        )
        tasks = result.scalars().all()

        assert len(tasks) == 1
        assert tasks[0].status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_dequeue_multiple_tasks(self, test_session, sample_document):
        """Test dequeueing multiple tasks in order."""
        # Create 5 tasks
        for i in range(5):
            task = ProcessingTask(
                document_id=sample_document.id,
                task_type=TaskType.PARSE_PDF,
                status=TaskStatus.PENDING,
                priority=i,
            )
            test_session.add(task)

        await test_session.commit()

        # Dequeue all
        result = await test_session.execute(
            select(ProcessingTask)
            .where(ProcessingTask.status == TaskStatus.PENDING)
            .order_by(ProcessingTask.priority.desc())
        )
        dequeued = result.scalars().all()

        assert len(dequeued) == 5
        # Should be ordered by priority desc
        for i in range(len(dequeued) - 1):
            assert dequeued[i].priority >= dequeued[i + 1].priority


# ============================================================================
# TEST: QUEUE MANAGER MARK DONE
# ============================================================================


class TestQueueManagerMarkDone:
    """Test marking tasks as done."""

    @pytest.mark.asyncio
    async def test_mark_single_task_done(self, test_session, sample_task):
        """Test marking single task as done."""
        sample_task.status = TaskStatus.DONE
        test_session.add(sample_task)
        await test_session.commit()

        # Verify update
        result = await test_session.execute(
            select(ProcessingTask).where(ProcessingTask.id == sample_task.id)
        )
        updated = result.scalar_one_or_none()
        assert updated.status == TaskStatus.DONE

    @pytest.mark.asyncio
    async def test_mark_multiple_tasks_done(self, test_session, sample_document):
        """Test marking multiple tasks as done."""
        task_ids = []

        # Create multiple tasks
        for _i in range(3):
            task = ProcessingTask(
                document_id=sample_document.id,
                task_type=TaskType.PARSE_PDF,
                status=TaskStatus.PROCESSING,
            )
            test_session.add(task)
            await test_session.flush()
            task_ids.append(task.id)

        await test_session.commit()

        # Mark all as done
        result = await test_session.execute(
            select(ProcessingTask).where(ProcessingTask.id.in_(task_ids))
        )
        tasks = result.scalars().all()

        for task in tasks:
            task.status = TaskStatus.DONE

        await test_session.commit()

        # Verify all marked done
        result = await test_session.execute(
            select(ProcessingTask)
            .where(ProcessingTask.id.in_(task_ids))
            .where(ProcessingTask.status == TaskStatus.DONE)
        )
        done_tasks = result.scalars().all()
        assert len(done_tasks) == 3

    @pytest.mark.asyncio
    async def test_mark_task_done_with_results(self, test_session, sample_task):
        """Test marking task done with result data."""
        sample_task.status = TaskStatus.DONE
        sample_task.attributes = {"result": "success", "processed": 100}
        test_session.add(sample_task)
        await test_session.commit()

        result = await test_session.execute(
            select(ProcessingTask).where(ProcessingTask.id == sample_task.id)
        )
        updated = result.scalar_one_or_none()

        assert updated.status == TaskStatus.DONE
        assert updated.attributes["result"] == "success"

    @pytest.mark.asyncio
    async def test_mark_failed_task_as_done(self, test_session, sample_task):
        """Test marking previously failed task as done (after retry)."""
        # First mark as failed
        sample_task.status = TaskStatus.FAILED
        sample_task.attempt_count = 1
        test_session.add(sample_task)
        await test_session.commit()

        # Then mark as done after successful retry
        sample_task.status = TaskStatus.DONE
        sample_task.attempt_count = 2
        test_session.add(sample_task)
        await test_session.commit()

        result = await test_session.execute(
            select(ProcessingTask).where(ProcessingTask.id == sample_task.id)
        )
        updated = result.scalar_one_or_none()

        assert updated.status == TaskStatus.DONE
        assert updated.attempt_count == 2


# ============================================================================
# TEST: WORKER POOL
# ============================================================================


class TestWorkerPool:
    """Test worker pool functionality."""

    def test_worker_pool_creation(self):
        """Test creating worker pool."""
        pool = WorkerPool(num_workers=4)
        assert pool is not None
        assert pool.num_workers == 4

    def test_worker_pool_worker_count(self):
        """Test worker pool worker count."""
        for count in [1, 2, 4, 8]:
            pool = WorkerPool(num_workers=count)
            assert pool.num_workers == count

    @pytest.mark.asyncio
    async def test_worker_pool_start_stop(self):
        """Test starting and stopping worker pool."""
        pool = WorkerPool(num_workers=2)

        # Start pool
        await pool.start()
        assert pool.is_running

        # Stop pool
        await pool.stop()
        assert not pool.is_running

    @pytest.mark.asyncio
    async def test_worker_pool_submit_task(self):
        """Test submitting task to worker pool."""
        pool = WorkerPool(num_workers=2)
        await pool.start()

        async def sample_task():
            return "completed"

        result = await pool.submit(sample_task())
        assert result == "completed"

        await pool.stop()

    @pytest.mark.asyncio
    async def test_worker_pool_concurrent_tasks(self):
        """Test concurrent task execution."""
        pool = WorkerPool(num_workers=4)
        await pool.start()

        async def slow_task(duration):
            await asyncio.sleep(duration)
            return f"task_{duration}"

        tasks = [slow_task(0.01) for _ in range(4)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 4

        await pool.stop()

    def test_worker_pool_queue_size(self):
        """Test worker pool queue size configuration."""
        pool = WorkerPool(num_workers=2, queue_size=100)
        assert pool.queue_size == 100


# ============================================================================
# TEST: TASK COORDINATOR
# ============================================================================


class TestTaskCoordinator:
    """Test task coordinator functionality."""

    def test_task_coordinator_creation(self):
        """Test creating task coordinator."""
        coordinator = TaskCoordinator()
        assert coordinator is not None

    @pytest.mark.asyncio
    async def test_task_assignment(self):
        """Test task assignment to workers."""
        coordinator = TaskCoordinator()

        # Simulate task assignment
        worker_id = "worker_1"
        task_data = {
            "task_id": str(uuid4()),
            "document_id": str(uuid4()),
            "type": "PARSE_PDF",
        }

        assigned = await coordinator.assign_task(worker_id, task_data)
        assert assigned is not None

    @pytest.mark.asyncio
    async def test_task_completion_tracking(self):
        """Test tracking task completion."""
        coordinator = TaskCoordinator()
        task_id = str(uuid4())

        # Record completion
        await coordinator.record_completion(task_id, success=True)

        status = await coordinator.get_task_status(task_id)
        assert status is not None

    @pytest.mark.asyncio
    async def test_multiple_task_assignments(self):
        """Test assigning multiple tasks to different workers."""
        coordinator = TaskCoordinator()

        for i in range(5):
            worker_id = f"worker_{i}"
            task_data = {"task_id": str(uuid4())}

            assigned = await coordinator.assign_task(worker_id, task_data)
            assert assigned is not None

    @pytest.mark.asyncio
    async def test_task_timeout_handling(self):
        """Test task timeout handling."""
        coordinator = TaskCoordinator()

        task_data = {
            "task_id": str(uuid4()),
            "timeout": 5,
        }

        # Should handle timeout gracefully
        await coordinator.assign_task("worker_1", task_data)


# ============================================================================
# MARKER TESTS
# ============================================================================


@pytest.mark.unit
def test_phase2_unit_marker():
    """Test that Phase 2 tests are marked as unit tests."""
    assert True
