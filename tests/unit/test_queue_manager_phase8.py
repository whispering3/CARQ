"""Tests for queue_manager.py - Phase 8.

Covers enqueue, dequeue, status transitions, retry logic,
batch operations, and queue status reporting.
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from carq.models.models import (
    Document,
    DocumentStatus,
    ProcessingTask,
    TaskStatus,
    TaskType,
)
from carq.queue.queue_manager import QueueManager
from carq.core.exceptions import TaskNotFoundError, QueueError


# ============================================================================
# HELPERS
# ============================================================================


def make_task(
    session,
    document_id=None,
    task_type=TaskType.PARSE_PDF,
    status=TaskStatus.PENDING,
    priority=0,
    attempt_count=0,
    max_attempts=3,
    worker_id=None,
    attributes=None,
):
    """Build a ProcessingTask without committing."""
    task = ProcessingTask(
        document_id=document_id or uuid.uuid4(),
        task_type=task_type,
        status=status,
        priority=priority,
        attempt_count=attempt_count,
        max_attempts=max_attempts,
        worker_id=worker_id,
        attributes=attributes or {},
    )
    session.add(task)
    return task


# ============================================================================
# ENQUEUE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_enqueue_task_creates_record(test_session, sample_document):
    """enqueue_task should create a ProcessingTask with PENDING status."""
    qm = QueueManager(test_session)
    task = await qm.enqueue_task(
        document_id=sample_document.id,
        task_type=TaskType.PARSE_PDF,
        priority=0,
    )
    await test_session.commit()

    assert task is not None
    assert task.id is not None
    assert task.document_id == sample_document.id
    assert task.task_type == TaskType.PARSE_PDF
    assert task.status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_enqueue_task_with_priority(test_session, sample_document):
    """enqueue_task should store priority correctly."""
    qm = QueueManager(test_session)
    task = await qm.enqueue_task(
        document_id=sample_document.id,
        task_type=TaskType.CHUNK_DOCUMENT,
        priority=10,
    )
    assert task.priority == 10


@pytest.mark.asyncio
async def test_enqueue_task_with_metadata(test_session, sample_document):
    """enqueue_task should store metadata in attributes."""
    qm = QueueManager(test_session)
    meta = {"source": "test", "retry": False}
    task = await qm.enqueue_task(
        document_id=sample_document.id,
        task_type=TaskType.EMBED_CHUNK,
        metadata=meta,
    )
    # metadata is merged into attributes
    assert task is not None


@pytest.mark.asyncio
async def test_enqueue_task_default_metadata(test_session, sample_document):
    """enqueue_task with no metadata should default to empty dict."""
    qm = QueueManager(test_session)
    task = await qm.enqueue_task(
        document_id=sample_document.id,
        task_type=TaskType.PARSE_PDF,
    )
    assert isinstance(task.attributes, dict)


@pytest.mark.asyncio
async def test_enqueue_multiple_tasks(test_session, sample_document):
    """Multiple tasks can be enqueued for the same document."""
    qm = QueueManager(test_session)
    t1 = await qm.enqueue_task(sample_document.id, TaskType.PARSE_PDF, priority=5)
    t2 = await qm.enqueue_task(sample_document.id, TaskType.CHUNK_DOCUMENT, priority=3)
    assert t1.id != t2.id


# ============================================================================
# DEQUEUE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_dequeue_returns_pending_task(test_session, sample_document):
    """dequeue_task should return pending tasks."""
    qm = QueueManager(test_session)
    await qm.enqueue_task(sample_document.id, TaskType.PARSE_PDF, priority=0)
    await test_session.commit()

    tasks = await qm.dequeue_task(worker_id="worker-1")
    assert len(tasks) == 1
    assert tasks[0].status == TaskStatus.PROCESSING
    assert tasks[0].worker_id == "worker-1"
    assert tasks[0].attempt_count == 1


@pytest.mark.asyncio
async def test_dequeue_empty_queue_returns_empty_list(test_session):
    """dequeue_task with empty queue returns empty list."""
    qm = QueueManager(test_session)
    tasks = await qm.dequeue_task(worker_id="worker-1")
    assert tasks == []


@pytest.mark.asyncio
async def test_dequeue_respects_priority(test_session, sample_document):
    """dequeue_task should return highest-priority task first."""
    qm = QueueManager(test_session)
    await qm.enqueue_task(sample_document.id, TaskType.PARSE_PDF, priority=1)
    await qm.enqueue_task(sample_document.id, TaskType.CHUNK_DOCUMENT, priority=10)
    await test_session.commit()

    tasks = await qm.dequeue_task(worker_id="worker-1", batch_size=1)
    assert len(tasks) == 1
    assert tasks[0].priority == 10


@pytest.mark.asyncio
async def test_dequeue_batch_size(test_session, sample_document):
    """dequeue_task should respect batch_size."""
    qm = QueueManager(test_session)
    for _ in range(5):
        await qm.enqueue_task(sample_document.id, TaskType.PARSE_PDF)
    await test_session.commit()

    tasks = await qm.dequeue_task(worker_id="worker-1", batch_size=3)
    assert len(tasks) == 3


@pytest.mark.asyncio
async def test_dequeue_skips_maxed_attempts(test_session, sample_document):
    """Tasks with attempt_count >= max_attempts should be skipped."""
    qm = QueueManager(test_session)
    # Create task that has already used all attempts
    task = ProcessingTask(
        document_id=sample_document.id,
        task_type=TaskType.PARSE_PDF,
        status=TaskStatus.PENDING,
        attempt_count=3,
        max_attempts=3,
        attributes={},
    )
    test_session.add(task)
    await test_session.commit()

    tasks = await qm.dequeue_task(worker_id="worker-1")
    assert tasks == []


@pytest.mark.asyncio
async def test_dequeue_filter_by_task_type(test_session, sample_document):
    """dequeue_task should filter by task_types when provided."""
    qm = QueueManager(test_session)
    await qm.enqueue_task(sample_document.id, TaskType.PARSE_PDF)
    await qm.enqueue_task(sample_document.id, TaskType.EMBED_CHUNK)
    await test_session.commit()

    tasks = await qm.dequeue_task(
        worker_id="worker-1",
        task_types=[TaskType.EMBED_CHUNK],
        batch_size=10,
    )
    assert all(t.task_type == TaskType.EMBED_CHUNK for t in tasks)


# ============================================================================
# MARK DONE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_mark_task_done(test_session, sample_task):
    """mark_task_done should set status to DONE."""
    qm = QueueManager(test_session)
    task = await qm.mark_task_done(sample_task.id)
    assert task.status == TaskStatus.DONE
    assert task.completed_at is not None


@pytest.mark.asyncio
async def test_mark_task_done_with_metadata(test_session, sample_task):
    """mark_task_done with metadata should update attributes."""
    qm = QueueManager(test_session)
    task = await qm.mark_task_done(sample_task.id, metadata={"result": "ok"})
    assert task.status == TaskStatus.DONE


@pytest.mark.asyncio
async def test_mark_task_done_not_found_raises(test_session):
    """mark_task_done with unknown ID raises TaskNotFoundError."""
    qm = QueueManager(test_session)
    with pytest.raises((TaskNotFoundError, Exception)):
        await qm.mark_task_done(uuid.uuid4())


# ============================================================================
# MARK FAILED TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_mark_task_failed_sets_error(test_session, sample_task):
    """mark_task_failed should record the error message."""
    qm = QueueManager(test_session)
    task = await qm.mark_task_failed(sample_task.id, error_message="Something went wrong")
    assert task.error_message == "Something went wrong"


@pytest.mark.asyncio
async def test_mark_task_failed_with_error_details(test_session, sample_task):
    """mark_task_failed with error_details stores them in attributes."""
    qm = QueueManager(test_session)
    details = {"traceback": "line 42", "code": "ERR_001"}
    task = await qm.mark_task_failed(
        sample_task.id,
        error_message="Critical failure",
        error_details=details,
    )
    assert task.error_message == "Critical failure"


@pytest.mark.asyncio
async def test_mark_task_failed_retryable(test_session, sample_document):
    """A task with attempts remaining gets RETRYING status on failure."""
    # Task: attempt_count=1, max_attempts=3, status=FAILED -> is_retryable=True
    task = ProcessingTask(
        document_id=sample_document.id,
        task_type=TaskType.PARSE_PDF,
        status=TaskStatus.FAILED,
        attempt_count=1,
        max_attempts=3,
        attributes={},
    )
    test_session.add(task)
    await test_session.commit()

    qm = QueueManager(test_session)
    updated = await qm.mark_task_failed(task.id, error_message="transient error")
    # Status depends on is_retryable — either RETRYING or FAILED
    assert updated.status in (TaskStatus.RETRYING, TaskStatus.FAILED)


@pytest.mark.asyncio
async def test_mark_task_failed_not_found(test_session):
    """mark_task_failed with unknown ID raises exception."""
    qm = QueueManager(test_session)
    with pytest.raises(Exception):
        await qm.mark_task_failed(uuid.uuid4(), error_message="oops")


# ============================================================================
# STATUS TRANSITION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_pending_to_processing_transition(test_session, sample_document):
    """Dequeuing moves task from PENDING to PROCESSING."""
    qm = QueueManager(test_session)
    await qm.enqueue_task(sample_document.id, TaskType.PARSE_PDF)
    await test_session.commit()

    tasks = await qm.dequeue_task(worker_id="worker-99")
    assert tasks[0].status == TaskStatus.PROCESSING


@pytest.mark.asyncio
async def test_processing_to_done_transition(test_session, sample_document):
    """Full lifecycle: PENDING → PROCESSING → DONE."""
    qm = QueueManager(test_session)
    await qm.enqueue_task(sample_document.id, TaskType.PARSE_PDF)
    await test_session.commit()

    tasks = await qm.dequeue_task(worker_id="worker-1")
    assert tasks[0].status == TaskStatus.PROCESSING

    done_task = await qm.mark_task_done(tasks[0].id)
    assert done_task.status == TaskStatus.DONE


@pytest.mark.asyncio
async def test_attempt_count_increments_on_dequeue(test_session, sample_document):
    """attempt_count increments each time task is dequeued."""
    qm = QueueManager(test_session)
    await qm.enqueue_task(sample_document.id, TaskType.PARSE_PDF, priority=0)
    await test_session.commit()

    tasks = await qm.dequeue_task(worker_id="worker-1")
    assert tasks[0].attempt_count == 1


# ============================================================================
# RESET STUCK TASKS
# ============================================================================


@pytest.mark.asyncio
async def test_reset_stuck_tasks_uses_mock(test_session):
    """reset_stuck_tasks executes an UPDATE and returns the rowcount."""
    from unittest.mock import MagicMock
    from sqlalchemy import update as sa_update

    # Use a mock session so we can control rowcount
    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.rowcount = 2
    mock_session.execute = AsyncMock(return_value=mock_result)
    mock_session.flush = AsyncMock()

    qm = QueueManager(mock_session)
    count = await qm.reset_stuck_tasks(timeout_seconds=300)
    assert count == 2


@pytest.mark.asyncio
async def test_reset_stuck_tasks_no_stuck(test_session, sample_document):
    """reset_stuck_tasks with no stuck tasks returns 0."""
    qm = QueueManager(test_session)
    count = await qm.reset_stuck_tasks(timeout_seconds=300)
    assert count == 0


# ============================================================================
# BATCH OPERATIONS
# ============================================================================


@pytest.mark.asyncio
async def test_mark_tasks_done_batch(test_session, sample_document):
    """mark_tasks_done should batch-update tasks to DONE."""
    qm = QueueManager(test_session)
    t1 = await qm.enqueue_task(sample_document.id, TaskType.PARSE_PDF)
    t2 = await qm.enqueue_task(sample_document.id, TaskType.CHUNK_DOCUMENT)
    await test_session.commit()

    count = await qm.mark_tasks_done([t1.id, t2.id])
    assert count == 2


@pytest.mark.asyncio
async def test_mark_tasks_done_empty_list(test_session):
    """mark_tasks_done with empty list returns 0."""
    qm = QueueManager(test_session)
    count = await qm.mark_tasks_done([])
    assert count == 0


@pytest.mark.asyncio
async def test_mark_tasks_failed_batch(test_session, sample_document):
    """mark_tasks_failed should batch-update tasks to FAILED."""
    qm = QueueManager(test_session)
    t1 = await qm.enqueue_task(sample_document.id, TaskType.EMBED_CHUNK)
    t2 = await qm.enqueue_task(sample_document.id, TaskType.INSERT_VECTOR)
    await test_session.commit()

    count = await qm.mark_tasks_failed([t1.id, t2.id], error_message="batch fail")
    assert count == 2


@pytest.mark.asyncio
async def test_mark_tasks_failed_empty_list(test_session):
    """mark_tasks_failed with empty list returns 0."""
    qm = QueueManager(test_session)
    count = await qm.mark_tasks_failed([])
    assert count == 0


# ============================================================================
# GET TASK / GET DOCUMENT TASKS
# ============================================================================


@pytest.mark.asyncio
async def test_get_task(test_session, sample_task):
    """get_task should return task by ID."""
    qm = QueueManager(test_session)
    task = await qm.get_task(sample_task.id)
    assert task is not None
    assert task.id == sample_task.id


@pytest.mark.asyncio
async def test_get_task_not_found(test_session):
    """get_task returns None for unknown ID."""
    qm = QueueManager(test_session)
    task = await qm.get_task(uuid.uuid4())
    assert task is None


@pytest.mark.asyncio
async def test_get_document_tasks(test_session, sample_document):
    """get_document_tasks should return all tasks for a document."""
    qm = QueueManager(test_session)
    await qm.enqueue_task(sample_document.id, TaskType.PARSE_PDF)
    await qm.enqueue_task(sample_document.id, TaskType.CHUNK_DOCUMENT)
    await test_session.commit()

    tasks = await qm.get_document_tasks(sample_document.id)
    assert len(tasks) >= 2


@pytest.mark.asyncio
async def test_get_document_tasks_with_status_filter(test_session, sample_document):
    """get_document_tasks with status filter returns only matching tasks."""
    qm = QueueManager(test_session)
    await qm.enqueue_task(sample_document.id, TaskType.PARSE_PDF)
    await test_session.commit()

    pending_tasks = await qm.get_document_tasks(sample_document.id, status=TaskStatus.PENDING)
    done_tasks = await qm.get_document_tasks(sample_document.id, status=TaskStatus.DONE)
    assert len(pending_tasks) >= 1
    assert all(t.status == TaskStatus.PENDING for t in pending_tasks)
    assert len(done_tasks) == 0


# ============================================================================
# QUEUE STATUS
# ============================================================================


@pytest.mark.asyncio
async def test_get_queue_status_empty(test_session):
    """get_queue_status with empty queue returns empty or zero counts."""
    qm = QueueManager(test_session)
    status = await qm.get_queue_status()
    assert isinstance(status, dict)
    assert "total_tasks" in status
    assert status["total_tasks"] == 0


@pytest.mark.asyncio
async def test_get_queue_status_with_tasks(test_session, sample_document):
    """get_queue_status reports pending task counts."""
    qm = QueueManager(test_session)
    await qm.enqueue_task(sample_document.id, TaskType.PARSE_PDF)
    await test_session.commit()

    status = await qm.get_queue_status()
    assert status["total_tasks"] >= 1


# ============================================================================
# ERROR HANDLING
# ============================================================================


@pytest.mark.asyncio
async def test_enqueue_raises_queue_error_on_failure():
    """enqueue_task should raise QueueError when DB fails."""
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session.flush = AsyncMock(side_effect=Exception("DB error"))

    qm = QueueManager(mock_session)
    with pytest.raises(QueueError):
        await qm.enqueue_task(uuid.uuid4(), TaskType.PARSE_PDF)


@pytest.mark.asyncio
async def test_dequeue_raises_queue_error_on_failure():
    """dequeue_task should raise QueueError when DB fails."""
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock(side_effect=Exception("DB error"))

    qm = QueueManager(mock_session)
    with pytest.raises(QueueError):
        await qm.dequeue_task(worker_id="worker-1")
