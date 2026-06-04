"""
Unit tests for monitoring/error_handler.py — DLQ management.
"""
import uuid
from datetime import datetime, timezone

import pytest

from carq.models.models import ProcessingTask, TaskStatus, TaskType
from carq.monitoring.error_handler import DLQEntry, ErrorHandler

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def session(test_session):
    return test_session


@pytest.fixture
def error_handler(test_session):
    return ErrorHandler(session=test_session, max_attempts=3)


@pytest.fixture
def failed_task(sample_task):
    """A task in FAILED state that has exceeded max_attempts."""
    sample_task.status = TaskStatus.FAILED
    sample_task.attempt_count = 3
    sample_task.attributes = {
        "error_details": {"error_type": "RuntimeError", "error_message": "oops"},
    }
    return sample_task


# ---------------------------------------------------------------------------
# Tests: DLQEntry
# ---------------------------------------------------------------------------

class TestDLQEntry:
    def test_to_dict(self):
        now = datetime.now(timezone.utc)
        entry = DLQEntry(
            task_id="t1",
            document_id="d1",
            task_type="PARSE_PDF",
            error_details={"err": "oops"},
            failed_at=now,
            attempt_count=3,
        )
        d = entry.to_dict()
        assert d["task_id"] == "t1"
        assert d["document_id"] == "d1"
        assert d["attempt_count"] == 3
        assert "failed_at" in d
        assert d["error_details"] == {"err": "oops"}


# ---------------------------------------------------------------------------
# Tests: scan_and_move_to_dlq
# ---------------------------------------------------------------------------

class TestScanAndMoveToDLQ:
    async def test_no_failed_tasks_returns_empty(self, error_handler):
        entries = await error_handler.scan_and_move_to_dlq()
        assert entries == []

    async def test_moves_failed_task_to_dlq(self, error_handler, session, failed_task):
        session.add(failed_task)
        await session.flush()

        entries = await error_handler.scan_and_move_to_dlq()
        assert len(entries) == 1
        assert entries[0].task_id == str(failed_task.id)
        assert entries[0].attempt_count == 3

        # Check task.attributes updated
        attrs = failed_task.attributes or {}
        assert attrs.get("dlq") is True
        assert "dlq_at" in attrs

    async def test_does_not_double_move(self, error_handler, session, failed_task):
        """Task already in DLQ should not be moved again."""
        failed_task.attributes = {"dlq": True, "dlq_at": "2026-01-01T00:00:00"}
        session.add(failed_task)
        await session.flush()

        entries = await error_handler.scan_and_move_to_dlq()
        assert entries == []

    async def test_pending_task_not_moved(self, error_handler, session, sample_task):
        """Pending task should never be moved to DLQ."""
        session.add(sample_task)
        await session.flush()

        entries = await error_handler.scan_and_move_to_dlq()
        assert entries == []

    async def test_failed_below_max_not_moved(self, error_handler, session, sample_task):
        """Failed task with attempt_count below threshold should NOT be moved."""
        sample_task.status = TaskStatus.FAILED
        sample_task.attempt_count = 1  # below max_attempts=3
        session.add(sample_task)
        await session.flush()

        entries = await error_handler.scan_and_move_to_dlq()
        assert entries == []


# ---------------------------------------------------------------------------
# Tests: get_dlq_entries
# ---------------------------------------------------------------------------

class TestGetDLQEntries:
    async def test_empty_dlq(self, error_handler):
        entries = await error_handler.get_dlq_entries()
        assert entries == []

    async def test_returns_dlq_entries_only(self, error_handler, session, failed_task):
        # Mark as DLQ
        failed_task.attributes = {"dlq": True, "dlq_at": "2026-01-01T00:00:00"}
        session.add(failed_task)
        await session.flush()

        entries = await error_handler.get_dlq_entries()
        assert len(entries) == 1

    async def test_does_not_return_non_dlq_failed(self, error_handler, session, sample_task):
        sample_task.status = TaskStatus.FAILED
        sample_task.attributes = {}  # no dlq flag
        session.add(sample_task)
        await session.flush()

        entries = await error_handler.get_dlq_entries()
        assert entries == []


# ---------------------------------------------------------------------------
# Tests: get_dlq_count
# ---------------------------------------------------------------------------

class TestGetDLQCount:
    async def test_empty(self, error_handler):
        count = await error_handler.get_dlq_count()
        assert count == 0

    async def test_count_matches(self, error_handler, session, failed_task):
        failed_task.attributes = {"dlq": True, "dlq_at": "2026-01-01T00:00:00"}
        session.add(failed_task)
        await session.flush()

        count = await error_handler.get_dlq_count()
        assert count == 1


# ---------------------------------------------------------------------------
# Tests: replay_task
# ---------------------------------------------------------------------------

class TestReplayTask:
    async def test_replay_resets_task(self, error_handler, session, failed_task):
        failed_task.attributes = {"dlq": True, "dlq_at": "2026-01-01T00:00:00"}
        session.add(failed_task)
        await session.flush()

        result = await error_handler.replay_task(str(failed_task.id))
        assert result is True
        assert failed_task.status == TaskStatus.PENDING
        assert failed_task.attempt_count == 0
        assert "dlq" not in (failed_task.attributes or {})

    async def test_replay_nonexistent_task(self, error_handler):
        result = await error_handler.replay_task(str(uuid.uuid4()))
        assert result is False

    async def test_replay_invalid_uuid(self, error_handler):
        result = await error_handler.replay_task("not-a-uuid")
        assert result is False


# ---------------------------------------------------------------------------
# Tests: replay_all
# ---------------------------------------------------------------------------

class TestReplayAll:
    async def test_replay_all_empty(self, error_handler):
        count = await error_handler.replay_all()
        assert count == 0

    async def test_replay_all_returns_count(self, error_handler, session, failed_task):
        failed_task.attributes = {"dlq": True, "dlq_at": "2026-01-01T00:00:00"}
        session.add(failed_task)
        await session.flush()

        count = await error_handler.replay_all()
        assert count == 1
        assert failed_task.status == TaskStatus.PENDING


# ---------------------------------------------------------------------------
# Tests: get_error_report
# ---------------------------------------------------------------------------

class TestGetErrorReport:
    async def test_report_structure(self, error_handler, session, sample_task):
        session.add(sample_task)
        await session.flush()

        report = await error_handler.get_error_report()
        assert "total_tasks" in report
        assert "by_status" in report
        assert "dlq_count" in report
        assert "dlq_entries" in report
        assert "generated_at" in report

    async def test_report_counts(self, error_handler, session, sample_document):
        import uuid

        from carq.models.models import TaskStatus
        # One pending task
        t1 = ProcessingTask(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.PENDING,
            attempt_count=0,
            priority=0,
        )
        # One DLQ failed task
        t2 = ProcessingTask(
            id=uuid.uuid4(),
            document_id=sample_document.id,
            task_type=TaskType.PARSE_PDF,
            status=TaskStatus.FAILED,
            attempt_count=3,
            priority=0,
            attributes={"dlq": True, "dlq_at": "2026-01-01T00:00:00"},
        )
        session.add(t1)
        session.add(t2)
        await session.flush()

        report = await error_handler.get_error_report()
        assert report["total_tasks"] >= 2
        assert report["dlq_count"] == 1


# ---------------------------------------------------------------------------
# Tests: handle_exception
# ---------------------------------------------------------------------------

class TestHandleException:
    async def test_increments_attempt_count(self, error_handler, session, sample_task):
        session.add(sample_task)
        await session.flush()
        original_count = sample_task.attempt_count

        await error_handler.handle_exception(sample_task, RuntimeError("test error"))
        assert sample_task.attempt_count == original_count + 1

    async def test_marks_task_as_failed(self, error_handler, session, sample_task):
        session.add(sample_task)
        await session.flush()

        await error_handler.handle_exception(sample_task, ValueError("bad value"))
        assert sample_task.status == TaskStatus.FAILED

    async def test_stores_error_details(self, error_handler, session, sample_task):
        session.add(sample_task)
        await session.flush()

        await error_handler.handle_exception(sample_task, RuntimeError("disk full"))
        attrs = sample_task.attributes or {}
        assert "error_details" in attrs
        assert attrs["error_details"]["error_type"] == "RuntimeError"
        assert "disk full" in attrs["error_details"]["error_message"]

    async def test_moves_to_dlq_after_max_attempts(self, error_handler, session, sample_task):
        sample_task.attempt_count = 2  # next attempt is 3 = max_attempts
        session.add(sample_task)
        await session.flush()

        moved = await error_handler.handle_exception(sample_task, RuntimeError("final failure"))
        assert moved is True
        assert (sample_task.attributes or {}).get("dlq") is True

    async def test_does_not_move_to_dlq_below_max(self, error_handler, session, sample_task):
        sample_task.attempt_count = 0
        session.add(sample_task)
        await session.flush()

        moved = await error_handler.handle_exception(sample_task, RuntimeError("first fail"))
        assert moved is False
        assert not (sample_task.attributes or {}).get("dlq")

    async def test_no_dlq_if_disabled(self, error_handler, session, sample_task):
        sample_task.attempt_count = 10  # well above max
        session.add(sample_task)
        await session.flush()

        moved = await error_handler.handle_exception(
            sample_task, RuntimeError("err"), move_to_dlq_if_max_attempts=False
        )
        assert moved is False
