"""
Unit tests for worker/worker_pool.py and worker/task_coordinator.py.
"""
import asyncio
import uuid
from unittest.mock import AsyncMock

import pytest

from carq.core.exceptions import RetryableError, TaskTimeoutError
from carq.models.models import TaskType
from carq.worker.task_coordinator import CoordinatorConfig, TaskCoordinator
from carq.worker.worker_pool import WorkerConfig, WorkerPool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    return WorkerConfig(
        worker_id="test-worker-1",
        max_workers=2,
        task_timeout=5,
        max_retries=3,
    )


@pytest.fixture
def pool(config):
    return WorkerPool(config=config, num_workers=2)


@pytest.fixture
def mock_task(sample_task):
    return sample_task


# ---------------------------------------------------------------------------
# Tests: WorkerConfig
# ---------------------------------------------------------------------------

class TestWorkerConfig:
    def test_defaults(self):
        c = WorkerConfig(worker_id="w1")
        assert c.max_workers == 4
        assert c.task_timeout == 300
        assert c.max_retries == 3
        assert c.task_types is None

    def test_custom_values(self):
        c = WorkerConfig(worker_id="w2", max_workers=8, task_timeout=60)
        assert c.max_workers == 8
        assert c.task_timeout == 60


# ---------------------------------------------------------------------------
# Tests: WorkerPool initialization
# ---------------------------------------------------------------------------

class TestWorkerPoolInit:
    def test_default_config(self):
        pool = WorkerPool(num_workers=2)
        assert pool.config is not None
        assert pool.num_workers == 2
        assert pool.is_running is False

    def test_custom_config(self, pool, config):
        assert pool.config.worker_id == "test-worker-1"
        assert pool.config.max_workers == 2

    def test_no_handlers_initially(self, pool):
        assert pool._task_handlers == {}

    def test_active_task_count_zero(self, pool):
        assert pool.active_task_count == 0


# ---------------------------------------------------------------------------
# Tests: register_handler
# ---------------------------------------------------------------------------

class TestRegisterHandler:
    def test_register_single_handler(self, pool):
        handler = AsyncMock()
        pool.register_handler(TaskType.PARSE_PDF, handler)
        assert TaskType.PARSE_PDF in pool._task_handlers

    def test_register_multiple_handlers(self, pool):
        h1, h2 = AsyncMock(), AsyncMock()
        pool.register_handler(TaskType.PARSE_PDF, h1)
        pool.register_handler(TaskType.CHUNK_DOCUMENT, h2)
        assert len(pool._task_handlers) == 2


# ---------------------------------------------------------------------------
# Tests: execute_task
# ---------------------------------------------------------------------------

class TestExecuteTask:
    async def test_execute_success(self, pool, mock_task):
        handler = AsyncMock(return_value="done")
        result = await pool.execute_task(mock_task, handler)
        assert result == "done"
        handler.assert_awaited_once_with(mock_task)

    async def test_execute_removes_from_active_on_success(self, pool, mock_task):
        handler = AsyncMock(return_value=None)
        await pool.execute_task(mock_task, handler)
        assert mock_task.id not in pool._active_tasks

    async def test_execute_timeout_raises_task_timeout_error(self, pool, mock_task):
        async def slow_handler(task):
            await asyncio.sleep(100)

        pool.config.task_timeout = 0.01
        with pytest.raises(TaskTimeoutError):
            await pool.execute_task(mock_task, slow_handler)

    async def test_execute_removes_from_active_on_timeout(self, pool, mock_task):
        async def slow_handler(task):
            await asyncio.sleep(100)

        pool.config.task_timeout = 0.01
        with pytest.raises(TaskTimeoutError):
            await pool.execute_task(mock_task, slow_handler)
        assert mock_task.id not in pool._active_tasks

    async def test_execute_propagates_exception(self, pool, mock_task):
        handler = AsyncMock(side_effect=ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            await pool.execute_task(mock_task, handler)

    async def test_execute_removes_from_active_on_error(self, pool, mock_task):
        handler = AsyncMock(side_effect=RuntimeError("crash"))
        with pytest.raises(RuntimeError):
            await pool.execute_task(mock_task, handler)
        assert mock_task.id not in pool._active_tasks


# ---------------------------------------------------------------------------
# Tests: process_task
# ---------------------------------------------------------------------------

class TestProcessTask:
    async def test_no_handler_returns_false(self, pool, mock_task):
        success, error = await pool.process_task(mock_task)
        assert success is False
        assert "No handler" in error

    async def test_success_returns_true(self, pool, mock_task):
        handler = AsyncMock(return_value="ok")
        pool.register_handler(mock_task.task_type, handler)
        success, error = await pool.process_task(mock_task)
        assert success is True
        assert error is None

    async def test_retryable_error_returns_false(self, pool, mock_task):
        handler = AsyncMock(side_effect=RetryableError("rate limited"))
        pool.register_handler(mock_task.task_type, handler)
        success, error = await pool.process_task(mock_task)
        assert success is False
        assert error is not None

    async def test_timeout_returns_false(self, pool, mock_task):
        async def slow(task):
            await asyncio.sleep(100)

        pool.config.task_timeout = 0.01
        pool.register_handler(mock_task.task_type, slow)
        success, error = await pool.process_task(mock_task)
        assert success is False
        assert error is not None

    async def test_generic_exception_returns_false(self, pool, mock_task):
        handler = AsyncMock(side_effect=Exception("unexpected"))
        pool.register_handler(mock_task.task_type, handler)
        success, error = await pool.process_task(mock_task)
        assert success is False
        assert "unexpected" in error


# ---------------------------------------------------------------------------
# Tests: start / stop
# ---------------------------------------------------------------------------

class TestWorkerPoolLifecycle:
    async def test_start_sets_running(self, pool):
        await pool.start()
        assert pool.is_running is True
        await pool.stop(wait=False)

    async def test_stop_clears_running(self, pool):
        await pool.start()
        await pool.stop(wait=False)
        assert pool.is_running is False

    async def test_stop_with_no_active_tasks(self, pool):
        await pool.start()
        await pool.stop(wait=True)
        assert pool.is_running is False


# ---------------------------------------------------------------------------
# Tests: get_status
# ---------------------------------------------------------------------------

class TestGetStatus:
    async def test_status_structure(self, pool):
        await pool.start()
        status = pool.get_status()
        assert "worker_id" in status
        assert "running" in status
        assert "active_tasks" in status
        assert "max_workers" in status
        assert "handlers" in status
        await pool.stop(wait=False)

    async def test_status_reflects_handlers(self, pool):
        pool.register_handler(TaskType.PARSE_PDF, AsyncMock())
        await pool.start()
        status = pool.get_status()
        assert TaskType.PARSE_PDF.value in status["handlers"]
        await pool.stop(wait=False)


# ---------------------------------------------------------------------------
# Tests: submit
# ---------------------------------------------------------------------------

class TestSubmit:
    async def test_submit_executes_coroutine(self, pool):
        async def my_coro():
            return 42

        result = await pool.submit(my_coro())
        assert result == 42


# ---------------------------------------------------------------------------
# Tests: TaskCoordinator
# ---------------------------------------------------------------------------

class TestTaskCoordinatorInit:
    def test_init_no_session(self):
        coord = TaskCoordinator()
        assert coord.session is None
        assert coord.queue_manager is None
        assert coord._running is False

    def test_init_with_config(self):
        cfg = CoordinatorConfig(dequeue_interval=5, stuck_task_timeout=120)
        coord = TaskCoordinator(config=cfg)
        assert coord.config.dequeue_interval == 5
        assert coord.config.stuck_task_timeout == 120


class TestRegisterWorkerPool:
    def test_register_pool(self):
        coord = TaskCoordinator()
        pool = WorkerPool(config=WorkerConfig(worker_id="w-test"))
        coord.register_worker_pool(pool)
        assert "w-test" in coord._worker_pools


class TestCoordinatorLifecycle:
    async def test_start_stop(self):
        coord = TaskCoordinator()
        await coord.start()
        assert coord._running is True
        await coord.stop()
        assert coord._running is False

    async def test_start_starts_pools(self):
        coord = TaskCoordinator()
        pool = WorkerPool(config=WorkerConfig(worker_id="w1"))
        coord.register_worker_pool(pool)
        await coord.start()
        assert pool.is_running is True
        await coord.stop()

    async def test_stop_stops_pools(self):
        coord = TaskCoordinator()
        pool = WorkerPool(config=WorkerConfig(worker_id="w1"))
        coord.register_worker_pool(pool)
        await coord.start()
        await coord.stop()
        assert pool.is_running is False


class TestAssignTask:
    async def test_assign_returns_task_info(self):
        coord = TaskCoordinator()
        result = await coord.assign_task("worker-1", {"type": "parse"})
        assert "task_id" in result
        assert result["worker_id"] == "worker-1"
        assert result["status"] == "assigned"

    async def test_assign_tracks_assignment(self):
        coord = TaskCoordinator()
        result = await coord.assign_task("worker-1", {})
        status = await coord.get_task_status(result["task_id"])
        assert status["worker_id"] == "worker-1"


class TestRecordCompletion:
    async def test_record_success(self):
        coord = TaskCoordinator()
        await coord.record_completion("any-id", success=True)
        assert coord._metrics["tasks_completed"] == 1

    async def test_record_failure(self):
        coord = TaskCoordinator()
        await coord.record_completion("any-id", success=False)
        assert coord._metrics["tasks_failed"] == 1


class TestGetTaskStatus:
    async def test_unknown_task(self):
        coord = TaskCoordinator()
        status = await coord.get_task_status(str(uuid.uuid4()))
        assert status["status"] == "unknown"

    async def test_assigned_task(self):
        coord = TaskCoordinator()
        result = await coord.assign_task("w1", {})
        status = await coord.get_task_status(result["task_id"])
        assert status["status"] == "assigned"


class TestGetMetrics:
    def test_initial_metrics(self):
        coord = TaskCoordinator()
        m = coord.get_metrics()
        assert m["tasks_completed"] == 0
        assert m["tasks_failed"] == 0
        assert m["tasks_retried"] == 0
        assert m["active_workers"] == 0

    def test_metrics_with_pool(self):
        coord = TaskCoordinator()
        pool = WorkerPool(config=WorkerConfig(worker_id="w1"))
        coord.register_worker_pool(pool)
        m = coord.get_metrics()
        assert m["total_worker_pools"] == 1


class TestGetCoordinatorStatus:
    async def test_status_structure(self):
        coord = TaskCoordinator()
        await coord.start()
        status = coord.get_status()
        assert "running" in status
        assert "worker_pools" in status
        assert "metrics" in status
        assert "background_tasks" in status
        assert status["running"] is True
        await coord.stop()

    async def test_status_running_false_when_stopped(self):
        coord = TaskCoordinator()
        await coord.start()
        await coord.stop()
        status = coord.get_status()
        assert status["running"] is False


# ---------------------------------------------------------------------------
# Tests: CoordinatorConfig
# ---------------------------------------------------------------------------

class TestCoordinatorConfig:
    def test_defaults(self):
        cfg = CoordinatorConfig()
        assert cfg.dequeue_interval == 1
        assert cfg.stuck_task_timeout == 300
        assert cfg.status_update_interval == 30
        assert cfg.enable_stuck_task_recovery is True

    def test_custom(self):
        cfg = CoordinatorConfig(dequeue_interval=10, stuck_task_timeout=60)
        assert cfg.dequeue_interval == 10
        assert cfg.stuck_task_timeout == 60
