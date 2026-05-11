"""Tests for worker/task_coordinator.py - Phase 8.

Covers: creation, assign_task, record_completion, get_task_status,
metrics, register_worker_pool, and start/stop lifecycle.
"""

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from carq.worker.task_coordinator import TaskCoordinator, CoordinatorConfig
from carq.worker.worker_pool import WorkerPool, WorkerConfig
from carq.models.models import TaskType


# ============================================================================
# CREATION TESTS
# ============================================================================


def test_coordinator_creation_no_args():
    """TaskCoordinator can be created with no arguments."""
    coordinator = TaskCoordinator()
    assert coordinator is not None
    assert coordinator._running is False
    assert coordinator._worker_pools == {}


def test_coordinator_creation_with_config():
    """TaskCoordinator accepts a CoordinatorConfig."""
    config = CoordinatorConfig(
        dequeue_interval=5,
        stuck_task_timeout=600,
        status_update_interval=60,
        enable_stuck_task_recovery=False,
    )
    coordinator = TaskCoordinator(config=config)
    assert coordinator.config.dequeue_interval == 5
    assert coordinator.config.stuck_task_timeout == 600


def test_coordinator_creation_with_session():
    """TaskCoordinator creates QueueManager when session is provided."""
    mock_session = AsyncMock()
    coordinator = TaskCoordinator(session=mock_session)
    assert coordinator.queue_manager is not None


def test_coordinator_creation_without_session():
    """TaskCoordinator has no queue_manager when session is None."""
    coordinator = TaskCoordinator(session=None)
    assert coordinator.queue_manager is None


def test_coordinator_default_config():
    """CoordinatorConfig has sensible defaults."""
    config = CoordinatorConfig()
    assert config.dequeue_interval == 1
    assert config.stuck_task_timeout == 300
    assert config.enable_stuck_task_recovery is True


# ============================================================================
# REGISTER WORKER POOL
# ============================================================================


def test_register_worker_pool():
    """register_worker_pool adds pool to internal dict."""
    coordinator = TaskCoordinator()
    config = WorkerConfig(worker_id="pool-1", max_workers=2)
    pool = WorkerPool(config=config)

    coordinator.register_worker_pool(pool)
    assert "pool-1" in coordinator._worker_pools


def test_register_multiple_worker_pools():
    """Multiple pools can be registered."""
    coordinator = TaskCoordinator()
    for i in range(3):
        config = WorkerConfig(worker_id=f"pool-{i}", max_workers=2)
        pool = WorkerPool(config=config)
        coordinator.register_worker_pool(pool)

    assert len(coordinator._worker_pools) == 3


# ============================================================================
# ASSIGN TASK
# ============================================================================


@pytest.mark.asyncio
async def test_assign_task_returns_dict():
    """assign_task returns a dict with task_id, worker_id, status."""
    coordinator = TaskCoordinator()
    result = await coordinator.assign_task(
        worker_id="worker-1",
        task_data={"type": "parse_pdf"},
    )
    assert "task_id" in result
    assert result["worker_id"] == "worker-1"
    assert result["status"] == "assigned"


@pytest.mark.asyncio
async def test_assign_task_generates_unique_ids():
    """Each assign_task call generates a distinct task_id."""
    coordinator = TaskCoordinator()
    r1 = await coordinator.assign_task("w1", {})
    r2 = await coordinator.assign_task("w1", {})
    assert r1["task_id"] != r2["task_id"]


@pytest.mark.asyncio
async def test_assign_task_stores_assignment():
    """Assigned task_id is tracked in _task_assignments."""
    coordinator = TaskCoordinator()
    result = await coordinator.assign_task("worker-99", {"type": "embed"})
    task_uuid = uuid.UUID(result["task_id"])
    assert task_uuid in coordinator._task_assignments
    assert coordinator._task_assignments[task_uuid] == "worker-99"


# ============================================================================
# RECORD COMPLETION
# ============================================================================


@pytest.mark.asyncio
async def test_record_completion_success_increments_completed():
    """record_completion with success=True increments tasks_completed."""
    coordinator = TaskCoordinator()
    await coordinator.record_completion("task-123", success=True)
    assert coordinator._metrics["tasks_completed"] == 1
    assert coordinator._metrics["tasks_failed"] == 0


@pytest.mark.asyncio
async def test_record_completion_failure_increments_failed():
    """record_completion with success=False increments tasks_failed."""
    coordinator = TaskCoordinator()
    await coordinator.record_completion("task-456", success=False)
    assert coordinator._metrics["tasks_failed"] == 1
    assert coordinator._metrics["tasks_completed"] == 0


@pytest.mark.asyncio
async def test_record_completion_multiple_calls():
    """Multiple record_completion calls accumulate correctly."""
    coordinator = TaskCoordinator()
    for _ in range(3):
        await coordinator.record_completion("t", success=True)
    for _ in range(2):
        await coordinator.record_completion("t", success=False)

    assert coordinator._metrics["tasks_completed"] == 3
    assert coordinator._metrics["tasks_failed"] == 2


# ============================================================================
# GET TASK STATUS
# ============================================================================


@pytest.mark.asyncio
async def test_get_task_status_unknown_task():
    """get_task_status returns unknown for unassigned task_id."""
    coordinator = TaskCoordinator()
    task_id = str(uuid.uuid4())
    result = await coordinator.get_task_status(task_id)
    assert result["status"] == "unknown"
    assert result["worker_id"] is None


@pytest.mark.asyncio
async def test_get_task_status_assigned_task():
    """get_task_status returns assigned status after assign_task."""
    coordinator = TaskCoordinator()
    assigned = await coordinator.assign_task("worker-7", {})
    task_id = assigned["task_id"]

    status = await coordinator.get_task_status(task_id)
    assert status["status"] == "assigned"
    assert status["worker_id"] == "worker-7"


@pytest.mark.asyncio
async def test_get_task_status_returns_task_id():
    """get_task_status echoes back the task_id."""
    coordinator = TaskCoordinator()
    task_id = str(uuid.uuid4())
    result = await coordinator.get_task_status(task_id)
    assert result["task_id"] == task_id


# ============================================================================
# METRICS
# ============================================================================


def test_get_metrics_initial_state():
    """get_metrics returns zeroed counters on new coordinator."""
    coordinator = TaskCoordinator()
    metrics = coordinator.get_metrics()
    assert metrics["tasks_completed"] == 0
    assert metrics["tasks_failed"] == 0
    assert metrics["tasks_retried"] == 0
    assert metrics["errors_total"] == 0
    assert metrics["active_workers"] == 0
    assert metrics["total_worker_pools"] == 0


def test_get_metrics_with_pools():
    """get_metrics reports total_worker_pools correctly."""
    coordinator = TaskCoordinator()
    for i in range(2):
        config = WorkerConfig(worker_id=f"pool-{i}", max_workers=2)
        pool = WorkerPool(config=config)
        coordinator.register_worker_pool(pool)

    metrics = coordinator.get_metrics()
    assert metrics["total_worker_pools"] == 2


def test_get_status_includes_running_flag():
    """get_status reports running=False before start."""
    coordinator = TaskCoordinator()
    status = coordinator.get_status()
    assert status["running"] is False


def test_get_status_includes_metrics():
    """get_status includes a metrics key."""
    coordinator = TaskCoordinator()
    status = coordinator.get_status()
    assert "metrics" in status


def test_get_status_includes_worker_pools():
    """get_status reports worker_pools."""
    coordinator = TaskCoordinator()
    status = coordinator.get_status()
    assert "worker_pools" in status


# ============================================================================
# START / STOP
# ============================================================================


@pytest.mark.asyncio
async def test_coordinator_stop_sets_running_false():
    """stop() sets _running to False."""
    coordinator = TaskCoordinator()
    coordinator._running = True
    await coordinator.stop()
    assert coordinator._running is False


@pytest.mark.asyncio
async def test_coordinator_start_sets_running_true():
    """start() sets _running to True (with mocked queue_manager)."""
    mock_session = AsyncMock()
    coordinator = TaskCoordinator(session=mock_session)

    # Mock queue_manager to avoid real DB calls in background tasks
    coordinator.queue_manager = AsyncMock()
    coordinator.queue_manager.dequeue_task = AsyncMock(return_value=[])
    coordinator.queue_manager.reset_stuck_tasks = AsyncMock(return_value=0)
    coordinator.queue_manager.get_queue_status = AsyncMock(return_value={})

    await coordinator.start()
    assert coordinator._running is True
    await coordinator.stop()


@pytest.mark.asyncio
async def test_coordinator_stop_cancels_background_tasks():
    """stop() cancels all background tasks."""
    coordinator = TaskCoordinator()

    # Add fake background tasks
    async def _forever():
        await asyncio.sleep(9999)

    t = asyncio.create_task(_forever())
    coordinator._background_tasks.append(t)

    await coordinator.stop()
    assert t.cancelled() or t.done()


@pytest.mark.asyncio
async def test_process_tasks_updates_metrics():
    """_process_tasks updates queue and metrics for success/failure."""
    coordinator = TaskCoordinator()
    coordinator.queue_manager = AsyncMock()
    coordinator.session = AsyncMock()
    pool = AsyncMock()
    success_task = AsyncMock()
    success_task.id = uuid.uuid4()
    success_task.attempt_count = 0
    success_task.max_attempts = 3

    failure_task = AsyncMock()
    failure_task.id = uuid.uuid4()
    failure_task.attempt_count = 3
    failure_task.max_attempts = 3

    pool.process_task = AsyncMock(side_effect=[(True, None), (False, "boom")])
    await coordinator._process_tasks([success_task, failure_task], pool)

    assert coordinator._metrics["tasks_completed"] == 1
    assert coordinator._metrics["tasks_failed"] == 1
    assert coordinator._metrics["errors_total"] == 1
    assert coordinator.queue_manager.mark_task_done.await_count == 1
    assert coordinator.queue_manager.mark_task_failed.await_count == 1


@pytest.mark.asyncio
async def test_start_registers_background_tasks():
    """start() starts pools and creates background tasks."""
    coordinator = TaskCoordinator()
    pool = AsyncMock()
    pool.config = WorkerConfig(worker_id="pool-1", max_workers=1, task_types=[TaskType.PARSE_PDF])
    pool.start = AsyncMock()
    coordinator.register_worker_pool(pool)
    coordinator.queue_manager = AsyncMock()
    coordinator.queue_manager.dequeue_task = AsyncMock(return_value=[])
    coordinator.queue_manager.reset_stuck_tasks = AsyncMock(return_value=0)
    coordinator.queue_manager.get_queue_status = AsyncMock(return_value={})

    dummy_tasks = []

    def fake_create_task(coro):
        coro.close()
        task = asyncio.get_running_loop().create_future()
        task.set_result(None)
        dummy_tasks.append(task)
        return task

    with patch("carq.worker.task_coordinator.asyncio.create_task", side_effect=fake_create_task):
        await coordinator.start()

    assert coordinator._running is True
    assert len(coordinator._background_tasks) == 3
    pool.start.assert_awaited_once()


@pytest.mark.asyncio
async def test_dequeue_loop_runs_once():
    """_dequeue_loop should dequeue and stop cleanly."""
    coordinator = TaskCoordinator()
    coordinator.queue_manager = AsyncMock()
    coordinator.queue_manager.dequeue_task = AsyncMock(return_value=[])
    coordinator._running = True
    pool = AsyncMock()
    pool.config = WorkerConfig(worker_id="pool-1", max_workers=1, task_types=[TaskType.PARSE_PDF])
    coordinator.register_worker_pool(pool)

    async def stop_after_first(*args, **kwargs):
        coordinator._running = False
        return []

    coordinator.queue_manager.dequeue_task.side_effect = stop_after_first

    async def noop_sleep(*args, **kwargs):
        return None

    with patch("carq.worker.task_coordinator.asyncio.sleep", side_effect=noop_sleep):
        await coordinator._dequeue_loop()

    assert coordinator.queue_manager.dequeue_task.await_count == 1


@pytest.mark.asyncio
async def test_stuck_task_and_status_loops_run_once():
    """Recovery/status loops should execute a single iteration."""
    coordinator = TaskCoordinator()
    coordinator.queue_manager = AsyncMock()
    coordinator.queue_manager.reset_stuck_tasks = AsyncMock(return_value=1)
    coordinator.queue_manager.get_queue_status = AsyncMock(return_value={"pending": 0})
    coordinator.session = AsyncMock()
    coordinator._running = True

    async def stop_recovery(*args, **kwargs):
        coordinator._running = False
        return 1

    coordinator.queue_manager.reset_stuck_tasks.side_effect = stop_recovery

    async def stop_status(*args, **kwargs):
        coordinator._running = False
        return {"pending": 0}

    coordinator.queue_manager.get_queue_status.side_effect = stop_status

    async def noop_sleep(*args, **kwargs):
        return None

    with patch("carq.worker.task_coordinator.asyncio.sleep", side_effect=noop_sleep):
        await coordinator._stuck_task_recovery_loop()
        coordinator._running = True
        await coordinator._status_update_loop()
