"""Tests for core/database.py - Phase 8.

Covers: DatabaseManager initialization, session factory, get_session
context manager, health_check, pool_status, and error handling.
"""

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch, PropertyMock

import pytest

from carq.core.database import DatabaseManager, db
from carq.core.exceptions import DatabaseError


# ============================================================================
# HELPERS
# ============================================================================


def make_db_manager():
    """Return a fresh DatabaseManager (not the global singleton)."""
    return DatabaseManager()


def make_initialized_manager():
    """Return a DatabaseManager in a fake-initialized state."""
    manager = DatabaseManager()
    manager._initialized = True

    # Mock engine
    mock_engine = MagicMock()
    mock_pool = MagicMock()
    mock_pool.__class__.__name__ = "QueuePool"
    mock_engine.pool = mock_pool
    mock_engine.dispose = AsyncMock()
    manager._engine = mock_engine

    # Mock session factory
    mock_session = AsyncMock()
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()
    mock_session.execute = AsyncMock()

    mock_factory = MagicMock()
    mock_factory.return_value = mock_session
    manager._session_factory = mock_factory

    return manager, mock_session


# ============================================================================
# INITIALIZATION TESTS
# ============================================================================


def test_database_manager_initial_state():
    """DatabaseManager starts uninitialized."""
    manager = make_db_manager()
    assert manager._initialized is False
    assert manager._engine is None
    assert manager._session_factory is None


def test_is_initialized_property_false():
    """is_initialized returns False before initialize()."""
    manager = make_db_manager()
    assert manager.is_initialized is False


def test_is_initialized_property_true():
    """is_initialized returns True after mock-initialization."""
    manager, _ = make_initialized_manager()
    assert manager.is_initialized is True


def test_engine_property_raises_when_not_initialized():
    """engine property raises DatabaseError when not initialized."""
    manager = make_db_manager()
    with pytest.raises(DatabaseError):
        _ = manager.engine


def test_engine_property_returns_engine_when_initialized():
    """engine property returns the engine when initialized."""
    manager, _ = make_initialized_manager()
    engine = manager.engine
    assert engine is not None


# ============================================================================
# INITIALIZE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_initialize_creates_engine():
    """initialize() creates an async engine."""
    manager = make_db_manager()

    mock_engine = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock()
    mock_engine.begin = MagicMock(return_value=AsyncMock(
        __aenter__=AsyncMock(return_value=mock_conn),
        __aexit__=AsyncMock(return_value=False),
    ))
    mock_engine.sync_engine = MagicMock()
    mock_engine.pool = MagicMock()

    mock_session_factory = MagicMock()
    mock_session = AsyncMock()
    mock_session.execute = AsyncMock()
    mock_session_factory.return_value = mock_session

    with patch("carq.core.database.create_async_engine", return_value=mock_engine) as mock_create, \
         patch("carq.core.database.async_sessionmaker", return_value=mock_session_factory), \
         patch.object(manager, "_setup_connection_handlers"), \
         patch.object(manager, "_test_connection", new=AsyncMock()):
        await manager.initialize()

    assert manager._initialized is True
    mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_initialize_idempotent():
    """initialize() is a no-op when already initialized."""
    manager, _ = make_initialized_manager()

    with patch("carq.core.database.create_async_engine") as mock_create:
        await manager.initialize()

    mock_create.assert_not_called()


@pytest.mark.asyncio
async def test_initialize_raises_database_error_on_failure():
    """initialize() raises DatabaseError when engine creation fails."""
    manager = make_db_manager()

    with patch("carq.core.database.create_async_engine", side_effect=RuntimeError("DB down")):
        with pytest.raises(DatabaseError):
            await manager.initialize()


# ============================================================================
# SESSION TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_get_session_raises_when_not_initialized():
    """get_session raises DatabaseError if not initialized."""
    manager = make_db_manager()
    with pytest.raises(DatabaseError):
        async with manager.get_session() as session:
            pass


@pytest.mark.asyncio
async def test_get_session_yields_session():
    """get_session yields an AsyncSession."""
    manager, mock_session = make_initialized_manager()

    # Make mock_session work as an async context manager
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    async with manager.get_session() as session:
        assert session is not None


@pytest.mark.asyncio
async def test_get_session_commits_on_success():
    """get_session commits on successful exit."""
    manager, mock_session = make_initialized_manager()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    async with manager.get_session() as session:
        pass

    mock_session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_get_session_rolls_back_on_exception():
    """get_session rolls back when an exception occurs inside."""
    manager, mock_session = make_initialized_manager()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with pytest.raises(ValueError):
        async with manager.get_session() as session:
            raise ValueError("something went wrong")

    mock_session.rollback.assert_called_once()


@pytest.mark.asyncio
async def test_get_session_closes_on_success():
    """get_session always closes the session in finally block."""
    manager, mock_session = make_initialized_manager()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    async with manager.get_session() as session:
        pass

    mock_session.close.assert_called_once()


@pytest.mark.asyncio
async def test_get_session_closes_on_exception():
    """get_session closes session even after an exception."""
    manager, mock_session = make_initialized_manager()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with pytest.raises(RuntimeError):
        async with manager.get_session() as session:
            raise RuntimeError("crash")

    mock_session.close.assert_called_once()


# ============================================================================
# CLOSE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_close_disposes_engine():
    """close() disposes the engine."""
    manager, _ = make_initialized_manager()
    manager._engine.dispose = AsyncMock()

    await manager.close()
    manager._engine.dispose.assert_called_once()


@pytest.mark.asyncio
async def test_close_sets_initialized_false():
    """close() marks manager as uninitialized."""
    manager, _ = make_initialized_manager()
    manager._engine.dispose = AsyncMock()

    await manager.close()
    assert manager._initialized is False


@pytest.mark.asyncio
async def test_close_no_engine_noop():
    """close() does nothing when engine is None."""
    manager = make_db_manager()
    # Should not raise
    await manager.close()


# ============================================================================
# HEALTH CHECK TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_health_check_returns_true_when_healthy():
    """health_check returns True when DB query succeeds."""
    manager, mock_session = make_initialized_manager()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute = AsyncMock(return_value=MagicMock())

    result = await manager.health_check()
    assert result is True


@pytest.mark.asyncio
async def test_health_check_returns_false_on_failure():
    """health_check returns False when DB query fails."""
    manager, mock_session = make_initialized_manager()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.execute = AsyncMock(side_effect=RuntimeError("Connection lost"))

    result = await manager.health_check()
    assert result is False


@pytest.mark.asyncio
async def test_health_check_not_initialized_returns_false():
    """health_check returns False when not initialized."""
    manager = make_db_manager()
    result = await manager.health_check()
    assert result is False


# ============================================================================
# POOL STATUS TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_get_pool_status_not_initialized():
    """get_pool_status returns not_initialized before init."""
    manager = make_db_manager()
    status = await manager.get_pool_status()
    assert status["status"] == "not_initialized"


@pytest.mark.asyncio
async def test_get_pool_status_initialized():
    """get_pool_status returns pool info when initialized."""
    manager, _ = make_initialized_manager()

    mock_pool = MagicMock()
    mock_pool.__class__.__name__ = "QueuePool"
    manager._engine.pool = mock_pool

    status = await manager.get_pool_status()
    assert "pool_type" in status
    assert status["pool_type"] == "QueuePool"


# ============================================================================
# GLOBAL SINGLETON
# ============================================================================


def test_global_db_is_database_manager():
    """The global `db` object is a DatabaseManager instance."""
    from carq.core.database import db
    assert isinstance(db, DatabaseManager)


# ============================================================================
# GET DB SESSION DEPENDENCY
# ============================================================================


@pytest.mark.asyncio
async def test_get_db_session_dependency():
    """get_db_session is an async generator that yields a session."""
    from carq.core.database import get_db_session, db

    manager, mock_session = make_initialized_manager()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)

    with patch("carq.core.database.db", manager):
        gen = get_db_session()
        session = await gen.__anext__()
        assert session is not None
        try:
            await gen.__anext__()
        except StopAsyncIteration:
            pass
