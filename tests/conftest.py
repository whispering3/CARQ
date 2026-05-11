"""
Comprehensive conftest for test fixtures and configuration.
Provides fixtures for unit, integration, and performance tests.
"""

import os
import asyncio
import json
from datetime import datetime, timedelta
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

# Disable logging during test discovery
import logging
logging.disable(logging.CRITICAL)

# Conditional imports to handle version compatibility
try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
    from sqlalchemy import text
    from sqlalchemy.pool import StaticPool
except (ImportError, AssertionError):
    # Handle SQLAlchemy compatibility issues
    create_async_engine = None
    AsyncSession = None
    async_sessionmaker = None
    text = None
    StaticPool = None

try:
    from carq.models.base import Base
    from carq.core.config import Settings, DatabaseSettings, RedisSettings
    from carq.models.models import (
        Document, DocumentStatus,
        Chunk, ChunkStatus,
        Embedding,
        ProcessingTask, TaskStatus, TaskType,
        TaskDeadletter,
    )
    from carq.queue.rate_limiter import TokenBucket, RateLimiter
    from carq.queue.circuit_breaker import CircuitBreaker
    from carq.queue.backoff_strategy import BackoffStrategy
except (ImportError, AttributeError, ValueError):
    # Handle missing modules
    Base = None
    Settings = None
    DatabaseSettings = None
    RedisSettings = None

# Re-enable logging
logging.disable(logging.NOTSET)


# ============================================================================
# EVENT LOOP FIXTURES
# ============================================================================


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for session-scoped tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# DATABASE FIXTURES
# ============================================================================


@pytest.fixture
async def test_engine():
    """Create test database engine (function-scoped for isolation)."""
    if create_async_engine is None:
        pytest.skip("SQLAlchemy not available")
    
    database_url = os.getenv(
        "TEST_DATABASE_URL",
        "sqlite+aiosqlite:///:memory:",
    )

    engine = create_async_engine(
        database_url,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    if Base is not None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    yield engine

    if Base is not None:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    await engine.dispose()


@pytest.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create isolated test database session for each test."""
    if async_sessionmaker is None or AsyncSession is None:
        pytest.skip("SQLAlchemy not available")
    
    session_maker = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture
def anyio_backend():
    """Use asyncio as backend for async tests."""
    return "asyncio"


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================


@pytest.fixture
def test_settings():
    """Create test configuration."""
    if Settings is None:
        pytest.skip("Settings not available")
    
    return Settings(
        app_name="CARQ Test",
        debug=True,
        api_base_url="http://localhost:8000",
        jwt_secret="test-secret-key-not-for-production",
        jwt_algorithm="HS256",
        database=DatabaseSettings(
            host="localhost",
            port=5432,
            user="test",
            password="test",
            database="carq_test",
        ),
        redis=RedisSettings(
            enabled=False,
        ),
        openai_api_key="sk-test-key",
        embedding_model="text-embedding-3-small",
    )


# ============================================================================
# TEST DATA FIXTURES
# ============================================================================


@pytest.fixture
def sample_document_data() -> dict:
    """Create sample document data."""
    try:
        status = DocumentStatus.PENDING if DocumentStatus else "pending"
    except:
        status = "pending"
    
    return {
        "source_uri": f"s3://bucket/test-{uuid4()}.pdf",
        "content_hash": b"test-hash-123",
        "status": status,
        "attributes": {
            "category": "test",
            "source": "unit_test",
        },
        "document_type": "pdf",
    }


@pytest.fixture
async def sample_document(test_session, sample_document_data):
    """Create a sample document in test database."""
    if Document is None:
        pytest.skip("Document model not available")
    
    doc = Document(**sample_document_data)
    test_session.add(doc)
    await test_session.commit()
    return doc


@pytest.fixture
def sample_chunk_data() -> dict:
    """Create sample chunk data."""
    doc_id = uuid4()
    return {
        "document_id": doc_id,
        "chunk_index": 0,
        "content": "This is a sample chunk for testing purposes.",
        "content_hash": b"chunk-hash",
        "status": ChunkStatus.PENDING,
        "attributes": {
            "page": 1,
            "source": "unit_test",
        },
    }


@pytest.fixture
async def sample_chunk(test_session, sample_document):
    """Create a sample chunk in test database."""
    chunk = Chunk(
        document_id=sample_document.id,
        chunk_index=0,
        content="Sample chunk content for testing.",
        content_hash=b"chunk-hash-123",
        status=ChunkStatus.PENDING,
        attributes={"page": 1},
    )
    test_session.add(chunk)
    await test_session.commit()
    return chunk


@pytest.fixture
def sample_task_data() -> dict:
    """Create sample task data."""
    doc_id = uuid4()
    return {
        "document_id": doc_id,
        "task_type": TaskType.PARSE_PDF,
        "status": TaskStatus.PENDING,
        "priority": 5,
        "max_attempts": 3,
        "attributes": {
            "source": "unit_test",
        },
    }


@pytest.fixture
async def sample_task(test_session, sample_document):
    """Create a sample task in test database."""
    task = ProcessingTask(
        document_id=sample_document.id,
        task_type=TaskType.PARSE_PDF,
        status=TaskStatus.PENDING,
        priority=5,
        attributes={"source": "unit_test"},
    )
    test_session.add(task)
    await test_session.commit()
    return task


@pytest.fixture
def sample_embedding_data() -> dict:
    """Create sample embedding data."""
    chunk_id = uuid4()
    return {
        "chunk_id": chunk_id,
        "text": "Sample embedding text",
        "model": "text-embedding-3-small",
        "embedding": [0.1] * 1536,
        "tokens_used": 10,
        "cost_usd": 0.00001,
        "attributes": {
            "model_version": "v1",
        },
    }


@pytest.fixture
async def sample_embedding(test_session, sample_chunk):
    """Create a sample embedding in test database."""
    embedding = Embedding(
        chunk_id=sample_chunk.id,
        text="Sample embedding text",
        model="text-embedding-3-small",
        embedding=[0.1] * 1536,
        tokens_used=10,
        cost_usd=0.00001,
        attributes={"model_version": "v1"},
    )
    test_session.add(embedding)
    await test_session.commit()
    return embedding


# ============================================================================
# MOCK SERVICE FIXTURES
# ============================================================================


@pytest.fixture
def mock_openai_client():
    """Create mock OpenAI client."""
    mock_client = AsyncMock()
    mock_client.embeddings.create = AsyncMock(
        return_value=Mock(
            data=[
                Mock(embedding=[0.1] * 1536, index=0),
                Mock(embedding=[0.2] * 1536, index=1),
            ]
        )
    )
    return mock_client


@pytest.fixture
def mock_pdf_parser():
    """Create mock PDF parser."""
    mock_parser = Mock()
    mock_parser.parse = Mock(
        return_value=Mock(
            num_pages=5,
            text_content="Sample PDF content",
            metadata={"title": "Test PDF"},
        )
    )
    return mock_parser


@pytest.fixture
def mock_redis_client():
    """Create mock Redis client."""
    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)
    mock_redis.set = AsyncMock(return_value=True)
    mock_redis.delete = AsyncMock(return_value=1)
    return mock_redis


@pytest.fixture
def mock_database_session():
    """Create mock database session."""
    mock_session = AsyncMock(spec=AsyncSession)
    mock_session.commit = AsyncMock()
    mock_session.rollback = AsyncMock()
    mock_session.close = AsyncMock()
    return mock_session


# ============================================================================
# RESILIENCE FIXTURES
# ============================================================================


@pytest.fixture
def rate_limiter():
    """Create rate limiter for testing."""
    return TokenBucket(
        capacity=100,
        refill_rate=10,
    )


@pytest.fixture
def circuit_breaker():
    """Create circuit breaker for testing."""
    return CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=60,
        success_threshold=2,
    )


@pytest.fixture
def backoff_strategy():
    """Create backoff strategy for testing."""
    return BackoffStrategy(
        initial_delay=0.1,
        max_delay=10,
        exponential_base=2,
        jitter=True,
    )


# ============================================================================
# PERFORMANCE TESTING FIXTURES
# ============================================================================


@pytest.fixture
def performance_timer():
    """Create performance timer."""
    class PerfTimer:
        def __init__(self):
            self.start_time = None
            self.end_time = None

        def start(self):
            self.start_time = datetime.now()

        def stop(self):
            self.end_time = datetime.now()

        @property
        def elapsed_ms(self) -> float:
            if self.start_time and self.end_time:
                return (self.end_time - self.start_time).total_seconds() * 1000
            return 0

    return PerfTimer()


@pytest.fixture
def metrics_collector():
    """Create metrics collector for performance testing."""
    class MetricsCollector:
        def __init__(self):
            self.timings = []
            self.errors = []
            self.successes = 0

        def record_timing(self, elapsed_ms: float):
            self.timings.append(elapsed_ms)

        def record_error(self, error: str):
            self.errors.append(error)

        def record_success(self):
            self.successes += 1

        @property
        def p50(self) -> float:
            if not self.timings:
                return 0
            sorted_timings = sorted(self.timings)
            return sorted_timings[len(sorted_timings) // 2]

        @property
        def p99(self) -> float:
            if not self.timings:
                return 0
            sorted_timings = sorted(self.timings)
            index = int(len(sorted_timings) * 0.99)
            return sorted_timings[index]

        @property
        def error_rate(self) -> float:
            total = self.successes + len(self.errors)
            if total == 0:
                return 0
            return len(self.errors) / total

    return MetricsCollector()


# ============================================================================
# PYTEST CONFIGURATION
# ============================================================================


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "asyncio: Async tests")
    config.addinivalue_line("markers", "db: Database tests")
