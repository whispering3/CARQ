"""
Conftest for integration tests.
Provides integration-test-specific fixtures with real database and services.
"""

import pytest
import asyncio
from typing import List
from unittest.mock import AsyncMock, patch


# ============================================================================
# REAL DATABASE FIXTURES FOR INTEGRATION TESTS
# ============================================================================


@pytest.fixture
async def integration_db_session(test_session):
    """Provide database session for integration tests."""
    yield test_session


@pytest.fixture
async def cleanup_db(test_session):
    """Clean up database after integration tests."""
    yield
    # Cleanup is handled by test_session fixture rollback


# ============================================================================
# CONCURRENT ACCESS FIXTURES
# ============================================================================


@pytest.fixture
def concurrent_task_generator():
    """Generate tasks for concurrent access testing."""
    async def generate_tasks(count: int, task_factory):
        """Generate multiple tasks concurrently."""
        tasks = [task_factory() for _ in range(count)]
        return await asyncio.gather(*tasks, return_exceptions=True)

    return generate_tasks


# ============================================================================
# ERROR SCENARIO FIXTURES
# ============================================================================


@pytest.fixture
def mock_network_timeout():
    """Create mock for network timeout scenario."""
    async def timeout_handler(*args, **kwargs):
        await asyncio.sleep(5)
        raise TimeoutError("Network timeout")

    return timeout_handler


@pytest.fixture
def mock_database_error():
    """Create mock for database error scenario."""
    async def error_handler(*args, **kwargs):
        raise Exception("Database connection failed")

    return error_handler


@pytest.fixture
def mock_malformed_response():
    """Create mock for malformed response."""
    return {
        "error": "Invalid response format",
        "data": None,
    }


# ============================================================================
# PERFORMANCE TESTING FIXTURES
# ============================================================================


@pytest.fixture
def performance_benchmark():
    """Create performance benchmark fixture."""
    class Benchmark:
        def __init__(self):
            self.results = {}

        async def run_async(self, name: str, func, *args, **kwargs):
            """Run async function and measure time."""
            import time
            start = time.time()
            result = await func(*args, **kwargs)
            elapsed = time.time() - start
            self.results[name] = elapsed
            return result

        def run_sync(self, name: str, func, *args, **kwargs):
            """Run sync function and measure time."""
            import time
            start = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            self.results[name] = elapsed
            return result

        def get_summary(self):
            """Get summary of benchmark results."""
            if not self.results:
                return {}
            times = list(self.results.values())
            return {
                "total": sum(times),
                "min": min(times),
                "max": max(times),
                "avg": sum(times) / len(times),
                "count": len(times),
            }

    return Benchmark()


# ============================================================================
# LOAD TESTING FIXTURES
# ============================================================================


@pytest.fixture
def load_test_config():
    """Create configuration for load tests."""
    return {
        "concurrent_users": 10,
        "requests_per_user": 100,
        "think_time": 0.1,  # seconds
        "timeout": 30,  # seconds
    }


@pytest.fixture
def load_test_results():
    """Create results collector for load tests."""
    class LoadTestResults:
        def __init__(self):
            self.success_count = 0
            self.error_count = 0
            self.response_times = []

        def record_success(self, response_time: float):
            self.success_count += 1
            self.response_times.append(response_time)

        def record_error(self):
            self.error_count += 1

        @property
        def total_requests(self):
            return self.success_count + self.error_count

        @property
        def error_rate(self):
            if self.total_requests == 0:
                return 0
            return self.error_count / self.total_requests

        @property
        def avg_response_time(self):
            if not self.response_times:
                return 0
            return sum(self.response_times) / len(self.response_times)

        @property
        def p95_response_time(self):
            if not self.response_times:
                return 0
            sorted_times = sorted(self.response_times)
            index = int(len(sorted_times) * 0.95)
            return sorted_times[index]

        @property
        def p99_response_time(self):
            if not self.response_times:
                return 0
            sorted_times = sorted(self.response_times)
            index = int(len(sorted_times) * 0.99)
            return sorted_times[index]

    return LoadTestResults()


# ============================================================================
# E2E TESTING FIXTURES
# ============================================================================


@pytest.fixture
async def e2e_test_harness(test_session):
    """Create E2E test harness with all required services."""
    class E2EHarness:
        def __init__(self, session):
            self.session = session
            self.created_documents = []
            self.created_tasks = []

        async def cleanup(self):
            """Clean up created entities."""
            # Cleanup is handled by test_session fixture
            pass

    return E2EHarness(test_session)


# ============================================================================
# RACE CONDITION TESTING FIXTURES
# ============================================================================


@pytest.fixture
def race_condition_barrier():
    """Create barrier for synchronizing concurrent operations."""
    class Barrier:
        def __init__(self, count: int):
            self.count = count
            self.events = [asyncio.Event() for _ in range(count)]
            self.ready_count = 0

        async def wait_all(self, index: int):
            """Wait for all threads to reach this point."""
            self.ready_count += 1
            if self.ready_count >= self.count:
                for event in self.events:
                    event.set()
            await self.events[index].wait()

    def create_barrier(count: int):
        return Barrier(count)

    return create_barrier


# ============================================================================
# MOCK EXTERNAL SERVICE FIXTURES
# ============================================================================


@pytest.fixture
def mock_openai_service():
    """Create mock OpenAI service for integration tests."""
    async def mock_embed(texts: List[str], model: str = "text-embedding-3-small"):
        """Mock OpenAI embedding."""
        return [
            {
                "embedding": [0.1 + i * 0.001] * 1536,
                "index": i,
            }
            for i in range(len(texts))
        ]

    return {
        "embed": mock_embed,
    }


@pytest.fixture
def mock_pdf_service():
    """Create mock PDF service for integration tests."""
    async def mock_parse(file_path: str):
        """Mock PDF parsing."""
        return {
            "num_pages": 5,
            "text_content": "Sample PDF content",
            "metadata": {"title": "Test PDF"},
        }

    return {
        "parse": mock_parse,
    }


# ============================================================================
# STATE TRACKING FIXTURES
# ============================================================================


@pytest.fixture
def state_tracker():
    """Create state tracker for integration tests."""
    class StateTracker:
        def __init__(self):
            self.state = {}

        def set(self, key: str, value):
            self.state[key] = value

        def get(self, key: str):
            return self.state.get(key)

        def clear(self):
            self.state.clear()

    return StateTracker()
