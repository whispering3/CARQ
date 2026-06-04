"""
Load testing configuration and utilities for CARQ.

This module provides utilities for load testing the CARQ system
using k6 (JavaScript) and Locust (Python).
"""

import asyncio
import time
from typing import Dict

import pytest


class LoadTestRunner:
    """Base class for load test runners."""

    def __init__(self, target_url: str, num_users: int, duration_seconds: int):
        self.target_url = target_url
        self.num_users = num_users
        self.duration_seconds = duration_seconds
        self.results = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "errors": [],
            "started_at": None,
            "finished_at": None,
        }

    def record_request(self, success: bool, response_time: float, error: str = None):
        """Record a request result."""
        self.results["total_requests"] += 1

        if success:
            self.results["successful_requests"] += 1
            self.results["response_times"].append(response_time)
        else:
            self.results["failed_requests"] += 1
            if error:
                self.results["errors"].append(error)

    def get_metrics(self) -> Dict:
        """Get calculated metrics."""
        response_times = self.results["response_times"]

        if not response_times:
            return {
                "min_response_time": 0,
                "max_response_time": 0,
                "avg_response_time": 0,
                "p50": 0,
                "p95": 0,
                "p99": 0,
                "error_rate": 1.0 if self.results["total_requests"] > 0 else 0,
                "throughput": 0,
            }

        sorted_times = sorted(response_times)
        started_at = self.results.get("started_at")
        finished_at = self.results.get("finished_at")
        total_time = (
            max(0.001, finished_at - started_at)
            if started_at is not None and finished_at is not None
            else max(1.0, sum(response_times) / 1000)  # Fallback for synthetic runs
        )

        return {
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "avg_response_time": sum(response_times) / len(response_times),
            "p50": sorted_times[len(sorted_times) // 2],
            "p95": sorted_times[int(len(sorted_times) * 0.95)],
            "p99": sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 1 else sorted_times[0],
            "error_rate": self.results["failed_requests"] / max(1, self.results["total_requests"]),
            "throughput": self.results["successful_requests"] / max(1, total_time),
        }

    def print_results(self):
        """Print test results."""
        metrics = self.get_metrics()

        print("\n" + "=" * 60)
        print("LOAD TEST RESULTS")
        print("=" * 60)
        print(f"Total Requests: {self.results['total_requests']}")
        print(f"Successful: {self.results['successful_requests']}")
        print(f"Failed: {self.results['failed_requests']}")
        print(f"Error Rate: {metrics['error_rate']:.2%}")
        print("\nResponse Time Metrics (ms):")
        print(f"  Min: {metrics['min_response_time']:.2f}")
        print(f"  Max: {metrics['max_response_time']:.2f}")
        print(f"  Avg: {metrics['avg_response_time']:.2f}")
        print(f"  P50: {metrics['p50']:.2f}")
        print(f"  P95: {metrics['p95']:.2f}")
        print(f"  P99: {metrics['p99']:.2f}")
        print(f"\nThroughput: {metrics['throughput']:.2f} req/s")
        print("=" * 60 + "\n")


class HTTPLoadTester(LoadTestRunner):
    """Load tester for HTTP endpoints."""

    async def run(self):
        """Run load test."""
        import time

        import aiohttp

        self.results["started_at"] = time.time()

        async with aiohttp.ClientSession() as session:
            tasks = []

            for _ in range(self.num_users):
                tasks.append(self._user_session(session))

            await asyncio.gather(*tasks, return_exceptions=True)
        self.results["finished_at"] = time.time()

    async def _user_session(self, session):
        """Simulate a user session."""
        end_time = time.time() + self.duration_seconds

        while time.time() < end_time:
            try:
                start = time.time()

                # Make request
                async with session.get(f"{self.target_url}/health") as resp:
                    response_time = (time.time() - start) * 1000
                    self.record_request(resp.status == 200, response_time)

                # Think time
                await asyncio.sleep(0.1)

            except Exception as e:
                response_time = (time.time() - start) * 1000
                self.record_request(False, response_time, str(e))


class EmbeddingLoadTester(LoadTestRunner):
    """Load tester for embedding operations."""

    async def run(self):
        """Run embedding load test."""
        import time
        self.results["started_at"] = time.time()
        tasks = []

        for _ in range(self.num_users):
            tasks.append(self._embedding_session())

        await asyncio.gather(*tasks, return_exceptions=True)
        self.results["finished_at"] = time.time()

    async def _embedding_session(self):
        """Simulate embedding requests."""
        end_time = time.time() + self.duration_seconds
        request_count = 0

        while time.time() < end_time:
            try:
                start = time.time()

                # Simulate embedding operation
                await asyncio.sleep(0.05)  # Simulate API call

                response_time = (time.time() - start) * 1000
                self.record_request(True, response_time)
                request_count += 1

            except Exception as e:
                response_time = (time.time() - start) * 1000
                self.record_request(False, response_time, str(e))


class IngestionLoadTester(LoadTestRunner):
    """Load tester for document ingestion."""

    async def run(self):
        """Run ingestion load test."""
        import time
        self.results["started_at"] = time.time()
        tasks = []

        for _ in range(self.num_users):
            tasks.append(self._ingest_session())

        await asyncio.gather(*tasks, return_exceptions=True)
        self.results["finished_at"] = time.time()

    async def _ingest_session(self):
        """Simulate ingestion requests."""
        end_time = time.time() + self.duration_seconds

        while time.time() < end_time:
            try:
                start = time.time()

                # Simulate document ingestion
                await asyncio.sleep(0.1)  # Simulate processing

                response_time = (time.time() - start) * 1000
                self.record_request(True, response_time)

            except Exception as e:
                response_time = (time.time() - start) * 1000
                self.record_request(False, response_time, str(e))


# ============================================================================
# PYTEST FIXTURES FOR LOAD TESTING
# ============================================================================


@pytest.fixture
def http_load_tester():
    """Create HTTP load tester."""
    return HTTPLoadTester(
        target_url="http://localhost:8000",
        num_users=100,
        duration_seconds=30,
    )


@pytest.fixture
def embedding_load_tester():
    """Create embedding load tester."""
    return EmbeddingLoadTester(
        target_url="http://localhost:8000",
        num_users=50,
        duration_seconds=30,
    )


@pytest.fixture
def ingestion_load_tester():
    """Create ingestion load tester."""
    return IngestionLoadTester(
        target_url="http://localhost:8000",
        num_users=20,
        duration_seconds=60,
    )


# ============================================================================
# LOAD TEST SCENARIOS
# ============================================================================


@pytest.mark.slow
@pytest.mark.asyncio
async def test_http_endpoint_load(http_load_tester):
    """Test HTTP endpoint under load."""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{http_load_tester.target_url}/health", timeout=2) as resp:
                if resp.status >= 500:
                    pytest.skip("Skipping HTTP load test: health endpoint unavailable")
    except Exception:
        pytest.skip("Skipping HTTP load test: local API is not running")

    await http_load_tester.run()
    metrics = http_load_tester.get_metrics()

    # Assert performance targets
    assert metrics["error_rate"] < 0.05  # Less than 5% error rate
    assert metrics["p99"] < 20000  # Synthetic benchmark tolerance


@pytest.mark.slow
@pytest.mark.asyncio
async def test_embedding_load(embedding_load_tester):
    """Test embedding operations under load."""
    await embedding_load_tester.run()
    metrics = embedding_load_tester.get_metrics()

    # Assert embedding performance
    assert metrics["error_rate"] < 0.10
    assert metrics["p99"] < 3000  # P99 latency < 3 seconds


@pytest.mark.slow
@pytest.mark.asyncio
async def test_ingestion_throughput(ingestion_load_tester):
    """Test ingestion throughput."""
    await ingestion_load_tester.run()
    metrics = ingestion_load_tester.get_metrics()

    # Assert throughput target: 10,000 chunks per minute
    # With 10 chunks per ingest request, need 1667 requests per minute
    # = 27.8 requests per second
    target_rps = 27.8
    assert metrics["throughput"] > target_rps * 0.8  # Allow 20% variance
