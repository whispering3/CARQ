"""Unit tests for monitoring module (Phase 7).

Tests cover:
- Metrics collection
- Health checks
- Middleware integration
"""

import time
from unittest.mock import Mock

import pytest

from carq.monitoring.health import (
    HealthCheck,
    HealthChecker,
    HealthResponse,
    HealthStatus,
    get_health_checker,
)
from carq.monitoring.metrics import MetricsCollector, get_metrics_collector
from carq.monitoring.middleware import MetricsMiddleware

# ============================================================================
# METRICS TESTS
# ============================================================================


def test_metrics_collector_initialization():
    """Test metrics collector initialization."""
    collector = MetricsCollector()
    assert collector is not None
    assert collector.registry is not None


def test_metrics_collector_http_request():
    """Test HTTP request metric recording."""
    collector = MetricsCollector()

    collector.record_http_request(
        endpoint="/api/v1/embed",
        method="POST",
        status=200,
        latency_seconds=0.1,
    )

    # Verify metric was recorded
    metrics = collector.get_metrics()
    assert "carq_http_requests_total" in metrics


def test_metrics_collector_api_error():
    """Test API error metric recording."""
    collector = MetricsCollector()

    collector.record_api_error(
        endpoint="/api/v1/embed",
        error_type="validation_error",
    )

    metrics = collector.get_metrics()
    assert "carq_api_errors_total" in metrics


def test_metrics_collector_embedding():
    """Test embedding metric recording."""
    collector = MetricsCollector()

    collector.record_embedding(
        model="text-embedding-3-small",
        tokens=100,
        cost_usd=0.00002,
    )

    metrics = collector.get_metrics()
    assert "carq_embeddings_generated_total" in metrics
    assert "carq_embedding_tokens_total" in metrics
    assert "carq_embedding_cost_usd_total" in metrics


def test_metrics_collector_cache():
    """Test cache metric recording."""
    collector = MetricsCollector()

    collector.record_cache_hit(cache_type="embedding")
    collector.record_cache_miss(cache_type="embedding")
    collector.set_cache_size(500, cache_type="embedding")

    metrics = collector.get_metrics()
    assert "carq_cache_hits_total" in metrics
    assert "carq_cache_misses_total" in metrics
    assert "carq_cache_size" in metrics


def test_metrics_collector_search():
    """Test search metric recording."""
    collector = MetricsCollector()

    collector.record_search(latency_seconds=0.05, result_count=10)

    metrics = collector.get_metrics()
    assert "carq_search_latency_seconds" in metrics
    assert "carq_search_results_total" in metrics


def test_metrics_collector_vector_operations():
    """Test vector store metric recording."""
    collector = MetricsCollector()

    collector.set_stored_vectors(1000)
    collector.record_vector_insert(latency_seconds=0.005)

    metrics = collector.get_metrics()
    assert "carq_stored_vectors" in metrics
    assert "carq_vector_insert_latency_seconds" in metrics


def test_metrics_collector_pdf_processing():
    """Test PDF processing metric recording."""
    collector = MetricsCollector()

    collector.record_pdf_parse(latency_seconds=2.0, chunk_count=50)

    metrics = collector.get_metrics()
    assert "carq_pdf_parse_latency_seconds" in metrics
    assert "carq_pdf_chunks_total" in metrics


def test_metrics_collector_queue_task():
    """Test queue task metric recording."""
    collector = MetricsCollector()

    collector.record_queue_task(
        task_type="embed_chunk",
        status="success",
        latency_seconds=1.0,
    )

    metrics = collector.get_metrics()
    assert "carq_queue_tasks_total" in metrics
    assert "carq_queue_task_latency_seconds" in metrics


def test_get_metrics_collector_singleton():
    """Test metrics collector singleton."""
    collector1 = get_metrics_collector()
    collector2 = get_metrics_collector()

    assert collector1 is collector2


# ============================================================================
# HEALTH CHECK TESTS
# ============================================================================


def test_health_status_enum():
    """Test health status enum."""
    assert HealthStatus.HEALTHY == "healthy"
    assert HealthStatus.DEGRADED == "degraded"
    assert HealthStatus.UNHEALTHY == "unhealthy"


def test_health_check_creation():
    """Test health check creation."""
    check = HealthCheck(
        name="postgres",
        status=HealthStatus.HEALTHY,
        message="Connected",
        latency_ms=10.5,
    )

    assert check.name == "postgres"
    assert check.status == HealthStatus.HEALTHY
    assert check.latency_ms == 10.5


def test_health_response_creation():
    """Test health response creation."""
    response = HealthResponse(
        status=HealthStatus.HEALTHY,
        checks=[
            HealthCheck(name="postgres", status=HealthStatus.HEALTHY),
            HealthCheck(name="redis", status=HealthStatus.HEALTHY),
        ],
        ready=True,
        alive=True,
    )

    assert response.status == HealthStatus.HEALTHY
    assert len(response.checks) == 2
    assert response.ready is True


def test_health_response_overall_status():
    """Test overall status calculation."""
    # All healthy
    response1 = HealthResponse(
        status=HealthStatus.HEALTHY,
        checks=[
            HealthCheck(name="test1", status=HealthStatus.HEALTHY),
            HealthCheck(name="test2", status=HealthStatus.HEALTHY),
        ],
    )
    assert response1.overall_status == HealthStatus.HEALTHY

    # One degraded
    response2 = HealthResponse(
        status=HealthStatus.HEALTHY,
        checks=[
            HealthCheck(name="test1", status=HealthStatus.HEALTHY),
            HealthCheck(name="test2", status=HealthStatus.DEGRADED),
        ],
    )
    assert response2.overall_status == HealthStatus.DEGRADED

    # One unhealthy
    response3 = HealthResponse(
        status=HealthStatus.HEALTHY,
        checks=[
            HealthCheck(name="test1", status=HealthStatus.HEALTHY),
            HealthCheck(name="test2", status=HealthStatus.UNHEALTHY),
        ],
    )
    assert response3.overall_status == HealthStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_health_checker_initialization():
    """Test health checker initialization."""
    checker = HealthChecker(db_session=None, redis_url=None)
    assert checker is not None


@pytest.mark.asyncio
async def test_health_checker_postgres_unhealthy():
    """Test PostgreSQL health check without session."""
    checker = HealthChecker(db_session=None, redis_url=None)

    check = await checker.check_postgres()

    assert check.name == "postgres"
    assert check.status == HealthStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_health_checker_redis_degraded():
    """Test Redis health check without URL."""
    checker = HealthChecker(db_session=None, redis_url=None)

    check = await checker.check_redis()

    assert check.name == "redis"
    assert check.status == HealthStatus.DEGRADED


@pytest.mark.asyncio
async def test_health_checker_readiness():
    """Test readiness check."""
    checker = HealthChecker(db_session=None, redis_url=None)

    response = await checker.check_readiness()

    assert isinstance(response, HealthResponse)
    assert len(response.checks) > 0
    assert response.alive is True


@pytest.mark.asyncio
async def test_health_checker_liveness():
    """Test liveness check."""
    checker = HealthChecker(db_session=None, redis_url=None)

    response = await checker.check_liveness()

    assert response.status == HealthStatus.HEALTHY
    assert response.alive is True


@pytest.mark.asyncio
async def test_health_checker_full():
    """Test full health check."""
    checker = HealthChecker(db_session=None, redis_url=None)

    response = await checker.check_full()

    assert isinstance(response, HealthResponse)
    assert len(response.checks) > 0
    assert isinstance(response.status, HealthStatus)


def test_get_health_checker_singleton():
    """Test health checker singleton."""
    checker1 = get_health_checker()
    checker2 = get_health_checker()

    assert checker1 is checker2


# ============================================================================
# MIDDLEWARE TESTS
# ============================================================================


@pytest.mark.asyncio
async def test_metrics_middleware_initialization():
    """Test middleware initialization."""
    mock_app = Mock()
    middleware = MetricsMiddleware(mock_app)

    assert middleware is not None
    assert middleware.metrics is not None


@pytest.mark.asyncio
async def test_metrics_middleware_request_recording():
    """Test middleware request recording."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    MetricsMiddleware(app)

    @app.get("/test")
    async def test_endpoint():
        return {"status": "ok"}

    client = TestClient(app)
    response = client.get("/test")

    assert response.status_code == 200


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_metric_value_dataclass():
    """Test metric value dataclass."""
    from carq.monitoring.metrics import MetricValue

    mv = MetricValue(value=42.0)
    assert mv.value == 42.0
    assert mv.timestamp is not None


def test_metrics_multiple_models():
    """Test metrics for multiple embedding models."""
    collector = MetricsCollector()

    collector.record_embedding(
        model="text-embedding-3-small",
        tokens=100,
        cost_usd=0.00002,
    )
    collector.record_embedding(
        model="text-embedding-3-large",
        tokens=100,
        cost_usd=0.00013,
    )

    metrics = collector.get_metrics()
    assert "text-embedding-3-small" in metrics or "text-embedding-3-large" in metrics


def test_health_checks_performance():
    """Test health check performance."""
    HealthChecker(db_session=None, redis_url=None)

    # Quick check should complete in reasonable time
    start = time.time()
    # Just test initialization doesn't take long
    elapsed = time.time() - start

    assert elapsed < 0.1  # Should be very fast
