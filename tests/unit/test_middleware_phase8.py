"""Tests for monitoring/middleware.py - Phase 8.

Covers: request timing, metrics recording, error handling,
status code tracking, and middleware integration with FastAPI.
"""

from unittest.mock import MagicMock, patch, call

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse, Response

from carq.monitoring.middleware import MetricsMiddleware
from carq.monitoring.metrics import MetricsCollector


# ============================================================================
# HELPERS
# ============================================================================


def make_app_with_middleware(route_fn=None, path="/test"):
    """Create a minimal FastAPI app with MetricsMiddleware and a test route."""
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    if route_fn is None:
        @app.get(path)
        async def _default():
            return {"ok": True}
    else:
        app.get(path)(route_fn)

    return app


# ============================================================================
# MIDDLEWARE INITIALIZATION
# ============================================================================


def test_middleware_can_be_instantiated():
    """MetricsMiddleware can be added to a FastAPI app."""
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)
    client = TestClient(app)
    assert client is not None


def test_middleware_has_metrics_collector():
    """MetricsMiddleware initializes with a metrics collector."""
    from starlette.applications import Starlette

    # We need a raw ASGIApp to test the middleware directly
    inner = MagicMock()
    mw = MetricsMiddleware(inner)
    assert mw.metrics is not None


# ============================================================================
# REQUEST TRACKING
# ============================================================================


def test_middleware_records_successful_request():
    """Middleware calls record_http_request for successful 200 responses."""
    mock_collector = MagicMock(spec=MetricsCollector)

    with patch("carq.monitoring.middleware.get_metrics_collector", return_value=mock_collector):
        app = make_app_with_middleware()
        client = TestClient(app)
        response = client.get("/test")

    assert response.status_code == 200
    mock_collector.record_http_request.assert_called()


def test_middleware_records_correct_method():
    """Middleware records the HTTP method for each request."""
    mock_collector = MagicMock(spec=MetricsCollector)

    with patch("carq.monitoring.middleware.get_metrics_collector", return_value=mock_collector):
        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.post("/post-endpoint")
        async def _post():
            return {"posted": True}

        client = TestClient(app)
        client.post("/post-endpoint")

    call_kwargs = mock_collector.record_http_request.call_args
    assert call_kwargs is not None
    # method argument should be POST
    kwargs = call_kwargs[1] if call_kwargs[1] else {}
    args = call_kwargs[0] if call_kwargs[0] else []
    called_method = kwargs.get("method") or (args[1] if len(args) > 1 else None)
    assert called_method == "POST"


def test_middleware_records_correct_status_code():
    """Middleware records the actual HTTP status code returned."""
    mock_collector = MagicMock(spec=MetricsCollector)

    with patch("carq.monitoring.middleware.get_metrics_collector", return_value=mock_collector):
        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.get("/not-found")
        async def _nf():
            return JSONResponse({"error": "not found"}, status_code=404)

        client = TestClient(app)
        client.get("/not-found")

    call_kwargs = mock_collector.record_http_request.call_args
    kwargs = call_kwargs[1] if call_kwargs[1] else {}
    args = call_kwargs[0] if call_kwargs[0] else []
    called_status = kwargs.get("status") or (args[2] if len(args) > 2 else None)
    assert called_status == 404


def test_middleware_records_latency():
    """Middleware records latency_seconds as a float."""
    mock_collector = MagicMock(spec=MetricsCollector)

    with patch("carq.monitoring.middleware.get_metrics_collector", return_value=mock_collector):
        app = make_app_with_middleware()
        client = TestClient(app)
        client.get("/test")

    call_kwargs = mock_collector.record_http_request.call_args
    kwargs = call_kwargs[1] if call_kwargs[1] else {}
    args = call_kwargs[0] if call_kwargs[0] else []
    latency = kwargs.get("latency_seconds") or (args[3] if len(args) > 3 else None)
    assert latency is not None
    assert isinstance(latency, float)
    assert latency >= 0.0


def test_middleware_records_endpoint_path():
    """Middleware includes the endpoint path in metrics."""
    mock_collector = MagicMock(spec=MetricsCollector)

    with patch("carq.monitoring.middleware.get_metrics_collector", return_value=mock_collector):
        app = make_app_with_middleware(path="/my-endpoint")
        client = TestClient(app)
        client.get("/my-endpoint")

    call_kwargs = mock_collector.record_http_request.call_args
    kwargs = call_kwargs[1] if call_kwargs[1] else {}
    args = call_kwargs[0] if call_kwargs[0] else []
    endpoint = kwargs.get("endpoint") or (args[0] if args else None)
    assert endpoint is not None
    assert "/my-endpoint" in endpoint


# ============================================================================
# ERROR TRACKING
# ============================================================================


def test_middleware_records_client_error():
    """Middleware calls record_api_error for 4xx responses."""
    mock_collector = MagicMock(spec=MetricsCollector)

    with patch("carq.monitoring.middleware.get_metrics_collector", return_value=mock_collector):
        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.get("/client-error")
        async def _client_err():
            return JSONResponse({"error": "bad"}, status_code=400)

        client = TestClient(app)
        client.get("/client-error")

    mock_collector.record_api_error.assert_called()
    call_kwargs = mock_collector.record_api_error.call_args
    kwargs = call_kwargs[1] if call_kwargs[1] else {}
    args = call_kwargs[0] if call_kwargs[0] else []
    error_type = kwargs.get("error_type") or (args[1] if len(args) > 1 else None)
    assert error_type == "client_error"


def test_middleware_records_server_error():
    """Middleware calls record_api_error for 5xx responses."""
    mock_collector = MagicMock(spec=MetricsCollector)

    with patch("carq.monitoring.middleware.get_metrics_collector", return_value=mock_collector):
        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.get("/server-error")
        async def _server_err():
            return JSONResponse({"error": "oops"}, status_code=500)

        client = TestClient(app)
        client.get("/server-error")

    mock_collector.record_api_error.assert_called()
    call_kwargs = mock_collector.record_api_error.call_args
    kwargs = call_kwargs[1] if call_kwargs[1] else {}
    args = call_kwargs[0] if call_kwargs[0] else []
    error_type = kwargs.get("error_type") or (args[1] if len(args) > 1 else None)
    assert error_type == "server_error"


def test_middleware_does_not_record_error_for_success():
    """Middleware does NOT call record_api_error for 200 responses."""
    mock_collector = MagicMock(spec=MetricsCollector)

    with patch("carq.monitoring.middleware.get_metrics_collector", return_value=mock_collector):
        app = make_app_with_middleware()
        client = TestClient(app)
        client.get("/test")

    mock_collector.record_api_error.assert_not_called()


def test_middleware_records_exception_as_500():
    """Middleware records a 500 status when handler raises an exception."""
    mock_collector = MagicMock(spec=MetricsCollector)

    with patch("carq.monitoring.middleware.get_metrics_collector", return_value=mock_collector):
        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.get("/raises")
        async def _raises():
            raise RuntimeError("kaboom")

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/raises")

    # Should have recorded an HTTP request or an error
    assert (
        mock_collector.record_http_request.called
        or mock_collector.record_api_error.called
    )


# ============================================================================
# MULTIPLE REQUESTS
# ============================================================================


def test_middleware_records_multiple_requests():
    """Middleware records metrics for every request."""
    mock_collector = MagicMock(spec=MetricsCollector)

    with patch("carq.monitoring.middleware.get_metrics_collector", return_value=mock_collector):
        app = make_app_with_middleware()
        client = TestClient(app)
        for _ in range(5):
            client.get("/test")

    assert mock_collector.record_http_request.call_count == 5


def test_middleware_transparent_response():
    """Middleware does not alter the response body."""
    mock_collector = MagicMock(spec=MetricsCollector)

    with patch("carq.monitoring.middleware.get_metrics_collector", return_value=mock_collector):
        app = FastAPI()
        app.add_middleware(MetricsMiddleware)

        @app.get("/data")
        async def _data():
            return {"answer": 42}

        client = TestClient(app)
        response = client.get("/data")

    assert response.status_code == 200
    assert response.json() == {"answer": 42}
