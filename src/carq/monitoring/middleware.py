"""Middleware FastAPI para coleta automática de métricas de requisições."""

import logging
import time
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from carq.monitoring.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware ASGI para coleta automática de métricas de requisições."""

    def __init__(self, app: ASGIApp):
        super().__init__(app)
        self.metrics = get_metrics_collector()
        self.logger = logging.getLogger(__name__)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        endpoint = request.url.path
        method = request.method
        start_time = time.time()

        try:
            response = await call_next(request)
        except Exception as e:
            status = 500
            latency = (time.time() - start_time)

            self.metrics.record_http_request(
                endpoint=endpoint,
                method=method,
                status=status,
                latency_seconds=latency,
            )

            self.metrics.record_api_error(
                endpoint=endpoint,
                error_type=type(e).__name__,
            )

            self.logger.error(f"Request error: {e}", extra={"endpoint": endpoint})
            raise

        latency = time.time() - start_time

        self.metrics.record_http_request(
            endpoint=endpoint,
            method=method,
            status=response.status_code,
            latency_seconds=latency,
        )

        if response.status_code >= 400:
            error_type = "client_error" if response.status_code < 500 else "server_error"
            self.metrics.record_api_error(
                endpoint=endpoint,
                error_type=error_type,
            )

        return response
