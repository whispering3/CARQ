"""Coleta e rastreamento de métricas com Prometheus para requisições, embeddings e cache."""

import asyncio
import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

logger = logging.getLogger(__name__)


@dataclass
class MetricValue:
    """Valor de métrica único com timestamp."""

    value: float
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class MetricsCollector:
    """Coleta métricas Prometheus para o sistema CARQ."""

    # CORREÇÃO CRÍTICA: Whitelist de endpoints para evitar cardinalidade ilimitada
    VALID_ENDPOINTS = {"/embed", "/batch", "/search", "/health", "other"}
    VALID_ERROR_TYPES = {"validation", "timeout", "rate_limit", "internal", "unknown"}

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self.logger = logging.getLogger(__name__)

        self.http_requests_total = Counter(
            "carq_http_requests_total",
            "Total HTTP requests",
            ["endpoint", "method", "status"],
            registry=self.registry,
        )

        self.http_request_latency = Histogram(
            "carq_http_request_latency_seconds",
            "HTTP request latency in seconds",
            ["endpoint", "method"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 5.0),
            registry=self.registry,
        )

        self.api_errors_total = Counter(
            "carq_api_errors_total",
            "Total API errors",
            ["endpoint", "error_type"],
            registry=self.registry,
        )

        self.embeddings_generated_total = Counter(
            "carq_embeddings_generated_total",
            "Total embeddings generated",
            ["model"],
            registry=self.registry,
        )

        self.embedding_tokens_total = Counter(
            "carq_embedding_tokens_total",
            "Total tokens used for embeddings",
            ["model"],
            registry=self.registry,
        )

        self.embedding_cost_usd_total = Counter(
            "carq_embedding_cost_usd_total",
            "Total cost in USD for embeddings",
            ["model"],
            registry=self.registry,
        )

        self.cache_hits_total = Counter(
            "carq_cache_hits_total",
            "Total cache hits",
            ["cache_type"],
            registry=self.registry,
        )

        self.cache_misses_total = Counter(
            "carq_cache_misses_total",
            "Total cache misses",
            ["cache_type"],
            registry=self.registry,
        )

        self.cache_size = Gauge(
            "carq_cache_size",
            "Number of items in cache",
            ["cache_type"],
            registry=self.registry,
        )

        self.search_latency = Histogram(
            "carq_search_latency_seconds",
            "Vector search latency in seconds",
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0),
            registry=self.registry,
        )

        self.search_results_total = Counter(
            "carq_search_results_total",
            "Total search result items returned",
            registry=self.registry,
        )

        self.stored_vectors = Gauge(
            "carq_stored_vectors",
            "Number of vectors in store",
            registry=self.registry,
        )

        self.vector_insert_latency = Histogram(
            "carq_vector_insert_latency_seconds",
            "Vector insert latency in seconds",
            buckets=(0.001, 0.005, 0.01, 0.05, 0.1),
            registry=self.registry,
        )

        self.pdf_parse_latency = Histogram(
            "carq_pdf_parse_latency_seconds",
            "PDF parsing latency in seconds",
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0),
            registry=self.registry,
        )

        self.pdf_chunks_total = Counter(
            "carq_pdf_chunks_total",
            "Total chunks created from PDFs",
            registry=self.registry,
        )

        self.queue_tasks_total = Counter(
            "carq_queue_tasks_total",
            "Total tasks processed",
            ["task_type", "status"],
            registry=self.registry,
        )

        self.queue_task_latency = Histogram(
            "carq_queue_task_latency_seconds",
            "Task processing latency in seconds",
            ["task_type"],
            buckets=(0.1, 0.5, 1.0, 5.0, 10.0),
            registry=self.registry,
        )

    def record_http_request(
        self,
        endpoint: str,
        method: str,
        status: int,
        latency_seconds: float,
    ):
        """Registra métricas de requisição HTTP.

        CORREÇÃO CRÍTICA: Valida endpoint para evitar cardinalidade ilimitada.
        """
        endpoint = self._validate_endpoint(endpoint)

        self.http_requests_total.labels(
            endpoint=endpoint,
            method=method,
            status=status,
        ).inc()

        self.http_request_latency.labels(
            endpoint=endpoint,
            method=method,
        ).observe(latency_seconds)

    def record_api_error(
        self,
        endpoint: str,
        error_type: str,
    ):
        """Registra erro da API.

        CORREÇÃO CRÍTICA: Valida error_type para evitar cardinalidade ilimitada.
        """
        endpoint = self._validate_endpoint(endpoint)
        error_type = self._validate_error_type(error_type)
        self.api_errors_total.labels(
            endpoint=endpoint,
            error_type=error_type,
        ).inc()

    def record_embedding(
        self,
        model: str,
        tokens: int,
        cost_usd: float,
    ):
        """Registra a geração de embedding."""
        self.embeddings_generated_total.labels(model=model).inc()
        self.embedding_tokens_total.labels(model=model).inc(tokens)
        self.embedding_cost_usd_total.labels(model=model).inc(cost_usd)

    def record_cache_hit(self, cache_type: str = "embedding"):
        self.cache_hits_total.labels(cache_type=cache_type).inc()

    def record_cache_miss(self, cache_type: str = "embedding"):
        self.cache_misses_total.labels(cache_type=cache_type).inc()

    def set_cache_size(
        self,
        size: int,
        cache_type: str = "embedding",
    ):
        self.cache_size.labels(cache_type=cache_type).set(size)

    def record_search(
        self,
        latency_seconds: float,
        result_count: int,
    ):
        self.search_latency.observe(latency_seconds)
        self.search_results_total.inc(result_count)

    def set_stored_vectors(self, count: int):
        self.stored_vectors.set(count)

    def record_vector_insert(self, latency_seconds: float):
        self.vector_insert_latency.observe(latency_seconds)

    def record_pdf_parse(
        self,
        latency_seconds: float,
        chunk_count: int,
    ):
        self.pdf_parse_latency.observe(latency_seconds)
        self.pdf_chunks_total.inc(chunk_count)

    def record_queue_task(
        self,
        task_type: str,
        status: str,
        latency_seconds: float,
    ):
        self.queue_tasks_total.labels(task_type=task_type, status=status).inc()
        self.queue_task_latency.labels(task_type=task_type).observe(latency_seconds)

    @staticmethod
    def _validate_endpoint(endpoint: str) -> str:
        """Valida endpoint contra a whitelist.

        CORREÇÃO CRÍTICA: Evita cardinalidade ilimitada por endpoints arbitrários.
        """
        return endpoint if endpoint in MetricsCollector.VALID_ENDPOINTS else "other"

    @staticmethod
    def _validate_error_type(error_type: str) -> str:
        """Valida tipo de erro contra a whitelist.

        CORREÇÃO CRÍTICA: Evita cardinalidade ilimitada por tipos de erro arbitrários.
        """
        return error_type if error_type in MetricsCollector.VALID_ERROR_TYPES else "unknown"

    def get_metrics(self) -> str:
        """Retorna métricas no formato de exposição do Prometheus."""
        from prometheus_client import generate_latest

        output = generate_latest(self.registry)
        return output.decode("utf-8")


# Instância global de métricas
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Obtém ou cria o coletor de métricas global (singleton)."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def record_api_request(endpoint: str, method: str, status: int, latency_seconds: float):
    """Registra uma requisição de API nas métricas globais."""
    collector = get_metrics_collector()
    collector.record_http_request(endpoint, method, status, latency_seconds)


def timing_decorator(func: Callable) -> Callable:
    """Decorator que cronometra automaticamente a execução de funções."""

    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        try:
            return await func(*args, **kwargs)
        finally:
            latency = time.time() - start
            logger.debug(f"{func.__name__} took {latency*1000:.2f}ms")

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            latency = time.time() - start
            logger.debug(f"{func.__name__} took {latency*1000:.2f}ms")

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper
