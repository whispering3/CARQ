"""Módulo de gerenciamento de filas - Orquestração de tarefas do CARQ."""

from carq.queue.backoff_strategy import (
    BackoffConfig,
    BackoffStrategy,
    ExponentialBackoff,
    FixedBackoff,
    LinearBackoff,
    RetryPolicy,
    retry_with_backoff,
)
from carq.queue.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerManager,
    CircuitState,
    create_breaker,
    get_circuit_breaker_manager,
)
from carq.queue.queue_manager import QueueManager
from carq.queue.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimitProvider,
    TokenBucket,
    get_rate_limiter,
    rate_limit,
)

__all__ = [
    "QueueManager",
    "RateLimiter",
    "RateLimitProvider",
    "RateLimitConfig",
    "TokenBucket",
    "get_rate_limiter",
    "rate_limit",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerManager",
    "CircuitState",
    "get_circuit_breaker_manager",
    "create_breaker",
    "BackoffStrategy",
    "BackoffConfig",
    "ExponentialBackoff",
    "LinearBackoff",
    "FixedBackoff",
    "RetryPolicy",
    "retry_with_backoff",
]
