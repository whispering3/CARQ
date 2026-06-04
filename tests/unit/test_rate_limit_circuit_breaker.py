"""
Unit tests for rate limiting and circuit breaker.
"""

import asyncio

import pytest

from carq.core.exceptions import (
    CircuitBreakerOpenError,
    RateLimitError,
    TaskTimeoutError,
)
from carq.queue import (
    BackoffConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    ExponentialBackoff,
    RateLimitConfig,
    RateLimiter,
    RateLimitProvider,
    RetryPolicy,
)


class TestRateLimiter:
    """Test RateLimiter."""

    def test_register_provider(self):
        """Test registering a provider."""
        limiter = RateLimiter()
        config = RateLimitConfig(
            provider=RateLimitProvider.OPENAI,
            rpm=3000,
            tpm=1500000,
        )

        limiter.register_provider(config)
        assert RateLimitProvider.OPENAI in limiter._buckets

    @pytest.mark.asyncio
    async def test_acquire_with_capacity(self):
        """Test acquiring tokens with available capacity."""
        limiter = RateLimiter()
        config = RateLimitConfig(
            provider=RateLimitProvider.OPENAI,
            rpm=3000,
            burst_capacity=1000.0,
            refill_rate=100.0,
        )

        limiter.register_provider(config)

        # Should acquire immediately
        wait_time = await limiter.acquire(
            RateLimitProvider.OPENAI,
            request_tokens=10.0,
            wait=False,
        )

        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_acquire_without_capacity(self):
        """Test acquiring tokens without capacity."""
        limiter = RateLimiter()
        config = RateLimitConfig(
            provider=RateLimitProvider.OPENAI,
            burst_capacity=10.0,
            refill_rate=0.1,
        )

        limiter.register_provider(config)

        # Exhaust capacity
        await limiter.acquire(RateLimitProvider.OPENAI, request_tokens=10.0, wait=False)

        # Should fail without wait
        with pytest.raises(RateLimitError):
            await limiter.acquire(
                RateLimitProvider.OPENAI,
                request_tokens=1.0,
                wait=False,
            )


class TestCircuitBreaker:
    """Test CircuitBreaker."""

    def test_creation(self):
        """Test creating circuit breaker."""
        config = CircuitBreakerConfig(
            name="test_breaker",
            failure_threshold=5,
            timeout=60,
        )

        breaker = CircuitBreaker(config)

        assert breaker.state == CircuitState.CLOSED
        assert breaker.failure_count == 0

    @pytest.mark.asyncio
    async def test_successful_call(self):
        """Test successful call through circuit breaker."""
        config = CircuitBreakerConfig(name="test")
        breaker = CircuitBreaker(config)

        async def successful_func():
            return "success"

        result = await breaker.call(successful_func)

        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_open_after_failures(self):
        """Test circuit opens after failure threshold."""
        config = CircuitBreakerConfig(
            name="test",
            failure_threshold=3,
        )

        breaker = CircuitBreaker(config)

        async def failing_func():
            raise Exception("Service error")

        # Make failing calls
        for _i in range(3):
            with pytest.raises(Exception):
                await breaker.call(failing_func)

        # Circuit should be open
        assert breaker.state == CircuitState.OPEN

        # Next call should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(failing_func)

    @pytest.mark.asyncio
    async def test_half_open_recovery(self):
        """Test recovery from HALF_OPEN state."""
        config = CircuitBreakerConfig(
            name="test",
            failure_threshold=1,
            success_threshold=1,
            timeout=0,  # Immediate reset
        )

        breaker = CircuitBreaker(config)

        # Fail once
        async def failing_func():
            raise Exception("Error")

        with pytest.raises(Exception):
            await breaker.call(failing_func)

        assert breaker.state == CircuitState.OPEN

        # Wait timeout
        await asyncio.sleep(0.1)

        # Attempt reset
        async def success_func():
            return "ok"

        result = await breaker.call(success_func)

        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED


class TestBackoffStrategy:
    """Test backoff strategies."""

    def test_exponential_delay_calculation(self):
        """Test exponential backoff delay."""
        config = BackoffConfig(
            initial_delay=1.0,
            exponential_base=2.0,
            jitter_factor=0.0,  # No jitter for testing
        )

        backoff = ExponentialBackoff(config)

        # Delays should increase exponentially
        delay_0 = backoff.get_delay(0)
        delay_1 = backoff.get_delay(1)
        delay_2 = backoff.get_delay(2)

        assert delay_0 == 1.0  # 1 * (2^0) = 1
        assert delay_1 == 2.0  # 1 * (2^1) = 2
        assert delay_2 == 4.0  # 1 * (2^2) = 4

    def test_max_delay_cap(self):
        """Test max delay is capped."""
        config = BackoffConfig(
            initial_delay=1.0,
            max_delay=10.0,
            exponential_base=2.0,
            jitter_factor=0.0,
        )

        backoff = ExponentialBackoff(config)

        delay_10 = backoff.get_delay(10)

        assert delay_10 <= 10.0  # Capped at max_delay

    @pytest.mark.asyncio
    async def test_retry_policy(self):
        """Test retry policy with backoff."""
        config = BackoffConfig(
            initial_delay=0.01,  # Small delay for testing
            jitter_factor=0.0,
        )

        backoff = ExponentialBackoff(config)
        policy = RetryPolicy(max_attempts=3, backoff=backoff)

        call_count = 0

        async def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TaskTimeoutError("id", 1)
            return "success"

        result = await policy.execute_with_retry(flaky_func)

        assert result == "success"
        assert call_count == 3  # Called 3 times

    @pytest.mark.asyncio
    async def test_retry_policy_max_attempts(self):
        """Test retry policy gives up after max attempts."""
        config = BackoffConfig(
            initial_delay=0.01,
            jitter_factor=0.0,
        )

        backoff = ExponentialBackoff(config)
        policy = RetryPolicy(max_attempts=2, backoff=backoff)

        call_count = 0

        async def always_failing():
            nonlocal call_count
            call_count += 1
            raise TaskTimeoutError("id", 1)

        with pytest.raises(TaskTimeoutError):
            await policy.execute_with_retry(always_failing)

        assert call_count == 2  # Only tried twice


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for rate limiting + circuit breaker."""

    async def test_rate_limit_and_circuit_breaker_combined(self):
        """Test rate limiter with circuit breaker."""
        # Setup rate limiter
        limiter = RateLimiter()
        limiter.register_provider(
            RateLimitConfig(
                provider=RateLimitProvider.OPENAI,
                burst_capacity=100.0,
                refill_rate=10.0,
            )
        )

        # Setup circuit breaker
        breaker_config = CircuitBreakerConfig(
            name="openai",
            failure_threshold=3,
        )
        breaker = CircuitBreaker(breaker_config)

        # Simulate rate limited + failing calls
        async def flaky_api_call():
            # Rate limit check
            await limiter.acquire(
                RateLimitProvider.OPENAI,
                request_tokens=1.0,
                wait=False,
            )

            # API call
            raise Exception("Service unavailable")

        # First few calls should fail
        for _ in range(3):
            with pytest.raises(Exception):
                await breaker.call(flaky_api_call)

        # Circuit should be open
        assert breaker.state == CircuitState.OPEN

        # Subsequent calls should raise CircuitBreakerOpenError
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(flaky_api_call)
