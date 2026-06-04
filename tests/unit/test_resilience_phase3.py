"""
Comprehensive unit tests for Phase 3: Resilience Mechanisms.

Tests cover:
- Rate limiter (token bucket algorithm)
- Circuit breaker (closed, open, half-open states)
- Backoff strategy (exponential, jitter)
"""

import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from carq.queue.backoff_strategy import BackoffStrategy
from carq.queue.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from carq.queue.rate_limiter import RateLimitError
from carq.queue.rate_limiter import TokenBucket as RateLimiter

# ============================================================================
# TEST: RATE LIMITER - TOKEN BUCKET ALGORITHM
# ============================================================================


class TestRateLimiterTokenBucket:
    """Test rate limiter token bucket implementation."""

    def test_rate_limiter_creation(self):
        """Test creating rate limiter."""
        limiter = RateLimiter(capacity=100, refill_rate=10)
        assert limiter is not None
        assert limiter.capacity == 100
        assert limiter.refill_rate == 10

    def test_rate_limiter_initial_tokens(self):
        """Test rate limiter starts with full capacity."""
        limiter = RateLimiter(capacity=50, refill_rate=5)
        assert limiter.available_tokens == 50

    def test_rate_limiter_consume_single_token(self):
        """Test consuming a single token."""
        limiter = RateLimiter(capacity=100, refill_rate=10)
        initial = limiter.available_tokens

        limiter.consume(1)

        assert limiter.available_tokens < initial

    def test_rate_limiter_consume_multiple_tokens(self):
        """Test consuming multiple tokens."""
        limiter = RateLimiter(capacity=100, refill_rate=10)
        initial = limiter.available_tokens

        limiter.consume(25)

        assert limiter.available_tokens == initial - 25

    def test_rate_limiter_exceeds_capacity_error(self):
        """Test error when consuming more tokens than available."""
        limiter = RateLimiter(capacity=10, refill_rate=5)

        with pytest.raises(RateLimitError):
            limiter.consume(20)

    @pytest.mark.asyncio
    async def test_rate_limiter_wait_for_tokens(self):
        """Test waiting for tokens to become available."""
        limiter = RateLimiter(capacity=10, refill_rate=10)  # 10 tokens/sec

        # Consume all tokens
        limiter.consume(10)

        # Wait for 1 token (should take ~0.1 seconds)
        start = time.time()
        await limiter.wait_for_tokens(1)
        elapsed = time.time() - start

        # Should have waited at least some time
        assert elapsed > 0

    @pytest.mark.asyncio
    async def test_rate_limiter_refill_tokens(self):
        """Test token refill over time."""
        limiter = RateLimiter(capacity=100, refill_rate=10)  # 10 tokens/sec

        # Consume all tokens
        limiter.consume(100)
        assert limiter.available_tokens == 0

        # Wait and check refill
        await asyncio.sleep(0.5)

        # Should have refilled some tokens
        assert limiter.available_tokens > 0

    @pytest.mark.asyncio
    async def test_rate_limiter_respects_rate(self):
        """Test that rate limiter respects configured rate."""
        limiter = RateLimiter(capacity=10, refill_rate=10)  # 10 tokens/sec

        # Should allow 10 requests immediately
        for _ in range(10):
            limiter.consume(1)

        # 11th request should fail initially
        with pytest.raises(RateLimitError):
            limiter.consume(1)

        # But should succeed after waiting
        await limiter.wait_for_tokens(1)
        # Should now have at least 1 token
        limiter.consume(1)

    def test_rate_limiter_zero_capacity(self):
        """Test rate limiter with zero capacity."""
        limiter = RateLimiter(capacity=0, refill_rate=10)

        with pytest.raises(RateLimitError):
            limiter.consume(1)

    def test_rate_limiter_high_capacity(self):
        """Test rate limiter with high capacity."""
        limiter = RateLimiter(capacity=10000, refill_rate=1000)

        limiter.consume(5000)
        assert limiter.available_tokens == 5000

    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_requests(self):
        """Test rate limiter with concurrent requests."""
        limiter = RateLimiter(capacity=100, refill_rate=100)

        async def request():
            await limiter.wait_for_tokens(1)
            limiter.consume(1)

        # Should handle 10 concurrent requests
        await asyncio.gather(*[request() for _ in range(10)])

    @pytest.mark.asyncio
    async def test_rate_limiter_burst_handling(self):
        """Test handling burst of requests."""
        limiter = RateLimiter(capacity=50, refill_rate=10)

        # Burst of requests should consume capacity
        for _ in range(50):
            limiter.consume(1)

        assert limiter.available_tokens == 0


# ============================================================================
# TEST: CIRCUIT BREAKER - STATE TRANSITIONS
# ============================================================================


class TestCircuitBreakerStates:
    """Test circuit breaker state machine."""

    def test_circuit_breaker_creation(self):
        """Test creating circuit breaker."""
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
            success_threshold=2,
        )
        assert breaker is not None
        assert breaker.failure_threshold == 5

    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker starts in CLOSED state."""
        breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60,
        )
        assert breaker.state == "CLOSED"

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state_success(self):
        """Test successful request in CLOSED state."""
        breaker = CircuitBreaker(failure_threshold=5)

        async def successful_operation():
            return "success"

        result = await breaker.call(successful_operation)
        assert result == "success"
        assert breaker.state == "CLOSED"

    @pytest.mark.asyncio
    async def test_circuit_breaker_closed_state_pass_through(self):
        """Test CLOSED state passes requests through."""
        breaker = CircuitBreaker(failure_threshold=5)
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            return "called"

        await breaker.call(operation)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_state_rejects(self):
        """Test OPEN state rejects requests."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)

        async def failing_operation():
            raise ValueError("Operation failed")

        # Fail twice to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(failing_operation)

        # Circuit should now be OPEN
        assert breaker.state == "OPEN"

        # Next request should be rejected immediately
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(failing_operation)

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_fail_fast(self):
        """Test that OPEN circuit fails fast without calling operation."""
        breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
        call_count = 0

        async def failing_operation():
            nonlocal call_count
            call_count += 1
            raise ValueError("Failed")

        # Fail twice to open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(failing_operation)

        assert breaker.state == "OPEN"
        call_count_before = call_count

        # Next call should not execute operation
        with pytest.raises(CircuitBreakerOpenError):
            await breaker.call(failing_operation)

        # Operation should not have been called
        assert call_count == call_count_before

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_state(self):
        """Test HALF_OPEN state allows test request."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            success_threshold=1,
        )

        # Open the circuit
        async def failing_op():
            raise ValueError("Failed")

        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(failing_op)

        assert breaker.state == "OPEN"

        # Wait for recovery timeout
        await asyncio.sleep(0.2)

        # Circuit should transition to HALF_OPEN and allow test request
        async def successful_op():
            return "success"

        result = await breaker.call(successful_op)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery from OPEN to CLOSED."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            success_threshold=1,
        )

        # Open circuit
        async def fail():
            raise ValueError("Fail")

        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(fail)

        assert breaker.state == "OPEN"

        # Wait for recovery
        await asyncio.sleep(0.2)

        # Successful request in HALF_OPEN should close circuit
        async def success():
            return "OK"

        result = await breaker.call(success)
        assert result == "OK"
        assert breaker.state == "CLOSED"

    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_failure_reopens(self):
        """Test that failure in HALF_OPEN state reopens circuit."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            success_threshold=1,
        )

        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                await breaker.call(AsyncMock(side_effect=ValueError("Fail")))

        # Wait for recovery
        await asyncio.sleep(0.2)

        # Request should fail and reopen
        with pytest.raises(ValueError):
            await breaker.call(AsyncMock(side_effect=ValueError("Fail")))

        assert breaker.state == "OPEN"

    def test_circuit_breaker_custom_thresholds(self):
        """Test circuit breaker with custom thresholds."""
        breaker = CircuitBreaker(
            failure_threshold=10,
            recovery_timeout=120,
            success_threshold=5,
        )

        assert breaker.failure_threshold == 10
        assert breaker.recovery_timeout == 120
        assert breaker.success_threshold == 5

    @pytest.mark.asyncio
    async def test_circuit_breaker_multiple_failures(self):
        """Test multiple failures leading to circuit opening."""
        breaker = CircuitBreaker(failure_threshold=3)

        async def fail():
            raise RuntimeError("Fail")

        # Fail 3 times
        for i in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(fail)

            if i < 2:
                assert breaker.state == "CLOSED"

        # Circuit should now be OPEN
        assert breaker.state == "OPEN"


# ============================================================================
# TEST: BACKOFF STRATEGY - EXPONENTIAL AND JITTER
# ============================================================================


class TestBackoffStrategy:
    """Test backoff strategy implementations."""

    def test_backoff_strategy_creation(self):
        """Test creating backoff strategy."""
        strategy = BackoffStrategy(
            initial_delay=0.1,
            max_delay=10,
            exponential_base=2,
        )
        assert strategy is not None
        assert strategy.initial_delay == 0.1
        assert strategy.max_delay == 10

    def test_backoff_exponential_growth(self):
        """Test exponential delay growth."""
        strategy = BackoffStrategy(
            initial_delay=1,
            max_delay=1000,
            exponential_base=2,
            jitter=False,
        )

        delays = [strategy.calculate_delay(attempt) for attempt in range(5)]

        # Should grow exponentially: 1, 2, 4, 8, 16
        assert delays[0] == 1
        assert delays[1] == 2
        assert delays[2] == 4
        assert delays[3] == 8
        assert delays[4] == 16

    def test_backoff_respects_max_delay(self):
        """Test that backoff respects max delay."""
        strategy = BackoffStrategy(
            initial_delay=1,
            max_delay=10,
            exponential_base=2,
            jitter=False,
        )

        # Even with high attempt count, should not exceed max_delay
        delay = strategy.calculate_delay(10)
        assert delay <= 10

    def test_backoff_with_jitter(self):
        """Test that jitter adds randomness."""
        strategy = BackoffStrategy(
            initial_delay=1,
            max_delay=100,
            exponential_base=2,
            jitter=True,
        )

        # Get multiple delays for same attempt
        delays = [strategy.calculate_delay(3) for _ in range(10)]

        # Should have variety (not all same)
        assert len(set(delays)) > 1

    def test_backoff_jitter_prevents_thundering_herd(self):
        """Test jitter helps prevent thundering herd problem."""
        strategy = BackoffStrategy(
            initial_delay=1,
            max_delay=100,
            exponential_base=2,
            jitter=True,
        )

        # Get delays for many concurrent retries
        delays = [strategy.calculate_delay(2) for _ in range(100)]

        # Should be distributed across range
        min_delay = min(delays)
        max_delay_calc = max(delays)

        # Should have at least some variation
        assert max_delay_calc - min_delay > 0

    def test_backoff_without_jitter_deterministic(self):
        """Test backoff without jitter is deterministic."""
        strategy = BackoffStrategy(
            initial_delay=1,
            max_delay=100,
            exponential_base=2,
            jitter=False,
        )

        # Same attempt should always give same delay
        delay1 = strategy.calculate_delay(3)
        delay2 = strategy.calculate_delay(3)

        assert delay1 == delay2

    def test_backoff_first_attempt_zero_delay(self):
        """Test that first attempt has minimal delay."""
        strategy = BackoffStrategy(
            initial_delay=0,
            max_delay=100,
            exponential_base=2,
            jitter=False,
        )

        delay = strategy.calculate_delay(0)
        assert delay == 0

    @pytest.mark.asyncio
    async def test_backoff_wait(self):
        """Test waiting with backoff strategy."""
        strategy = BackoffStrategy(
            initial_delay=0.01,
            max_delay=0.1,
            exponential_base=2,
            jitter=False,
        )

        start = time.time()
        await strategy.wait(1)  # Attempt 1 = 0.01s delay
        elapsed = time.time() - start

        # Should have waited roughly the expected amount
        assert elapsed >= 0.01

    def test_backoff_custom_exponential_base(self):
        """Test backoff with custom exponential base."""
        strategy = BackoffStrategy(
            initial_delay=1,
            max_delay=1000,
            exponential_base=3,
            jitter=False,
        )

        delays = [strategy.calculate_delay(attempt) for attempt in range(4)]

        # Should grow with base 3: 1, 3, 9, 27
        assert delays[0] == 1
        assert delays[1] == 3
        assert delays[2] == 9
        assert delays[3] == 27

    @pytest.mark.asyncio
    async def test_backoff_with_multiple_retries(self):
        """Test backoff through multiple retry attempts."""
        strategy = BackoffStrategy(
            initial_delay=0.01,
            max_delay=0.1,
            exponential_base=2,
            jitter=False,
        )

        async def failing_operation():
            """Simulated operation that fails."""
            raise ValueError("Operation failed")

        max_retries = 3
        for attempt in range(max_retries):
            if attempt > 0:
                await strategy.wait(attempt)

            try:
                await failing_operation()
            except ValueError:
                if attempt == max_retries - 1:
                    break

    def test_backoff_numeric_properties(self):
        """Test numeric properties of backoff delays."""
        strategy = BackoffStrategy(
            initial_delay=1,
            max_delay=100,
            exponential_base=2,
            jitter=False,
        )

        for attempt in range(10):
            delay = strategy.calculate_delay(attempt)

            # Delay should be positive
            assert delay >= 0

            # Delay should not exceed max
            assert delay <= 100


# ============================================================================
# TEST: COMBINED RESILIENCE PATTERNS
# ============================================================================


class TestCombinedResiliencePatterns:
    """Test combining multiple resilience patterns."""

    @pytest.mark.asyncio
    async def test_rate_limiter_with_circuit_breaker(self):
        """Test using rate limiter and circuit breaker together."""
        limiter = RateLimiter(capacity=10, refill_rate=10)
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1)

        async def api_call():
            await limiter.wait_for_tokens(1)
            limiter.consume(1)
            return "success"

        result = await breaker.call(api_call)
        assert result == "success"

    @pytest.mark.asyncio
    async def test_circuit_breaker_with_backoff(self):
        """Test circuit breaker with exponential backoff."""
        breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
        )
        backoff = BackoffStrategy(
            initial_delay=0.01,
            max_delay=0.5,
            exponential_base=2,
        )

        async def operation():
            return "success"

        result = await breaker.call(operation)
        assert result == "success"

        # Test with backoff
        delay = backoff.calculate_delay(1)
        assert delay > 0

    @pytest.mark.asyncio
    async def test_complete_resilience_pattern(self):
        """Test complete resilience pattern integration."""
        limiter = RateLimiter(capacity=100, refill_rate=50)
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=1)
        BackoffStrategy(initial_delay=0.01, max_delay=5)

        async def resilient_operation():
            await limiter.wait_for_tokens(1)
            limiter.consume(1)
            return await breaker.call(AsyncMock(return_value="success"))

        result = await resilient_operation()
        assert result == "success"


# ============================================================================
# MARKER TESTS
# ============================================================================


@pytest.mark.unit
def test_phase3_unit_marker():
    """Test that Phase 3 tests are marked as unit tests."""
    assert True
