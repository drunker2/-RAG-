#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retry mechanism for API calls with exponential backoff.
Provides resilience for network operations and API rate limits.
"""

import time
import random
import threading
from typing import Callable, Any, Optional, List, Type, Tuple
from functools import wraps
from dataclasses import dataclass
from enum import Enum


class RetryState(Enum):
    """State of retry operation."""
    SUCCESS = "success"
    FAILED = "failed"
    EXHAUSTED = "exhausted"


@dataclass
class RetryResult:
    """Result of a retry operation."""
    state: RetryState
    result: Any = None
    error: Optional[Exception] = None
    attempts: int = 0
    total_wait_time: float = 0.0


class RetryPolicy:
    """
    Retry policy with exponential backoff and jitter.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        jitter: bool = True,
        retryable_exceptions: Optional[List[Type[Exception]]] = None
    ):
        """
        Initialize retry policy.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Initial delay in seconds
            max_delay: Maximum delay cap in seconds
            exponential_base: Base for exponential backoff
            jitter: Add random jitter to prevent thundering herd
            retryable_exceptions: List of exception types to retry on
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.jitter = jitter
        self.retryable_exceptions = retryable_exceptions or [
            ConnectionError,
            TimeoutError,
        ]

    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay for given attempt number.

        Args:
            attempt: Current attempt number (0-indexed)

        Returns:
            Delay in seconds
        """
        delay = self.base_delay * (self.exponential_base ** attempt)
        delay = min(delay, self.max_delay)

        if self.jitter:
            # Add random jitter (0.5x to 1.5x)
            delay = delay * (0.5 + random.random())

        return delay

    def should_retry(self, exception: Exception) -> bool:
        """
        Check if exception is retryable.

        Args:
            exception: Exception to check

        Returns:
            True if should retry
        """
        for retryable_type in self.retryable_exceptions:
            if isinstance(exception, retryable_type):
                return True

        # Check for common retryable error messages
        error_str = str(exception).lower()
        retryable_keywords = [
            'rate limit',
            'timeout',
            'connection',
            'network',
            'too many requests',
            'service unavailable',
            'internal server error',
        ]

        return any(keyword in error_str for keyword in retryable_keywords)


class RetryExecutor:
    """
    Executor for retry operations.
    """

    def __init__(self, policy: Optional[RetryPolicy] = None):
        """
        Initialize retry executor.

        Args:
            policy: Retry policy to use
        """
        self.policy = policy or RetryPolicy()
        self._lock = threading.Lock()

    def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> RetryResult:
        """
        Execute function with retry logic.

        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            RetryResult with outcome
        """
        attempts = 0
        total_wait = 0.0
        last_error = None

        while attempts <= self.policy.max_retries:
            try:
                result = func(*args, **kwargs)
                return RetryResult(
                    state=RetryState.SUCCESS,
                    result=result,
                    attempts=attempts + 1,
                    total_wait_time=total_wait
                )

            except Exception as e:
                last_error = e
                attempts += 1

                # Check if we should retry
                if attempts > self.policy.max_retries:
                    break

                if not self.policy.should_retry(e):
                    break

                # Calculate and wait
                delay = self.policy.get_delay(attempts - 1)
                time.sleep(delay)
                total_wait += delay

        return RetryResult(
            state=RetryState.FAILED,
            error=last_error,
            attempts=attempts,
            total_wait_time=total_wait
        )

    async def execute_async(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> RetryResult:
        """
        Execute async function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            RetryResult with outcome
        """
        import asyncio

        attempts = 0
        total_wait = 0.0
        last_error = None

        while attempts <= self.policy.max_retries:
            try:
                result = await func(*args, **kwargs)
                return RetryResult(
                    state=RetryState.SUCCESS,
                    result=result,
                    attempts=attempts + 1,
                    total_wait_time=total_wait
                )

            except Exception as e:
                last_error = e
                attempts += 1

                if attempts > self.policy.max_retries:
                    break

                if not self.policy.should_retry(e):
                    break

                delay = self.policy.get_delay(attempts - 1)
                await asyncio.sleep(delay)
                total_wait += delay

        return RetryResult(
            state=RetryState.FAILED,
            error=last_error,
            attempts=attempts,
            total_wait_time=total_wait
        )


# ============================================
# Decorator for retry
# ============================================

def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: Optional[List[Type[Exception]]] = None
):
    """
    Decorator to add retry logic to functions.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay cap
        retryable_exceptions: Exceptions to retry on

    Example:
        @with_retry(max_retries=3, base_delay=1.0)
        def call_api():
            return requests.get("https://api.example.com")
    """
    policy = RetryPolicy(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        retryable_exceptions=retryable_exceptions
    )
    executor = RetryExecutor(policy)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = executor.execute(func, *args, **kwargs)
            if result.state == RetryState.SUCCESS:
                return result.result
            raise result.error

        return wrapper

    return decorator


# ============================================
# Circuit Breaker Pattern
# ============================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if recovered


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    Prevents cascading failures by failing fast when service is down.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_requests: int = 3
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time in seconds before attempting recovery
            half_open_requests: Number of test requests in half-open state
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_requests = half_open_requests

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        with self._lock:
            self._check_recovery()
            return self._state

    def _check_recovery(self):
        """Check if we should transition from OPEN to HALF_OPEN."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._success_count = 0

    def can_execute(self) -> bool:
        """
        Check if request can be executed.

        Returns:
            True if request should be allowed
        """
        with self._lock:
            self._check_recovery()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                return self._success_count < self.half_open_requests

            return False

    def record_success(self):
        """Record a successful request."""
        with self._lock:
            self._failure_count = 0

            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_requests:
                    self._state = CircuitState.CLOSED

    def record_failure(self):
        """Record a failed request."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN

            elif self._failure_count >= self.failure_threshold:
                self._state = CircuitState.OPEN

    def reset(self):
        """Reset the circuit breaker."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time
            }


class CircuitBreakerError(Exception):
    """Exception raised when circuit is open."""
    pass


def with_circuit_breaker(circuit: CircuitBreaker):
    """
    Decorator to add circuit breaker protection.

    Args:
        circuit: CircuitBreaker instance

    Example:
        circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

        @with_circuit_breaker(circuit)
        def call_external_api():
            return requests.get("https://api.example.com")
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not circuit.can_execute():
                raise CircuitBreakerError(
                    f"Circuit breaker is {circuit.state.value}. "
                    f"Service unavailable."
                )

            try:
                result = func(*args, **kwargs)
                circuit.record_success()
                return result

            except Exception as e:
                circuit.record_failure()
                raise

        return wrapper

    return decorator


# ============================================
# Rate Limiter
# ============================================

class RateLimiter:
    """
    Token bucket rate limiter.
    Controls the rate of requests to prevent overwhelming services.
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst_size: Optional[int] = None
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second
            burst_size: Maximum burst size (defaults to 2x rate)
        """
        self.rate = requests_per_second
        self.burst_size = burst_size or int(requests_per_second * 2)
        self._tokens = float(self.burst_size)
        self._last_update = time.time()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        """
        Try to acquire tokens.

        Args:
            tokens: Number of tokens to acquire

        Returns:
            True if acquired, False if rate limited
        """
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update

            # Add tokens based on elapsed time
            self._tokens = min(
                self.burst_size,
                self._tokens + elapsed * self.rate
            )
            self._last_update = now

            if self._tokens >= tokens:
                self._tokens -= tokens
                return True

            return False

    def wait_and_acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Wait until tokens are available and acquire.

        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait (None = forever)

        Returns:
            True if acquired, False if timeout
        """
        start_time = time.time()

        while True:
            if self.acquire(tokens):
                return True

            if timeout:
                if time.time() - start_time >= timeout:
                    return False

            # Wait a bit before retrying
            wait_time = (tokens - self._tokens) / self.rate
            time.sleep(min(wait_time, 0.1))


def rate_limit(requests_per_second: float, burst_size: Optional[int] = None):
    """
    Decorator to rate limit function calls.

    Args:
        requests_per_second: Maximum calls per second
        burst_size: Maximum burst size

    Example:
        @rate_limit(requests_per_second=10)
        def call_api():
            return requests.get("https://api.example.com")
    """
    limiter = RateLimiter(requests_per_second, burst_size)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter.wait_and_acquire()
            return func(*args, **kwargs)

        return wrapper

    return decorator


# ============================================
# Global instances for common use cases
# ============================================

# Default retry executor
default_retry_executor = RetryExecutor()

# Circuit breaker for LLM API calls
llm_circuit_breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60.0
)

# Rate limiter for API calls (OpenAI default: 60 req/min)
api_rate_limiter = RateLimiter(
    requests_per_second=1.0,
    burst_size=5
)


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("Testing Retry Module...")
    print("=" * 60)

    # Test retry with simulated failures
    call_count = 0

    @with_retry(max_retries=3, base_delay=0.5)
    def unreliable_function():
        global call_count
        call_count += 1
        print(f"  Attempt {call_count}...")

        if call_count < 3:
            raise ConnectionError("Simulated failure")

        return "Success!"

    print("\n1. Testing retry decorator:")
    try:
        result = unreliable_function()
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed: {e}")

    # Test circuit breaker
    print("\n2. Testing circuit breaker:")
    circuit = CircuitBreaker(failure_threshold=2, recovery_timeout=2)

    @with_circuit_breaker(circuit)
    def protected_call():
        return "OK"

    # Normal operation
    print(f"  State: {circuit.state.value}")
    print(f"  Result: {protected_call()}")

    # Simulate failures
    circuit.record_failure()
    circuit.record_failure()
    print(f"  After 2 failures: {circuit.state.value}")

    # Try to call
    try:
        protected_call()
    except CircuitBreakerError as e:
        print(f"  Expected error: {e}")

    # Test rate limiter
    print("\n3. Testing rate limiter:")
    limiter = RateLimiter(requests_per_second=5, burst_size=3)

    for i in range(5):
        acquired = limiter.acquire()
        print(f"  Request {i+1}: {'Acquired' if acquired else 'Rate limited'}")

    print("\nRetry module test completed!")
