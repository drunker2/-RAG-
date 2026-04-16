#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Metrics and health check module for RAG System.
Provides observability for production environments.
"""

import os
import time
import threading
import platform
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Optional: psutil for system metrics
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class HealthStatus(Enum):
    """Health check status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    latency_ms: Optional[float] = None


class HealthChecker:
    """
    Health check manager for RAG System.
    """

    def __init__(self):
        self._checks: Dict[str, callable] = {}
        self._lock = threading.Lock()

    def register(self, name: str, check_func: callable) -> None:
        """
        Register a health check.

        Args:
            name: Check name
            check_func: Function that returns HealthCheckResult
        """
        with self._lock:
            self._checks[name] = check_func

    def check(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        Run health checks.

        Args:
            name: Specific check to run (all if None)

        Returns:
            Health check results
        """
        results = []

        checks_to_run = {name: self._checks[name]} if name else dict(self._checks)

        for check_name, check_func in checks_to_run.items():
            try:
                start_time = time.time()
                result = check_func()
                latency = (time.time() - start_time) * 1000

                if isinstance(result, HealthCheckResult):
                    result.latency_ms = latency
                    results.append(result)
                else:
                    results.append(HealthCheckResult(
                        name=check_name,
                        status=HealthStatus.HEALTHY,
                        details=result if isinstance(result, dict) else {}
                    ))

            except Exception as e:
                results.append(HealthCheckResult(
                    name=check_name,
                    status=HealthStatus.UNHEALTHY,
                    message=str(e)
                ))

        # Determine overall status
        overall = HealthStatus.HEALTHY
        for result in results:
            if result.status == HealthStatus.UNHEALTHY:
                overall = HealthStatus.UNHEALTHY
                break
            elif result.status == HealthStatus.DEGRADED:
                overall = HealthStatus.DEGRADED

        return {
            "status": overall.value,
            "timestamp": datetime.now().isoformat(),
            "checks": [
                {
                    "name": r.name,
                    "status": r.status.value,
                    "message": r.message,
                    "details": r.details,
                    "latency_ms": r.latency_ms
                }
                for r in results
            ]
        }


# ============================================
# Metrics Collector
# ============================================

class MetricsCollector:
    """
    Collects and aggregates metrics for RAG System.
    """

    def __init__(self):
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = {}
        self._timers: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

        # System metrics
        self._start_time = time.time()

    def increment(self, name: str, value: int = 1) -> None:
        """Increment a counter."""
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        with self._lock:
            self._gauges[name] = value

    def histogram(self, name: str, value: float) -> None:
        """Record a histogram value."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = []
            self._histograms[name].append(value)

            # Keep only last 1000 values
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]

    def timer(self, name: str, duration_ms: float) -> None:
        """Record a timer value."""
        with self._lock:
            if name not in self._timers:
                self._timers[name] = []
            self._timers[name].append(duration_ms)

            # Keep only last 1000 values
            if len(self._timers[name]) > 1000:
                self._timers[name] = self._timers[name][-1000:]

    def get_counters(self) -> Dict[str, int]:
        """Get all counters."""
        with self._lock:
            return dict(self._counters)

    def get_gauges(self) -> Dict[str, float]:
        """Get all gauges."""
        with self._lock:
            return dict(self._gauges)

    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a histogram."""
        with self._lock:
            values = self._histograms.get(name, [])
            if not values:
                return {}

            sorted_values = sorted(values)
            n = len(sorted_values)

            return {
                "count": n,
                "min": sorted_values[0],
                "max": sorted_values[-1],
                "mean": sum(values) / n,
                "p50": sorted_values[n // 2],
                "p90": sorted_values[int(n * 0.9)],
                "p99": sorted_values[int(n * 0.99)]
            }

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        with self._lock:
            uptime = time.time() - self._start_time

            result = {
                "uptime_seconds": uptime,
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "python": {
                    "version": platform.python_version(),
                    "platform": platform.platform()
                }
            }

            # Add system metrics if psutil is available
            if PSUTIL_AVAILABLE:
                result["system"] = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": psutil.disk_usage('/').percent if os.name != 'nt' else psutil.disk_usage('C:\\').percent,
                }
            else:
                result["system"] = {
                    "cpu_percent": None,
                    "memory_percent": None,
                    "disk_percent": None,
                    "note": "psutil not installed"
                }

            return result

    def reset(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._timers.clear()


# ============================================
# Timer Context Manager
# ============================================

class Timer:
    """Context manager for timing operations."""

    def __init__(self, metrics: MetricsCollector, name: str):
        self.metrics = metrics
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        duration_ms = (time.time() - self.start_time) * 1000
        self.metrics.timer(self.name, duration_ms)


# ============================================
# Global Instances
# ============================================

# Global health checker
health_checker = HealthChecker()

# Global metrics collector
metrics = MetricsCollector()


# ============================================
# Built-in Health Checks
# ============================================

def check_vector_store_health(vector_store) -> HealthCheckResult:
    """Check if vector store is healthy."""
    try:
        info = vector_store.get_collection_info()
        doc_count = info.get("document_count", 0)

        return HealthCheckResult(
            name="vector_store",
            status=HealthStatus.HEALTHY if doc_count >= 0 else HealthStatus.DEGRADED,
            message=f"Vector store operational with {doc_count} documents",
            details={"document_count": doc_count}
        )
    except Exception as e:
        return HealthCheckResult(
            name="vector_store",
            status=HealthStatus.UNHEALTHY,
            message=str(e)
        )


def check_embedding_model_health(embeddings) -> HealthCheckResult:
    """Check if embedding model is healthy."""
    try:
        # Try a simple embedding
        start = time.time()
        result = embeddings.embed_query("test")
        latency = (time.time() - start) * 1000

        if result and len(result) > 0:
            return HealthCheckResult(
                name="embedding_model",
                status=HealthStatus.HEALTHY,
                message="Embedding model operational",
                details={
                    "embedding_dim": len(result),
                    "latency_ms": latency
                }
            )
        else:
            return HealthCheckResult(
                name="embedding_model",
                status=HealthStatus.DEGRADED,
                message="Embedding model returned empty result"
            )
    except Exception as e:
        return HealthCheckResult(
            name="embedding_model",
            status=HealthStatus.UNHEALTHY,
            message=str(e)
        )


def check_llm_health(llm) -> HealthCheckResult:
    """Check if LLM is healthy."""
    try:
        if llm is None:
            return HealthCheckResult(
                name="llm",
                status=HealthStatus.DEGRADED,
                message="LLM not initialized (demo mode)"
            )

        return HealthCheckResult(
            name="llm",
            status=HealthStatus.HEALTHY,
            message="LLM operational"
        )
    except Exception as e:
        return HealthCheckResult(
            name="llm",
            status=HealthStatus.UNHEALTHY,
            message=str(e)
        )


# ============================================
# Decorator for timing functions
# ============================================

def timed(metric_name: str):
    """
    Decorator to time function execution.

    Usage:
        @timed("rag_query_duration")
        def query(question):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration_ms = (time.time() - start) * 1000
                metrics.timer(metric_name, duration_ms)
                metrics.increment(f"{metric_name}_count")

        return wrapper
    return decorator


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("Testing Metrics Module...")
    print("=" * 60)

    # Test metrics
    metrics.increment("queries_total", 1)
    metrics.increment("queries_total", 1)
    metrics.gauge("active_connections", 5)
    metrics.timer("query_latency", 123.45)
    metrics.timer("query_latency", 234.56)

    print("Counters:", metrics.get_counters())
    print("\nGauges:", metrics.get_gauges())
    print("\nHistogram stats:", metrics.get_histogram_stats("query_latency"))
    print("\nAll metrics:", metrics.get_all_metrics())

    # Test timer
    print("\nTesting Timer context manager:")
    with Timer(metrics, "operation_time"):
        time.sleep(0.1)

    print("Timer stats:", metrics.get_histogram_stats("operation_time"))

    # Test health checker
    print("\n" + "=" * 60)
    print("Testing Health Checker...")

    health_checker.register("system", lambda: HealthCheckResult(
        name="system",
        status=HealthStatus.HEALTHY,
        message="System OK"
    ))

    print(health_checker.check())

    print("\nMetrics test completed!")
