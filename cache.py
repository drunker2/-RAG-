#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Caching module for RAG System.
Provides LRU cache with TTL support for embeddings and query results.
"""

import os
import time
import hashlib
import threading
from typing import Dict, Any, Optional, Callable, TypeVar, Generic
from functools import wraps, lru_cache
from collections import OrderedDict
from dataclasses import dataclass, field

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with value and metadata."""
    value: T
    created_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # Time to live in seconds

    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl


class LRUCache(Generic[T]):
    """
    Thread-safe LRU Cache with TTL support.
    """

    def __init__(self, max_size: int = 1000, default_ttl: Optional[float] = None):
        """
        Initialize LRU cache.

        Args:
            max_size: Maximum number of entries
            default_ttl: Default time-to-live in seconds (None = no expiry)
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry[T]] = OrderedDict()
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[T]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with self._lock:
            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self._lock:
            # Use provided TTL or default
            entry_ttl = ttl if ttl is not None else self.default_ttl

            # Remove oldest if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._cache.popitem(last=False)

            # Add/update entry
            self._cache[key] = CacheEntry(
                value=value,
                ttl=entry_ttl
            )
            self._cache.move_to_end(key)

    def delete(self, key: str) -> bool:
        """
        Delete entry from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.2%}",
                "default_ttl": self.default_ttl
            }


# ============================================
# Global Cache Instances
# ============================================

# Embedding cache (larger, longer TTL)
_embedding_cache = LRUCache[list](max_size=10000, default_ttl=3600)  # 1 hour

# Query result cache (smaller, shorter TTL)
_query_cache = LRUCache[dict](max_size=1000, default_ttl=300)  # 5 minutes

# Document chunk cache
_chunk_cache = LRUCache[list](max_size=500, default_ttl=1800)  # 30 minutes


def get_embedding_cache() -> LRUCache:
    """Get the global embedding cache."""
    return _embedding_cache


def get_query_cache() -> LRUCache:
    """Get the global query result cache."""
    return _query_cache


def get_chunk_cache() -> LRUCache:
    """Get the global chunk cache."""
    return _chunk_cache


# ============================================
# Cache Key Generation
# ============================================

def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a cache key from arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        MD5 hash of arguments
    """
    # Convert to string representation
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_string = "|".join(key_parts)

    # Generate MD5 hash
    return hashlib.md5(key_string.encode()).hexdigest()


def text_hash(text: str) -> str:
    """
    Generate hash for text content.

    Args:
        text: Text content

    Returns:
        MD5 hash of text
    """
    return hashlib.md5(text.encode()).hexdigest()


# ============================================
# Caching Decorators
# ============================================

def cached(
    cache: Optional[LRUCache] = None,
    key_func: Optional[Callable] = None,
    ttl: Optional[float] = None
):
    """
    Decorator to cache function results.

    Args:
        cache: Cache instance to use (creates new one if None)
        key_func: Function to generate cache key from arguments
        ttl: Time-to-live for cached entries

    Example:
        @cached(ttl=300)
        def expensive_function(arg):
            return process(arg)
    """
    if cache is None:
        cache = LRUCache(max_size=100, default_ttl=ttl)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = generate_cache_key(func.__name__, *args, **kwargs)

            # Check cache
            result = cache.get(key)
            if result is not None:
                return result

            # Compute and cache
            result = func(*args, **kwargs)
            cache.set(key, result, ttl=ttl)

            return result

        # Add cache management methods
        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.stats

        return wrapper

    return decorator


# ============================================
# Cache Statistics Helper
# ============================================

def get_all_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all global caches."""
    return {
        "embedding_cache": _embedding_cache.stats(),
        "query_cache": _query_cache.stats(),
        "chunk_cache": _chunk_cache.stats()
    }


def clear_all_caches() -> None:
    """Clear all global caches."""
    _embedding_cache.clear()
    _query_cache.clear()
    _chunk_cache.clear()


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("Testing Cache Module...")
    print("=" * 60)

    # Test LRU Cache
    cache = LRUCache[str](max_size=3, default_ttl=5)

    cache.set("a", "value_a")
    cache.set("b", "value_b")
    cache.set("c", "value_c")

    print(f"Get 'a': {cache.get('a')}")  # Hit
    print(f"Get 'b': {cache.get('b')}")  # Hit
    print(f"Stats: {cache.stats()}")

    # Test expiration
    import time
    cache.set("expiring", "will_expire", ttl=1)
    print(f"Get 'expiring': {cache.get('expiring')}")
    time.sleep(1.1)
    print(f"Get 'expiring' after TTL: {cache.get('expiring')}")

    # Test decorator
    @cached(ttl=10)
    def expensive_computation(n: int) -> int:
        print(f"  Computing for n={n}...")
        return n * n

    print("\nTesting cached decorator:")
    print(f"Result 1: {expensive_computation(5)}")  # Computed
    print(f"Result 2: {expensive_computation(5)}")  # Cached
    print(f"Cache stats: {expensive_computation.cache_stats()}")

    print("\nCache test completed!")
