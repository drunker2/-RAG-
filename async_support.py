#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Async support module for RAG System.
Provides async operations for concurrent requests and batch processing.
"""

import asyncio
import time
from typing import List, Any, Callable, Optional, Dict, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading


@dataclass
class AsyncResult:
    """Result of an async operation."""
    success: bool
    result: Any = None
    error: Optional[Exception] = None
    duration_ms: float = 0.0


class AsyncExecutor:
    """
    Async executor for concurrent operations.
    Provides both thread pool and async execution patterns.
    """

    def __init__(self, max_workers: int = 4):
        """
        Initialize async executor.

        Args:
            max_workers: Maximum number of worker threads
        """
        self.max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def run_sync(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run synchronous function in thread pool.

        Args:
            func: Function to run
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        return self._executor.submit(func, *args, **kwargs).result()

    async def run_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run synchronous function asynchronously in thread pool.

        Args:
            func: Function to run
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs)
        )

    async def gather_with_limit(
        self,
        funcs: List[Callable],
        limit: int = 10
    ) -> List[AsyncResult]:
        """
        Execute multiple functions concurrently with a limit.

        Args:
            funcs: List of functions to execute
            limit: Maximum concurrent executions

        Returns:
            List of AsyncResult objects
        """
        semaphore = asyncio.Semaphore(limit)

        async def run_with_semaphore(func: Callable) -> AsyncResult:
            async with semaphore:
                start_time = time.time()
                try:
                    result = await self.run_async(func)
                    return AsyncResult(
                        success=True,
                        result=result,
                        duration_ms=(time.time() - start_time) * 1000
                    )
                except Exception as e:
                    return AsyncResult(
                        success=False,
                        error=e,
                        duration_ms=(time.time() - start_time) * 1000
                    )

        tasks = [run_with_semaphore(func) for func in funcs]
        return await asyncio.gather(*tasks)

    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown(wait=True)


class AsyncBatchProcessor:
    """
    Batch processor for handling multiple items concurrently.
    """

    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent_batches: int = 5
    ):
        """
        Initialize batch processor.

        Args:
            batch_size: Number of items per batch
            max_concurrent_batches: Maximum concurrent batch processing
        """
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self._executor = AsyncExecutor(max_workers=max_concurrent_batches)

    def process_sync(
        self,
        items: List[Any],
        process_func: Callable[[Any], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[AsyncResult]:
        """
        Process items in batches synchronously.

        Args:
            items: Items to process
            process_func: Function to process each item
            progress_callback: Callback for progress updates

        Returns:
            List of AsyncResult objects
        """
        results = []
        total = len(items)

        def process_batch(batch: List[Any]) -> List[AsyncResult]:
            batch_results = []
            for item in batch:
                start_time = time.time()
                try:
                    result = process_func(item)
                    batch_results.append(AsyncResult(
                        success=True,
                        result=result,
                        duration_ms=(time.time() - start_time) * 1000
                    ))
                except Exception as e:
                    batch_results.append(AsyncResult(
                        success=False,
                        error=e,
                        duration_ms=(time.time() - start_time) * 1000
                    ))
            return batch_results

        # Process in batches
        for i in range(0, total, self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = self._executor.run_sync(process_batch, batch)
            results.extend(batch_results)

            if progress_callback:
                progress_callback(min(i + self.batch_size, total), total)

        return results

    async def process_async(
        self,
        items: List[Any],
        process_func: Callable[[Any], Any],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[AsyncResult]:
        """
        Process items in batches asynchronously.

        Args:
            items: Items to process
            process_func: Function to process each item
            progress_callback: Callback for progress updates

        Returns:
            List of AsyncResult objects
        """
        results = []
        total = len(items)
        semaphore = asyncio.Semaphore(self.max_concurrent_batches)

        async def process_item(item: Any) -> AsyncResult:
            async with semaphore:
                start_time = time.time()
                try:
                    result = await self._executor.run_async(process_func, item)
                    return AsyncResult(
                        success=True,
                        result=result,
                        duration_ms=(time.time() - start_time) * 1000
                    )
                except Exception as e:
                    return AsyncResult(
                        success=False,
                        error=e,
                        duration_ms=(time.time() - start_time) * 1000
                    )

        tasks = [process_item(item) for item in items]
        completed = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1

            if progress_callback:
                progress_callback(completed, total)

        return results

    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown()


# ============================================
# Async Vector Store Operations
# ============================================

class AsyncVectorStoreOperations:
    """
    Async operations for vector store.
    Provides concurrent embedding and retrieval operations.
    """

    def __init__(self, vector_store, max_concurrent: int = 10):
        """
        Initialize async operations.

        Args:
            vector_store: VectorStore instance
            max_concurrent: Maximum concurrent operations
        """
        self.vector_store = vector_store
        self.max_concurrent = max_concurrent
        self._batch_processor = AsyncBatchProcessor(
            batch_size=max_concurrent,
            max_concurrent_batches=max_concurrent
        )

    def batch_embed(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Tuple[str, List[float]]]:
        """
        Create embeddings for multiple texts concurrently.

        Args:
            texts: Texts to embed
            progress_callback: Progress callback

        Returns:
            List of (text, embedding) tuples
        """
        def embed_text(text: str) -> Tuple[str, List[float]]:
            embedding = self.vector_store.embeddings.embed_query(text)
            return (text, embedding)

        results = self._batch_processor.process_sync(
            items=texts,
            process_func=embed_text,
            progress_callback=progress_callback
        )

        return [r.result for r in results if r.success]

    async def batch_embed_async(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Tuple[str, List[float]]]:
        """
        Create embeddings for multiple texts asynchronously.

        Args:
            texts: Texts to embed
            progress_callback: Progress callback

        Returns:
            List of (text, embedding) tuples
        """
        def embed_text(text: str) -> Tuple[str, List[float]]:
            embedding = self.vector_store.embeddings.embed_query(text)
            return (text, embedding)

        results = await self._batch_processor.process_async(
            items=texts,
            process_func=embed_text,
            progress_callback=progress_callback
        )

        return [r.result for r in results if r.success]

    def batch_search(
        self,
        queries: List[str],
        k: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[List[Any]]:
        """
        Search for multiple queries concurrently.

        Args:
            queries: Search queries
            k: Number of results per query
            progress_callback: Progress callback

        Returns:
            List of search results per query
        """
        def search(query: str) -> List[Any]:
            return self.vector_store.similarity_search(query, k=k)

        results = self._batch_processor.process_sync(
            items=queries,
            process_func=search,
            progress_callback=progress_callback
        )

        return [r.result for r in results if r.success]

    async def batch_search_async(
        self,
        queries: List[str],
        k: int = 4,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[List[Any]]:
        """
        Search for multiple queries asynchronously.

        Args:
            queries: Search queries
            k: Number of results per query
            progress_callback: Progress callback

        Returns:
            List of search results per query
        """
        def search(query: str) -> List[Any]:
            return self.vector_store.similarity_search(query, k=k)

        results = await self._batch_processor.process_async(
            items=queries,
            process_func=search,
            progress_callback=progress_callback
        )

        return [r.result for r in results if r.success]


# ============================================
# Async RAG Operations
# ============================================

class AsyncRAGOperations:
    """
    Async operations for RAG system.
    Provides concurrent document processing and querying.
    """

    def __init__(self, qa_system, max_concurrent: int = 5):
        """
        Initialize async operations.

        Args:
            qa_system: RAGQA instance
            max_concurrent: Maximum concurrent operations
        """
        self.qa_system = qa_system
        self.max_concurrent = max_concurrent
        self._batch_processor = AsyncBatchProcessor(
            batch_size=max_concurrent,
            max_concurrent_batches=max_concurrent
        )

    def batch_ask(
        self,
        questions: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ask multiple questions concurrently.

        Args:
            questions: Questions to ask
            progress_callback: Progress callback

        Returns:
            List of answer dictionaries
        """
        results = self._batch_processor.process_sync(
            items=questions,
            process_func=self.qa_system.ask,
            progress_callback=progress_callback
        )

        return [r.result for r in results if r.success]

    async def batch_ask_async(
        self,
        questions: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Ask multiple questions asynchronously.

        Args:
            questions: Questions to ask
            progress_callback: Progress callback

        Returns:
            List of answer dictionaries
        """
        results = await self._batch_processor.process_async(
            items=questions,
            process_func=self.qa_system.ask,
            progress_callback=progress_callback
        )

        return [r.result for r in results if r.success]


# ============================================
# Global instances
# ============================================

# Default async executor
async_executor = AsyncExecutor(max_workers=4)


# ============================================
# Helper functions
# ============================================

def run_async(coro):
    """
    Run async coroutine in sync context.

    Args:
        coro: Coroutine to run

    Returns:
        Result of the coroutine
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running (e.g., in Jupyter), create new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(coro)


async def gather_with_timeout(
    tasks: List[asyncio.Task],
    timeout: float
) -> List[Any]:
    """
    Gather tasks with timeout.

    Args:
        tasks: Tasks to gather
        timeout: Timeout in seconds

    Returns:
        List of results
    """
    try:
        return await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        for task in tasks:
            task.cancel()
        raise


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("Testing Async Support Module...")
    print("=" * 60)

    # Test async executor
    print("\n1. Testing AsyncExecutor:")

    executor = AsyncExecutor(max_workers=4)

    def slow_task(n: int) -> int:
        time.sleep(0.1)
        return n * 2

    async def test_async():
        results = await executor.gather_with_limit(
            [lambda i=i: slow_task(i) for i in range(10)],
            limit=3
        )
        return results

    results = run_async(test_async())
    print(f"  Results: {[r.result for r in results if r.success]}")
    print(f"  Success rate: {sum(1 for r in results if r.success)}/{len(results)}")

    # Test batch processor
    print("\n2. Testing BatchProcessor:")

    processor = AsyncBatchProcessor(batch_size=5, max_concurrent_batches=2)

    items = list(range(20))

    def process_item(x: int) -> int:
        time.sleep(0.05)
        return x ** 2

    def on_progress(current: int, total: int):
        print(f"  Progress: {current}/{total}")

    results = processor.process_sync(items, process_item, on_progress)
    print(f"  Processed: {len(results)} items")
    print(f"  Results: {[r.result for r in results[:5]]}...")

    executor.shutdown()
    processor.shutdown()

    print("\nAsync support module test completed!")
