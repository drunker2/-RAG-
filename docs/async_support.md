# 异步支持模块 (async_support.py)

## 一、模块概述

`async_support.py` 提供 RAG 系统的**异步并发能力**，基于线程池和协程实现，支持批量处理和并发执行，大幅提升 I/O 密集型操作的吞吐量。

> **核心价值**：将串行操作并行化，让系统吞吐量提升数倍甚至数十倍。

---

## 二、为什么需要异步？

### 2.1 I/O 密集型 vs CPU 密集型

```
┌─────────────────────────────────────────────────────────────┐
│                     操作类型对比                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CPU 密集型                                                  │
│  ───────────                                                 │
│  特点：计算量大，CPU 持续忙碌                                 │
│  示例：图像处理、机器学习训练                                 │
│  优化：多进程                                                │
│                                                             │
│  I/O 密集型                                                  │
│  ───────────                                                 │
│  特点：等待外部资源，CPU 大部分时间空闲                       │
│  示例：API 调用、数据库查询、文件读写                         │
│  优化：异步并发                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 RAG 系统中的 I/O 操作

| 操作 | 典型耗时 | 等待原因 |
|------|----------|----------|
| LLM API 调用 | 1-5 秒 | 网络等待模型响应 |
| 向量检索 | 10-100ms | 数据库查询 |
| 文本嵌入 | 100-500ms | 模型推理 |
| 文件读取 | 1-100ms | 磁盘 I/O |

### 2.3 性能对比

```
同步执行 10 个 LLM 调用（每个 2 秒）：
总时间 = 10 × 2秒 = 20 秒

异步并发执行 10 个 LLM 调用：
总时间 ≈ 2 秒（受限于最慢的那个）
提升：10 倍
```

---

## 三、核心概念

### 3.1 并发 vs 并行

```
并发（Concurrency）：
  单线程快速切换，"同时"处理多个任务
  ┌─────┐   ┌─────┐   ┌─────┐
  │ A   │ → │ B   │ → │ A   │ → ...
  └─────┘   └─────┘   └─────┘

并行（Parallelism）：
  多线程/多进程，真正同时执行多个任务
  ┌─────┐
  │ A   │ → ...
  └─────┘
  ┌─────┐
  │ B   │ → ...
  └─────┘
```

### 3.2 async/await 原理

```python
import asyncio

async def fetch_data():
    """异步函数"""
    await asyncio.sleep(1)  # 不阻塞，让出控制权
    return "data"

# 运行
result = asyncio.run(fetch_data())
```

### 3.3 本模块的混合方案

```
┌─────────────────────────────────────────────────────────────┐
│                     混合并发方案                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  同步代码（现有 RAG 模块）                                    │
│         │                                                   │
│         ▼                                                   │
│  ┌─────────────────────┐                                    │
│  │   线程池执行器       │  ← ThreadPoolExecutor             │
│  │  AsyncExecutor      │                                    │
│  └──────────┬──────────┘                                    │
│             │                                               │
│             ▼                                               │
│  ┌─────────────────────┐                                    │
│  │   协程调度器         │  ← asyncio                        │
│  │  gather_with_limit  │                                    │
│  └─────────────────────┘                                    │
│                                                             │
│  优点：兼容现有同步代码，同时享受并发优势                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、核心类详解

### 4.1 AsyncResult - 异步结果

```python
@dataclass
class AsyncResult:
    """异步操作的结果封装"""
    success: bool              # 是否成功
    result: Any = None         # 结果值
    error: Exception = None    # 错误信息
    duration_ms: float = 0.0   # 执行耗时（毫秒）
```

### 4.2 AsyncExecutor - 异步执行器

```python
class AsyncExecutor:
    """
    异步执行器

    在线程池中执行同步函数，实现并发
    """

    def __init__(self, max_workers: int = 4):
        """
        初始化执行器

        Args:
            max_workers: 线程池最大线程数
        """

    def run_sync(self, func: Callable, *args, **kwargs) -> Any:
        """
        同步执行函数

        在线程池中执行，不阻塞主线程
        """

    async def run_async(self, func: Callable, *args, **kwargs) -> Any:
        """
        异步执行函数

        返回协程，可以 await
        """

    async def gather_with_limit(
        self,
        funcs: List[Callable],
        limit: int = 10
    ) -> List[AsyncResult]:
        """
        并发执行多个函数，限制并发数

        Args:
            funcs: 函数列表
            limit: 最大并发数

        Returns:
            AsyncResult 列表
        """

    def shutdown(self):
        """关闭执行器，释放资源"""
```

### 4.3 AsyncBatchProcessor - 批量处理器

```python
class AsyncBatchProcessor:
    """
    批量处理器

    支持进度回调，适合大批量数据处理
    """

    def __init__(
        self,
        batch_size: int = 10,
        max_concurrent_batches: int = 5
    ):
        """
        初始化批量处理器

        Args:
            batch_size: 每批处理的数量
            max_concurrent_batches: 最大并发批次数
        """

    def process_sync(
        self,
        items: List[Any],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[AsyncResult]:
        """
        同步批量处理

        Args:
            items: 待处理项列表
            process_func: 处理函数
            progress_callback: 进度回调 (current, total)
        """

    async def process_async(
        self,
        items: List[Any],
        process_func: Callable,
        progress_callback: Optional[Callable] = None
    ) -> List[AsyncResult]:
        """
        异步批量处理
        """
```

---

## 五、使用示例

### 示例 1：基本异步执行

```python
from async_support import AsyncExecutor
import asyncio

# 创建执行器
executor = AsyncExecutor(max_workers=4)

# 定义耗时操作
def slow_operation(n):
    import time
    time.sleep(1)
    return n * 2

# 同步执行
result = executor.run_sync(slow_operation, 5)
print(result)  # 输出: 10

# 异步执行
async def main():
    result = await executor.run_async(slow_operation, 5)
    print(result)  # 输出: 10

asyncio.run(main())

# 关闭执行器
executor.shutdown()
```

### 示例 2：并发执行多个操作

```python
from async_support import AsyncExecutor
import asyncio

async def main():
    executor = AsyncExecutor(max_workers=4)

    def process(n):
        import time
        time.sleep(0.5)
        return n ** 2

    # 并发执行 10 个任务，限制最多 3 个并发
    results = await executor.gather_with_limit(
        funcs=[lambda i=i: process(i) for i in range(10)],
        limit=3
    )

    # 查看结果
    for r in results:
        if r.success:
            print(f"结果: {r.result}, 耗时: {r.duration_ms:.0f}ms")
        else:
            print(f"失败: {r.error}")

    executor.shutdown()

asyncio.run(main())
```

### 示例 3：批量处理

```python
from async_support import AsyncBatchProcessor

# 创建批量处理器
processor = AsyncBatchProcessor(
    batch_size=10,              # 每批 10 个
    max_concurrent_batches=5    # 最多 5 批并发
)

# 定义处理函数
def process_item(item):
    return item.upper()

# 进度回调
def on_progress(current, total):
    percent = (current / total) * 100
    print(f"\r进度: {percent:.1f}%", end="", flush=True)

# 同步批量处理
items = ["a", "b", "c", "d", "e"]
results = processor.process_sync(
    items=items,
    process_func=process_item,
    progress_callback=on_progress
)

# 查看结果
for r in results:
    if r.success:
        print(f"成功: {r.result}")
    else:
        print(f"失败: {r.error}")

# 关闭
processor.shutdown()
```

### 示例 4：批量嵌入

```python
from async_support import AsyncBatchProcessor

def batch_embed_documents(documents, embeddings):
    """批量生成文档嵌入"""
    processor = AsyncBatchProcessor(
        batch_size=16,
        max_concurrent_batches=4
    )

    def embed(doc):
        return embeddings.embed_query(doc.page_content)

    results = processor.process_sync(
        items=documents,
        process_func=embed,
        progress_callback=lambda c, t: print(f"嵌入进度: {c}/{t}")
    )

    return [r.result for r in results if r.success]
```

### 示例 5：批量查询

```python
from async_support import AsyncBatchProcessor

def batch_query(qa_system, questions):
    """批量处理问题"""
    processor = AsyncBatchProcessor(
        batch_size=5,
        max_concurrent_batches=3
    )

    results = processor.process_sync(
        items=questions,
        process_func=lambda q: qa_system.ask(q)
    )

    return [
        {"question": q, "answer": r.result.get("answer")}
        for q, r in zip(questions, results)
        if r.success
    ]
```

---

## 六、专用异步操作类

### 6.1 AsyncVectorStoreOperations

```python
class AsyncVectorStoreOperations:
    """
    向量存储的异步操作封装
    """

    def batch_embed(self, texts: List[str]) -> List[Tuple[str, List[float]]]:
        """并发生成嵌入向量"""

    async def batch_embed_async(self, texts: List[str]) -> List[Tuple[str, List[float]]]:
        """异步并发生成嵌入向量"""

    def batch_search(self, queries: List[str], k: int = 4) -> List[List[Any]]:
        """并发执行多个检索"""

    async def batch_search_async(self, queries: List[str], k: int = 4) -> List[List[Any]]:
        """异步并发检索"""
```

### 6.2 AsyncRAGOperations

```python
class AsyncRAGOperations:
    """
    RAG 系统的异步操作封装
    """

    def batch_ask(self, questions: List[str]) -> List[Dict[str, Any]]:
        """并发处理多个问题"""

    async def batch_ask_async(self, questions: List[str]) -> List[Dict[str, Any]]:
        """异步并发处理问题"""
```

---

## 七、辅助函数

### 7.1 run_async()

在同步代码中运行异步函数：

```python
from async_support import run_async

async def async_operation():
    await asyncio.sleep(1)
    return "完成"

# 在同步代码中运行
result = run_async(async_operation())
print(result)  # 输出: 完成
```

### 7.2 gather_with_timeout()

带超时的并发执行：

```python
from async_support import gather_with_timeout
import asyncio

async def main():
    tasks = [
        asyncio.create_task(asyncio.sleep(i))
        for i in range(5)
    ]

    try:
        results = await gather_with_timeout(tasks, timeout=2.0)
        print("全部完成")
    except asyncio.TimeoutError:
        print("超时，已取消未完成的任务")

asyncio.run(main())
```

---

## 八、并发控制

### 8.1 限制并发数

```python
from async_support import AsyncExecutor
import asyncio

async def main():
    executor = AsyncExecutor(max_workers=4)

    # 限制最多 2 个并发
    results = await executor.gather_with_limit(
        funcs=[lambda: slow_op(i) for i in range(10)],
        limit=2
    )

    executor.shutdown()
```

### 8.2 使用信号量

```python
import asyncio

async def limited_concurrency(items, max_concurrent=5):
    """使用信号量限制并发"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_limit(item):
        async with semaphore:
            return await process_item(item)

    tasks = [process_with_limit(item) for item in items]
    return await asyncio.gather(*tasks)
```

---

## 九、性能对比

### 9.1 同步 vs 异步

```python
import time
from async_support import AsyncExecutor

def slow_task(n):
    time.sleep(0.5)
    return n

# 同步执行 10 个任务
start = time.time()
for i in range(10):
    slow_task(i)
sync_time = time.time() - start
print(f"同步: {sync_time:.1f}秒")  # 约 5.0 秒

# 异步并发执行
import asyncio

async def async_test():
    executor = AsyncExecutor(max_workers=10)
    await executor.gather_with_limit(
        [lambda i=i: slow_task(i) for i in range(10)],
        limit=10
    )
    executor.shutdown()

start = time.time()
asyncio.run(async_test())
async_time = time.time() - start
print(f"异步: {async_time:.1f}秒")  # 约 0.5 秒
```

### 9.2 最佳并发数

```python
# CPU 密集型：并发数 ≈ CPU 核心数
cpu_bound_executor = AsyncExecutor(max_workers=4)

# I/O 密集型：并发数可以更高
io_bound_executor = AsyncExecutor(max_workers=20)

# 网络请求：考虑服务端限制
api_executor = AsyncExecutor(max_workers=10)
```

---

## 十、最佳实践

### 10.1 合理设置并发数

```python
# 根据任务类型调整
# CPU 密集型
cpu_executor = AsyncExecutor(max_workers=os.cpu_count())

# I/O 密集型
io_executor = AsyncExecutor(max_workers=20)

# API 调用（考虑速率限制）
api_executor = AsyncExecutor(max_workers=5)
```

### 10.2 处理异常

```python
from async_support import AsyncBatchProcessor

processor = AsyncBatchProcessor(batch_size=10)

results = processor.process_sync(items, risky_operation)

for r in results:
    if not r.success:
        print(f"处理失败: {r.error}")
        # 可以选择重试或记录日志
```

### 10.3 使用进度回调

```python
def on_progress(current, total):
    percent = (current / total) * 100
    print(f"\r处理进度: {percent:.1f}%", end="", flush=True)

results = processor.process_sync(
    items=large_list,
    process_func=process,
    progress_callback=on_progress
)
print()  # 换行
```

### 10.4 及时关闭执行器

```python
# 方式1：手动关闭
executor = AsyncExecutor(max_workers=4)
try:
    result = executor.run_sync(task)
finally:
    executor.shutdown()

# 方式2：封装为上下文管理器
class AsyncExecutorContext:
    def __init__(self, max_workers=4):
        self.executor = AsyncExecutor(max_workers)

    def __enter__(self):
        return self.executor

    def __exit__(self, *args):
        self.executor.shutdown()

# 使用
with AsyncExecutorContext(max_workers=4) as executor:
    result = executor.run_sync(task)
# 自动关闭
```

---

## 十一、常见问题

### Q: async/await 和线程池有什么区别？

**A:**
- **async/await**：单线程协作式多任务，适合 I/O 密集型
- **线程池**：多线程抢占式多任务，适合 CPU 密集型

本模块使用线程池实现并发，兼容同步代码。

### Q: 什么时候应该使用异步？

**A:** 当操作需要等待外部资源（网络、磁盘、API）时使用异步。对于纯 CPU 计算任务，异步不会提升性能。

### Q: 如何在 Jupyter 中使用？

**A:** Jupyter 已经运行事件循环，直接使用 `await`：

```python
# 在 Jupyter 中直接 await
result = await executor.run_async(task)
```

### Q: 并发数设置多少合适？

**A:**
- I/O 密集型：可以设置较高（10-100）
- CPU 密集型：约等于 CPU 核心数
- 有速率限制的 API：根据限制设置

---

## 十二、小结

`async_support.py` 为 RAG 系统提供并发能力：

- ✅ 线程池执行器
- ✅ 批量处理器
- ✅ 并发控制
- ✅ 进度回调
- ✅ 专用异步操作类

掌握异步模块，让系统吞吐量大幅提升！
