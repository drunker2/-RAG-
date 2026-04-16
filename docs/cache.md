# 缓存模块 (cache.py)

## 一、模块概述

`cache.py` 提供 RAG 系统的**高性能缓存层**，实现了线程安全的 LRU（最近最少使用）缓存，支持 TTL（生存时间）过期机制。

> **核心价值**：避免重复计算，将毫秒级甚至秒级的操作优化到微秒级响应，大幅提升系统吞吐量。

---

## 二、为什么需要缓存？

### 2.1 RAG 系统中的性能瓶颈

| 操作 | 典型耗时 | 是否适合缓存 |
|------|----------|-------------|
| 文本嵌入 | 100-500ms | 高 - 相同文本嵌入结果恒定 |
| 查询结果 | 1-5秒 | 高 - 相同问题答案相同 |
| 文档分块 | 10-100ms | 中 - 文档不变结果不变 |
| 向量检索 | 10-50ms | 中 - 索引不变结果相同 |

### 2.2 缓存的收益

```
┌─────────────────────────────────────────────────────────────┐
│                     缓存效果示例                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  无缓存查询:                                                 │
│  嵌入(300ms) + 检索(100ms) + LLM生成(2000ms) = 2400ms       │
│                                                             │
│  缓存命中:                                                   │
│  直接返回 = 1ms                                             │
│                                                             │
│  性能提升: 2400倍                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 LRU 缓存原理

LRU（Least Recently Used）是一种经典的缓存淘汰策略：

```
缓存容量 = 3

访问顺序: A → B → C → D → A

时刻1: [A, -, -]      # 添加 A
时刻2: [A, B, -]      # 添加 B
时刻3: [A, B, C]      # 添加 C，缓存满
时刻4: [D, B, C]      # 添加 D，淘汰最久未用的 A
时刻5: [D, B, C]      # 访问 A，未命中，需要重新加载
```

---

## 三、核心类详解

### 3.1 CacheEntry - 缓存条目

```python
@dataclass
class CacheEntry(Generic[T]):
    """缓存条目，存储值和元数据"""
    value: T                        # 缓存的值
    created_at: float               # 创建时间戳
    ttl: Optional[float] = None     # 生存时间（秒）

    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl
```

### 3.2 LRUCache - 核心 LRU 缓存

```python
class LRUCache(Generic[T]):
    """
    线程安全的 LRU 缓存，支持 TTL 过期机制

    特性:
    - 线程安全：所有操作都有锁保护
    - TTL 支持：每个条目可设置过期时间
    - 统计功能：记录命中率、缓存大小等
    """

    def __init__(self,
                 max_size: int = 1000,
                 default_ttl: Optional[float] = None):
        """
        初始化 LRU 缓存

        Args:
            max_size: 最大缓存条目数
            default_ttl: 默认过期时间（秒），None 表示永不过期
        """
```

### 3.3 核心方法详解

#### get() - 获取缓存

```python
def get(self, key: str) -> Optional[T]:
    """
    从缓存获取值

    工作流程:
    1. 检查 key 是否存在
    2. 检查是否过期
    3. 移动到最近使用位置（LRU 核心）
    4. 返回值

    Returns:
        缓存值或 None（未命中/已过期）
    """
```

#### set() - 设置缓存

```python
def set(self, key: str, value: T, ttl: Optional[float] = None) -> None:
    """
    设置缓存值

    工作流程:
    1. 确定 TTL（使用参数或默认值）
    2. 如果缓存已满，淘汰最久未使用的条目
    3. 添加/更新条目
    4. 移动到最近使用位置
    """
```

#### cleanup_expired() - 清理过期

```python
def cleanup_expired(self) -> int:
    """
    主动清理所有过期条目

    Returns:
        清理的条目数量

    建议: 在低峰期定期调用，释放内存
    """
```

#### stats() - 获取统计

```python
def stats(self) -> Dict[str, Any]:
    """
    获取缓存统计信息

    Returns:
        {
            'size': 当前条目数,
            'max_size': 最大容量,
            'hits': 命中次数,
            'misses': 未命中次数,
            'hit_rate': 命中率百分比,
            'default_ttl': 默认TTL
        }
    """
```

---

## 四、全局缓存实例

模块预配置了三个针对不同场景的缓存：

### 4.1 嵌入缓存

```python
# 特点：大容量、长TTL
# 原因：文本嵌入结果稳定，相同文本始终产生相同向量
_embedding_cache = LRUCache[list](
    max_size=10000,    # 10000 个文本
    default_ttl=3600   # 1 小时过期
)

def get_embedding_cache() -> LRUCache:
    """获取全局嵌入缓存"""
    return _embedding_cache
```

### 4.2 查询缓存

```python
# 特点：中等容量、短TTL
# 原因：查询结果可能因知识库更新而变化
_query_cache = LRUCache[dict](
    max_size=1000,     # 1000 个查询
    default_ttl=300    # 5 分钟过期
)

def get_query_cache() -> LRUCache:
    """获取全局查询缓存"""
    return _query_cache
```

### 4.3 文档块缓存

```python
# 特点：小容量、中TTL
# 原因：文档分块结果稳定
_chunk_cache = LRUCache[list](
    max_size=500,      # 500 个文档
    default_ttl=1800   # 30 分钟过期
)

def get_chunk_cache() -> LRUCache:
    """获取全局文档块缓存"""
    return _chunk_cache
```

---

## 五、缓存键生成

### 5.1 generate_cache_key()

从函数参数生成唯一的缓存键：

```python
def generate_cache_key(*args, **kwargs) -> str:
    """
    根据参数生成 MD5 哈希作为缓存键

    Example:
        key = generate_cache_key("func_name", "arg1", "arg2", param="value")
        # 返回 32 位十六进制字符串
    """
```

### 5.2 text_hash()

生成文本内容的哈希：

```python
def text_hash(text: str) -> str:
    """
    生成文本的 MD5 哈希

    用途: 文本内容作为缓存键

    Example:
        key = text_hash("这是一段需要缓存的文本")
    """
```

---

## 六、装饰器模式

### 6.1 @cached 装饰器

最便捷的缓存使用方式：

```python
def cached(
    cache: Optional[LRUCache] = None,
    key_func: Optional[Callable] = None,
    ttl: Optional[float] = None
):
    """
    缓存函数结果的装饰器

    Args:
        cache: 使用的缓存实例（None 则创建新缓存）
        key_func: 自定义缓存键生成函数
        ttl: 过期时间

    自动添加方法:
        func.cache          # 访问缓存实例
        func.cache_clear()  # 清空缓存
        func.cache_stats()  # 获取统计
    """
```

---

## 七、使用示例

### 示例 1：基本使用

```python
from cache import LRUCache

# 创建缓存
cache = LRUCache(max_size=100, default_ttl=300)

# 设置缓存
cache.set("user:1", {"name": "张三", "age": 25})

# 获取缓存
user = cache.get("user:1")
print(user)  # {'name': '张三', 'age': 25}

# 带自定义 TTL
cache.set("temp:data", "临时数据", ttl=60)  # 1 分钟后过期
```

### 示例 2：使用装饰器

```python
from cache import cached

@cached(ttl=300)  # 缓存 5 分钟
def expensive_computation(n):
    """耗时计算函数"""
    print(f"计算中... n={n}")
    return n * n

# 第一次调用，执行计算
result1 = expensive_computation(5)  # 输出: 计算中... n=5
print(result1)  # 输出: 25

# 第二次调用，使用缓存
result2 = expensive_computation(5)  # 无输出（命中缓存）
print(result2)  # 输出: 25

# 查看缓存统计
print(expensive_computation.cache_stats())
# {'size': 1, 'hits': 1, 'misses': 1, 'hit_rate': '50.00%'}
```

### 示例 3：缓存嵌入向量

```python
from cache import get_embedding_cache, text_hash

def get_embedding(text: str, embeddings_model):
    """获取文本嵌入向量（带缓存）"""
    # 使用文本哈希作为缓存键
    cache_key = text_hash(text)
    cache = get_embedding_cache()

    # 查缓存
    cached = cache.get(cache_key)
    if cached:
        return cached

    # 计算嵌入
    embedding = embeddings_model.embed_query(text)

    # 存入缓存
    cache.set(cache_key, embedding)

    return embedding
```

### 示例 4：缓存查询结果

```python
from cache import get_query_cache, generate_cache_key

def query_with_cache(rag_system, question: str, k: int = 4):
    """带缓存的查询"""
    cache = get_query_cache()

    # 生成缓存键
    cache_key = generate_cache_key("query", question, k=k)

    # 查缓存
    cached_result = cache.get(cache_key)
    if cached_result:
        print("缓存命中!")
        return cached_result

    # 执行查询
    result = rag_system.ask(question)

    # 存入缓存
    cache.set(cache_key, result)

    return result
```

### 示例 5：批量缓存管理

```python
from cache import get_all_cache_stats, clear_all_caches

# 查看所有缓存统计
stats = get_all_cache_stats()

for name, stat in stats.items():
    print(f"{name}:")
    print(f"  大小: {stat['size']}/{stat['max_size']}")
    print(f"  命中率: {stat['hit_rate']}")

# 清空所有缓存
clear_all_caches()
```

---

## 八、线程安全

所有缓存操作都是线程安全的，使用 `threading.RLock` 保护：

```python
import threading
from cache import LRUCache

cache = LRUCache(max_size=100)

def worker(thread_id):
    """多线程写入缓存"""
    for i in range(100):
        cache.set(f"thread_{thread_id}_key_{i}", i)
        cache.get(f"thread_{thread_id}_key_{i}")

# 创建 10 个线程并发操作
threads = []
for i in range(10):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

# 安全输出统计
print(cache.stats())
```

---

## 九、TTL 过期机制

### 9.1 工作原理

```
┌─────────────────────────────────────────────────────────────┐
│                     TTL 过期流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  set("key", "value", ttl=60)                                │
│         ↓                                                   │
│  ┌─────────────────┐                                        │
│  │ CacheEntry      │                                        │
│  │ value = "value" │                                        │
│  │ ttl = 60        │                                        │
│  │ created_at = T  │                                        │
│  └─────────────────┘                                        │
│         ↓                                                   │
│  get("key") at T+30:                                        │
│  检查: time.time() - created_at = 30 < 60 ✓                │
│  返回: "value"                                              │
│         ↓                                                   │
│  get("key") at T+61:                                        │
│  检查: time.time() - created_at = 61 > 60 ✗                │
│  返回: None (过期)                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 过期策略

```python
# 惰性过期：get() 时检查（默认）
# 优点：实现简单，无额外开销
# 缺点：过期条目占用内存直到被访问

# 主动过期：定期调用 cleanup_expired()
cache = LRUCache(max_size=1000, default_ttl=300)

# 每小时清理一次（建议在生产环境使用定时任务）
import schedule

def cleanup_job():
    removed = cache.cleanup_expired()
    print(f"清理了 {removed} 条过期缓存")

schedule.every().hour.do(cleanup_job)
```

---

## 十、LRU 淘汰机制详解

### 10.1 淘汰时机

当缓存达到最大容量时，淘汰最久未使用的条目：

```python
cache = LRUCache(max_size=3)

cache.set("a", 1)  # [a]
cache.set("b", 2)  # [a, b]
cache.set("c", 3)  # [a, b, c] 满了

cache.get("a")     # 访问 a，a 变最近使用 [b, c, a]

cache.set("d", 4)  # 添加 d，淘汰最久未使用的 b
                   # [c, a, d]

print(cache.get("b"))  # None（被淘汰）
print(cache.get("a"))  # 1（保留）
```

### 10.2 OrderedDict 实现

```python
# 内部使用 OrderedDict 实现高效的 LRU
# get() 后调用 move_to_end() 将条目移到末尾
# 淘汰时从头部删除（popitem(last=False)）
```

---

## 十一、最佳实践

### 11.1 选择合适的容量和 TTL

```python
# 嵌入向量：不变，长时缓存
embedding_cache = LRUCache(
    max_size=10000,    # 大容量
    default_ttl=3600   # 1 小时
)

# 查询结果：可能变化，短时缓存
query_cache = LRUCache(
    max_size=1000,     # 中容量
    default_ttl=300    # 5 分钟
)

# 实时数据：不缓存或极短 TTL
realtime_cache = LRUCache(
    max_size=100,
    default_ttl=10     # 10 秒
)
```

### 11.2 监控命中率

```python
stats = cache.stats()
hit_rate = float(stats['hit_rate'].rstrip('%')) / 100

if hit_rate < 0.3:
    print("警告：缓存命中率过低")
    print("可能原因：")
    print("1. 缓存键不稳定（每次调用 key 不同）")
    print("2. TTL 过短")
    print("3. 容量过小")
```

### 11.3 缓存键设计

```python
# 错误：时间戳导致 key 每次不同
def bad_cache_key(data):
    import time
    return f"data:{time.time()}"  # 每次调用 key 都不同！

# 正确：使用数据内容的哈希
def good_cache_key(data):
    from cache import text_hash
    return text_hash(str(data))
```

### 11.4 数据更新时清除缓存

```python
def update_document(doc_id, content, cache):
    # 1. 更新数据
    save_to_db(doc_id, content)

    # 2. 清除相关缓存
    cache.delete(f"doc:{doc_id}")
    cache.delete(f"embedding:{doc_id}")
```

---

## 十二、常见问题

### Q1: 缓存占用内存过大？

```python
# 解决方案1：减小容量
cache = LRUCache(max_size=100)  # 而非 10000

# 解决方案2：降低 TTL
cache = LRUCache(default_ttl=60)  # 1 分钟而非 1 小时

# 解决方案3：定期清理
cache.cleanup_expired()
```

### Q2: 缓存命中率低？

检查缓存键是否稳定：

```python
# 问题：使用不稳定参数
@cached()
def process(timestamp):  # timestamp 每次不同
    ...

# 解决：使用稳定参数
@cached(key_func=lambda text: text_hash(text))
def process(text):  # 相同文本 key 相同
    ...
```

### Q3: 数据不一致？

```python
# 方案1：降低 TTL
cache = LRUCache(default_ttl=60)  # 更短的 TTL

# 方案2：主动失效
def update_data(key, value):
    save_to_db(key, value)
    cache.delete(key)  # 主动删除缓存
```

---

## 十三、性能基准

### 13.1 操作延迟

| 操作 | 延迟 |
|------|------|
| get() 命中 | ~1μs |
| get() 未命中 | ~0.5μs |
| set() | ~2μs |
| cleanup_expired() | O(n) |

### 13.2 内存占用

```python
# 每个缓存条目约占用
# Python 对象开销 + key 字符串 + value 对象 + 元数据

# 示例：缓存 10000 个嵌入向量（384 维 float）
# 约 10000 * (384 * 8 + 100) ≈ 32MB
```

---

## 十四、小结

`cache.py` 为 RAG 系统提供生产级缓存能力：

- ✅ 线程安全的 LRU 缓存
- ✅ TTL 过期机制
- ✅ 装饰器便捷使用
- ✅ 命中率统计
- ✅ 全局缓存实例

掌握缓存模块，让系统性能提升一个数量级！
