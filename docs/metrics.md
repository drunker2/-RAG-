# 监控指标模块 (metrics.py)

## 一、模块概述

`metrics.py` 提供 RAG 系统的**可观测性能力**，包括健康检查（Health Check）和指标收集（Metrics Collection），让系统运行状态"可见"。

> **核心价值**：实时掌握系统健康状态，发现性能瓶颈，快速定位问题。

---

## 二、为什么需要监控？

### 2.1 生产环境的三问

```
1. 系统是否正常运行？    → 健康检查
2. 性能表现如何？        → 指标收集
3. 问题出在哪里？        → 日志追踪
```

### 2.2 监控金字塔

```
                    ┌─────────────┐
                    │    告警     │  ← 触发阈值自动通知
                    ├─────────────┤
                    │   可视化    │  ← Grafana、Dashboard
                    ├─────────────┤
                    │   指标      │  ← 本模块核心
                    ├─────────────┤
                    │    日志     │  ← logger.py
                    └─────────────┘
```

### 2.3 RAG 系统关键指标

| 指标类型 | 示例 | 意义 |
|----------|------|------|
| 计数器 | 查询次数、错误次数 | 累计值 |
| 仪表盘 | 当前连接数、内存使用 | 瞬时值 |
| 计时器 | 查询延迟、嵌入耗时 | 分布统计 |

---

## 三、健康检查系统

### 3.1 健康状态

```python
class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"      # 健康：所有组件正常
    DEGRADED = "degraded"    # 降级：部分功能受限
    UNHEALTHY = "unhealthy"  # 不健康：服务不可用
```

### 3.2 HealthCheckResult

```python
@dataclass
class HealthCheckResult:
    """健康检查结果"""
    name: str                        # 检查名称
    status: HealthStatus             # 状态
    message: str = ""                # 描述信息
    details: Dict[str, Any] = {}     # 详细信息
    latency_ms: Optional[float] = None  # 检查耗时
```

### 3.3 HealthChecker 类

```python
class HealthChecker:
    """
    健康检查管理器

    支持注册多个检查项，汇总整体健康状态
    """

    def register(self, name: str, check_func: callable) -> None:
        """
        注册健康检查

        Args:
            name: 检查项名称
            check_func: 检查函数，返回 HealthCheckResult
        """

    def check(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        执行健康检查

        Args:
            name: 指定检查项（None 表示全部）

        Returns:
            {
                'status': 'healthy' | 'degraded' | 'unhealthy',
                'timestamp': '2024-01-15T10:30:00',
                'checks': [HealthCheckResult, ...]
            }
        """
```

---

## 四、指标收集系统

### 4.1 指标类型

```
┌─────────────────────────────────────────────────────────────┐
│                     指标类型对比                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Counter（计数器）                                           │
│  ─────────────                                              │
│  只增不减的累计值                                             │
│  示例：请求总数、错误数                                       │
│                                                             │
│  Gauge（仪表盘）                                             │
│  ─────────────                                              │
│  可增可减的瞬时值                                             │
│  示例：当前连接数、内存使用                                   │
│                                                             │
│  Timer/Histogram（计时器/直方图）                             │
│  ─────────────                                              │
│  记录值的分布情况                                             │
│  示例：请求延迟、响应大小                                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 MetricsCollector 类

```python
class MetricsCollector:
    """
    指标收集器

    线程安全，支持多种指标类型
    """

    # 计数器
    def increment(self, name: str, value: int = 1) -> None:
        """增加计数器"""

    # 仪表盘
    def gauge(self, name: str, value: float) -> None:
        """设置仪表盘值"""

    # 计时器
    def timer(self, name: str, duration_ms: float) -> None:
        """记录耗时"""

    # 直方图
    def histogram(self, name: str, value: float) -> None:
        """记录直方图值"""

    # 获取统计
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """
        获取直方图统计

        Returns:
            {
                'count': 数量,
                'min': 最小值,
                'max': 最大值,
                'mean': 平均值,
                'p50': 中位数,
                'p90': 90分位数,
                'p99': 99分位数
            }
        """

    def get_all_metrics(self) -> Dict[str, Any]:
        """获取所有指标"""
```

### 4.3 Timer 上下文管理器

```python
class Timer:
    """
    计时器上下文管理器

    自动测量代码块执行时间
    """

    def __init__(self, metrics: MetricsCollector, name: str):
        ...

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, *args):
        duration_ms = (time.time() - self.start_time) * 1000
        self.metrics.timer(self.name, duration_ms)
```

---

## 五、预置健康检查

### 5.1 check_vector_store_health()

```python
def check_vector_store_health(vector_store) -> HealthCheckResult:
    """
    检查向量存储健康状态

    检查项:
    - 连接是否正常
    - 文档数量

    Returns:
        HealthCheckResult
    """
```

### 5.2 check_embedding_model_health()

```python
def check_embedding_model_health(embeddings) -> HealthCheckResult:
    """
    检查嵌入模型健康状态

    检查项:
    - 模型是否加载
    - 嵌入功能是否正常
    - 嵌入维度

    Returns:
        HealthCheckResult
    """
```

### 5.3 check_llm_health()

```python
def check_llm_health(llm) -> HealthCheckResult:
    """
    检查大模型健康状态

    检查项:
    - 模型是否初始化
    - 是否处于演示模式

    Returns:
        HealthCheckResult
    """
```

---

## 六、使用示例

### 示例 1：基本健康检查

```python
from metrics import HealthChecker, HealthStatus, HealthCheckResult

# 创建健康检查器
checker = HealthChecker()

# 定义检查函数
def check_database():
    try:
        db.ping()
        return HealthCheckResult(
            name="database",
            status=HealthStatus.HEALTHY,
            message="数据库连接正常"
        )
    except Exception as e:
        return HealthCheckResult(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"数据库连接失败: {e}"
        )

# 注册检查
checker.register("database", check_database)

# 执行检查
result = checker.check()
print(result)
# {
#     'status': 'healthy',
#     'timestamp': '2024-01-15T10:30:00',
#     'checks': [...]
# }
```

### 示例 2：收集指标

```python
from metrics import metrics

# 计数器
metrics.increment("queries_total")
metrics.increment("errors", 5)

# 仪表盘
metrics.gauge("active_connections", 10)
metrics.gauge("memory_usage_mb", 512.5)

# 计时器
metrics.timer("query_latency", 150.5)
metrics.timer("query_latency", 200.3)
metrics.timer("query_latency", 180.2)

# 获取统计
print(metrics.get_counters())
# {'queries_total': 1, 'errors': 5}

print(metrics.get_histogram_stats("query_latency"))
# {'count': 3, 'min': 150.5, 'max': 200.3, 'mean': 177.0, ...}
```

### 示例 3：使用计时器

```python
from metrics import metrics, Timer

# 方式1：上下文管理器
with Timer(metrics, "document_processing"):
    process_documents()
# 自动记录耗时

# 方式2：装饰器
from metrics import timed

@timed("rag_query_duration")
def query_rag(question: str):
    return qa_system.ask(question)

# 调用后自动记录耗时
result = query_rag("什么是AI？")
```

### 示例 4：获取所有指标

```python
from metrics import metrics

all_metrics = metrics.get_all_metrics()

print(all_metrics)
# {
#     'uptime_seconds': 3600.5,
#     'counters': {'queries_total': 100, 'errors': 2},
#     'gauges': {'active_connections': 5, 'memory_usage_mb': 512},
#     'system': {
#         'cpu_percent': 45.2,
#         'memory_percent': 62.8,
#         'disk_percent': 55.0
#     },
#     'python': {
#         'version': '3.12.0',
#         'platform': 'Windows-10'
#     }
# }
```

### 示例 5：与 FastAPI 集成

```python
from fastapi import FastAPI
from metrics import health_checker, metrics

app = FastAPI()

@app.get("/health")
async def health():
    """健康检查端点"""
    return health_checker.check()

@app.get("/metrics")
async def get_metrics():
    """指标端点"""
    return metrics.get_all_metrics()

@app.get("/ready")
async def ready():
    """就绪检查"""
    result = health_checker.check()
    if result["status"] == "unhealthy":
        return {"ready": False, "reason": result}
    return {"ready": True}
```

---

## 七、全局实例

模块提供两个全局实例，方便直接使用：

```python
from metrics import health_checker, metrics

# 全局健康检查器
health_checker.register("my_check", my_check_func)
result = health_checker.check()

# 全局指标收集器
metrics.increment("my_counter")
stats = metrics.get_all_metrics()
```

---

## 八、监控场景

### 8.1 监控查询性能

```python
from metrics import metrics, Timer

def query_rag(question):
    with Timer(metrics, "query_duration"):
        metrics.increment("queries_total")

        try:
            result = execute_query(question)
            metrics.increment("queries_success")
            return result
        except Exception as e:
            metrics.increment("queries_failed")
            raise

# 查看统计
stats = metrics.get_histogram_stats("query_duration")
print(f"平均响应时间: {stats['mean']:.2f}ms")
print(f"P99 响应时间: {stats['p99']:.2f}ms")
```

### 8.2 监控系统资源

```python
from metrics import metrics
import psutil

def collect_system_metrics():
    """定期收集系统指标"""
    metrics.gauge("cpu_percent", psutil.cpu_percent())
    metrics.gauge("memory_percent", psutil.virtual_memory().percent)
    metrics.gauge("disk_percent", psutil.disk_usage('/').percent)

# 使用定时任务
import schedule
schedule.every(10).seconds.do(collect_system_metrics)
```

### 8.3 监控缓存效果

```python
from cache import get_query_cache
from metrics import metrics

def record_cache_stats():
    """记录缓存统计"""
    stats = get_query_cache().stats()
    hit_rate = float(stats['hit_rate'].rstrip('%'))
    metrics.gauge("cache_hit_rate", hit_rate)
    metrics.gauge("cache_size", stats['size'])

schedule.every(30).seconds.do(record_cache_stats)
```

### 8.4 监控熔断器状态

```python
from retry import llm_circuit_breaker
from metrics import metrics

def record_circuit_breaker_status():
    """记录熔断器状态"""
    stats = llm_circuit_breaker.get_stats()
    metrics.gauge("circuit_breaker_failures", stats['failure_count'])

    # 状态转为数值
    state_value = {"closed": 0, "half_open": 1, "open": 2}
    metrics.gauge("circuit_breaker_state", state_value.get(stats['state'], 0))
```

---

## 九、输出格式

### 9.1 健康检查输出

```json
{
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00.123456",
    "checks": [
        {
            "name": "system",
            "status": "healthy",
            "message": "RAG system initialized",
            "details": {},
            "latency_ms": 0.05
        },
        {
            "name": "vector_store",
            "status": "healthy",
            "message": "Vector store operational with 1000 documents",
            "details": {"document_count": 1000},
            "latency_ms": 2.3
        },
        {
            "name": "llm",
            "status": "degraded",
            "message": "LLM not initialized (demo mode)",
            "details": {},
            "latency_ms": 0.01
        }
    ]
}
```

### 9.2 指标输出

```json
{
    "uptime_seconds": 3600.5,
    "counters": {
        "queries_total": 1000,
        "queries_success": 995,
        "queries_failed": 5,
        "documents_indexed": 500
    },
    "gauges": {
        "active_connections": 10,
        "cache_hit_rate": 85.5,
        "memory_usage_mb": 512
    },
    "system": {
        "cpu_percent": 45.2,
        "memory_percent": 62.8,
        "disk_percent": 55.0
    },
    "python": {
        "version": "3.12.0",
        "platform": "Windows-10"
    }
}
```

---

## 十、最佳实践

### 10.1 检查关键依赖

```python
def setup_health_checks(vector_store, embeddings, llm):
    """注册关键组件的健康检查"""
    from metrics import health_checker, check_vector_store_health

    health_checker.register(
        "vector_store",
        lambda: check_vector_store_health(vector_store)
    )

    health_checker.register(
        "embedding_model",
        lambda: check_embedding_model_health(embeddings)
    )

    health_checker.register(
        "llm",
        lambda: check_llm_health(llm)
    )
```

### 10.2 设置合理的超时

```python
import signal

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Health check timeout")

def check_with_timeout(check_func, timeout=5):
    """带超时的健康检查"""
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        return check_func()
    finally:
        signal.alarm(0)
```

### 10.3 返回有意义的详情

```python
def check_vector_store(vector_store):
    try:
        info = vector_store.get_collection_info()
        return HealthCheckResult(
            name="vector_store",
            status=HealthStatus.HEALTHY,
            message=f"运行正常，文档数: {info['document_count']}",
            details={
                "document_count": info['document_count'],
                "embedding_model": info.get('embedding_model', 'unknown')
            }
        )
    except Exception as e:
        return HealthCheckResult(
            name="vector_store",
            status=HealthStatus.UNHEALTHY,
            message=f"连接失败: {e}",
            details={"error": str(e)}
        )
```

### 10.4 区分健康和就绪

```python
# /health - 存活检查（进程是否在运行）
@app.get("/health")
async def health():
    return {"status": "alive"}

# /ready - 就绪检查（是否可以处理请求）
@app.get("/ready")
async def ready():
    result = health_checker.check()
    if result["status"] == "unhealthy":
        return {"ready": False}
    return {"ready": True}
```

---

## 十一、常见问题

### Q: p50、p90、p99 是什么？

**A:** 分位数（Percentile）统计：

```
p50（中位数）：50% 的请求低于此值
p90：90% 的请求低于此值
p99：99% 的请求低于此值

示例：p99 = 500ms 表示 99% 的请求在 500ms 内完成
只有 1% 的请求超过 500ms
```

### Q: 健康检查和指标收集有什么区别？

**A:**
- **健康检查**：判断系统是否正常（是/否）
- **指标收集**：记录系统运行数据（数值）

### Q: 为什么需要 psutil？

**A:** psutil 用于收集系统级指标（CPU、内存、磁盘）。没有安装时，这些指标会返回 None。

```bash
pip install psutil
```

---

## 十二、小结

`metrics.py` 提供 RAG 系统的可观测性能力：

- ✅ 健康检查系统
- ✅ 多类型指标收集
- ✅ 计时器装饰器
- ✅ 系统资源监控
- ✅ 预置检查函数

掌握监控模块，让系统运行状态一目了然！
