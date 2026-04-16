# 重试与容错模块 (retry.py)

## 一、模块概述

`retry.py` 提供 RAG 系统的**弹性容错能力**，实现三大核心模式：重试（Retry）、熔断器（Circuit Breaker）、限流器（Rate Limiter）。

> **核心价值**：让系统在面对网络抖动、服务不可用、API 限流等故障时依然稳定可靠。

---

## 二、为什么需要容错机制？

### 2.1 分布式系统的现实

在真实环境中，API 调用会因为各种原因失败：

| 失败类型 | 频率 | 应对策略 |
|----------|------|----------|
| 网络超时 | 常见 | 重试 |
| 服务暂时不可用 | 偶发 | 重试 + 熔断 |
| 速率限制 | 常见 | 限流 + 等待 |
| 认证错误 | 罕见 | 立即失败 |
| 参数错误 | 罕见 | 立即失败 |

### 2.2 容错模式概览

```
┌─────────────────────────────────────────────────────────────┐
│                     三大容错模式                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐       │
│  │   重试      │   │  熔断器     │   │   限流器    │       │
│  │  (Retry)    │   │(Circuit     │   │  (Rate      │       │
│  │             │   │ Breaker)    │   │ Limiter)    │       │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘       │
│         │                 │                 │              │
│         ▼                 ▼                 ▼              │
│   应对临时故障       防止级联故障       控制请求速率        │
│   自动重试           快速失败           防止过载           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、重试机制 (Retry)

### 3.1 指数退避算法

指数退避是重试的核心策略，避免频繁重试造成服务压力：

```
重试次数 | 延迟时间
--------|----------
第 1 次 | 1.0 秒
第 2 次 | 2.0 秒 (1.0 × 2)
第 3 次 | 4.0 秒 (2.0 × 2)
第 4 次 | 8.0 秒 (4.0 × 2)
...     | ...
最大    | 60 秒 (上限)

添加抖动(Jitter)后:
第 1 次 | 0.5 ~ 1.5 秒
第 2 次 | 1.0 ~ 3.0 秒
第 3 次 | 2.0 ~ 6.0 秒
```

### 3.2 RetryPolicy - 重试策略

```python
class RetryPolicy:
    """
    重试策略配置

    Attributes:
        max_retries: 最大重试次数
        base_delay: 初始延迟（秒）
        max_delay: 最大延迟上限（秒）
        exponential_base: 指数基数（通常为 2）
        jitter: 是否添加随机抖动
        retryable_exceptions: 可重试的异常类型列表
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
        ...
```

### 3.3 RetryExecutor - 重试执行器

```python
class RetryExecutor:
    """
    重试执行器

    自动处理重试逻辑，记录每次尝试的结果
    """

    def execute(self, func: Callable, *args, **kwargs) -> RetryResult:
        """
        执行函数，失败时自动重试

        Returns:
            RetryResult:
                - state: SUCCESS / FAILED
                - result: 函数返回值
                - error: 最后一次错误
                - attempts: 总尝试次数
                - total_wait_time: 总等待时间
        """
```

### 3.4 RetryResult - 执行结果

```python
@dataclass
class RetryResult:
    """重试操作的结果"""
    state: RetryState          # SUCCESS / FAILED / EXHAUSTED
    result: Any = None         # 成功时的返回值
    error: Exception = None    # 失败时的异常
    attempts: int = 0          # 总尝试次数
    total_wait_time: float = 0 # 总等待时间
```

---

## 四、熔断器 (Circuit Breaker)

### 4.1 熔断器原理

熔断器是一种保护机制，防止故障服务拖垮整个系统：

```
┌─────────────────────────────────────────────────────────────┐
│                     熔断器状态机                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                     ┌─────────┐                            │
│        ┌───────────│ CLOSED  │←──────────┐                │
│        │           └────┬────┘           │                │
│        │                │                │                │
│        │        连续失败 >= 阈值          │                │
│        │                │                │                │
│        │                ▼                │                │
│ 恢复成功     ┌─────────┐         探测成功                │
│        │     │  OPEN   │           │                     │
│        │     └────┬────┘           │                     │
│        │          │                │                     │
│        │   等待恢复时间            │                     │
│        │          │                │                     │
│        │          ▼                │                     │
│        │    ┌──────────┐           │                     │
│        └────│HALF_OPEN │───────────┘                     │
│             └──────────┘                                 │
│                  │                                       │
│           探测失败 │                                      │
│                  ▼                                       │
│             回到 OPEN                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 三种状态

| 状态 | 说明 | 行为 |
|------|------|------|
| CLOSED | 关闭（正常） | 正常执行所有请求 |
| OPEN | 打开（熔断） | 快速失败，不执行请求 |
| HALF_OPEN | 半开（探测） | 允许少量请求测试恢复 |

### 4.3 CircuitBreaker 类

```python
class CircuitBreaker:
    """
    熔断器实现

    防止对失败服务的持续请求，实现快速失败
    """

    def __init__(
        self,
        failure_threshold: int = 5,    # 触发熔断的失败次数
        recovery_timeout: float = 30.0, # 恢复探测等待时间
        half_open_requests: int = 3     # 半开状态测试请求数
    ):
        ...

    def can_execute(self) -> bool:
        """检查是否可以执行请求"""

    def record_success(self):
        """记录成功（可能关闭熔断器）"""

    def record_failure(self):
        """记录失败（可能打开熔断器）"""

    def reset(self):
        """重置熔断器状态"""

    def get_stats(self) -> dict:
        """获取熔断器统计信息"""
```

---

## 五、限流器 (Rate Limiter)

### 5.1 令牌桶算法

限流器使用令牌桶算法控制请求速率：

```
┌─────────────────────────────────────────────────────────────┐
│                     令牌桶算法                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│            令牌以固定速率产生                                 │
│                 ↓                                           │
│         ┌───────────────┐                                   │
│         │   令牌桶      │ ← 最大容量: burst_size            │
│         │  [●●●○○○○○]  │   当前: 3 个令牌                  │
│         └───────┬───────┘                                   │
│                 │                                           │
│                 ▼                                           │
│         请求到达时消耗令牌                                    │
│                                                             │
│  请求1: 有令牌 → 通过                                        │
│  请求2: 有令牌 → 通过                                        │
│  请求3: 有令牌 → 通过                                        │
│  请求4: 无令牌 → 拒绝/等待                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 RateLimiter 类

```python
class RateLimiter:
    """
    令牌桶限流器

    控制请求速率，防止压垮服务
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,  # 每秒请求数
        burst_size: Optional[int] = None    # 突发容量（默认 2x 速率）
    ):
        ...

    def acquire(self, tokens: int = 1) -> bool:
        """
        尝试获取令牌

        Returns:
            True: 获取成功，可执行请求
            False: 获取失败，被限流
        """

    def wait_and_acquire(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        等待直到获取令牌

        Args:
            timeout: 最大等待时间，None 表示无限等待

        Returns:
            True: 获取成功
            False: 超时
        """
```

---

## 六、使用示例

### 示例 1：基本重试

```python
from retry import with_retry

@with_retry(max_retries=3, base_delay=1.0)
def call_external_api():
    """自动重试的 API 调用"""
    response = requests.get("https://api.example.com/data")
    response.raise_for_status()
    return response.json()

# 调用，失败时自动重试最多 3 次
result = call_external_api()
```

### 示例 2：使用 RetryExecutor

```python
from retry import RetryPolicy, RetryExecutor

# 创建策略
policy = RetryPolicy(
    max_retries=3,
    base_delay=1.0,
    max_delay=30.0,
    retryable_exceptions=[ConnectionError, TimeoutError]
)

# 创建执行器
executor = RetryExecutor(policy)

# 执行
def risky_operation():
    return requests.get("https://api.example.com")

result = executor.execute(risky_operation)

if result.state == "success":
    print(f"成功: {result.result}")
else:
    print(f"失败: {result.error}, 尝试了 {result.attempts} 次")
```

### 示例 3：熔断器保护

```python
from retry import CircuitBreaker, CircuitBreakerError, with_circuit_breaker

# 创建熔断器
circuit = CircuitBreaker(
    failure_threshold=5,   # 5 次失败后熔断
    recovery_timeout=30.0, # 30 秒后尝试恢复
    half_open_requests=3   # 恢复时测试 3 个请求
)

@with_circuit_breaker(circuit)
def call_protected_api():
    return requests.get("https://api.example.com")

# 使用
try:
    result = call_protected_api()
except CircuitBreakerError:
    print("服务熔断中，请稍后重试")

# 检查状态
stats = circuit.get_stats()
print(f"熔断器状态: {stats['state']}")
print(f"失败次数: {stats['failure_count']}")
```

### 示例 4：限流控制

```python
from retry import RateLimiter, rate_limit

# 创建限流器：每秒最多 10 个请求，突发最多 20 个
limiter = RateLimiter(
    requests_per_second=10.0,
    burst_size=20
)

# 方式1：手动控制
def rate_limited_call():
    if limiter.acquire():
        return call_api()
    else:
        print("被限流，稍后重试")
        return None

# 方式2：装饰器
@rate_limit(requests_per_second=10, burst_size=20)
def rate_limited_api():
    return call_api()

# 方式3：阻塞等待
def wait_for_rate_limit():
    limiter.wait_and_acquire()  # 等待直到可以执行
    return call_api()
```

### 示例 5：组合使用

```python
from retry import (
    with_retry,
    CircuitBreaker,
    RateLimiter,
    with_circuit_breaker
)

# 创建熔断器和限流器
llm_circuit = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
api_limiter = RateLimiter(requests_per_second=1.0, burst_size=5)

@with_retry(max_retries=3, base_delay=2.0)
@with_circuit_breaker(llm_circuit)
def call_llm_api(prompt: str):
    """受保护的 LLM API 调用"""
    # 等待限流
    api_limiter.wait_and_acquire()

    # 调用 API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# 使用
try:
    answer = call_llm_api("什么是人工智能？")
except CircuitBreakerError:
    answer = "服务暂时不可用，请稍后重试"
```

---

## 七、全局实例

模块提供预配置的全局实例：

```python
from retry import (
    default_retry_executor,  # 默认重试执行器
    llm_circuit_breaker,     # LLM 熔断器
    api_rate_limiter         # API 限流器
)

# 使用全局熔断器
if llm_circuit_breaker.can_execute():
    result = call_llm()
    llm_circuit_breaker.record_success()
else:
    print("LLM 服务熔断中")

# 使用全局限流器
api_rate_limiter.wait_and_acquire()
response = call_api()
```

---

## 八、与 RAG 系统集成

### 8.1 保护 LLM 调用

```python
from retry import with_retry, CircuitBreaker

class RobustRAGQA:
    """带容错的 RAG 问答系统"""

    def __init__(self):
        self.circuit = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30
        )

    @with_retry(max_retries=2, base_delay=1.0)
    def ask(self, question: str):
        if not self.circuit.can_execute():
            return {"answer": "服务暂时不可用", "error": True}

        try:
            result = self._call_llm(question)
            self.circuit.record_success()
            return result
        except Exception as e:
            self.circuit.record_failure()
            raise
```

### 8.2 API 层容错

```python
from fastapi import FastAPI, HTTPException
from retry import CircuitBreakerError

app = FastAPI()

@app.exception_handler(CircuitBreakerError)
async def circuit_breaker_handler(request, exc):
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service Unavailable",
            "message": "服务暂时不可用，请稍后重试"
        }
    )
```

---

## 九、参数调优指南

### 9.1 重试参数

| 场景 | max_retries | base_delay | max_delay |
|------|-------------|------------|-----------|
| 用户交互 | 2-3 | 0.5s | 10s |
| 后台任务 | 5-10 | 1s | 60s |
| 关键服务 | 3-5 | 1s | 30s |

### 9.2 熔断器参数

| 场景 | failure_threshold | recovery_timeout |
|------|-------------------|------------------|
| 核心服务 | 3-5 | 30-60s |
| 非核心服务 | 5-10 | 10-30s |
| 第三方API | 2-3 | 60-120s |

### 9.3 限流器参数

| 服务 | requests_per_second | burst_size |
|------|---------------------|------------|
| OpenAI API | 1-2 | 5 |
| 内部服务 | 10-100 | 20-200 |
| 数据库 | 50-200 | 100-500 |

---

## 十、最佳实践

### 1. 区分可重试和不可重试错误

```python
from retry import RetryPolicy

policy = RetryPolicy(
    max_retries=3,
    retryable_exceptions=[
        ConnectionError,   # 网络问题，重试
        TimeoutError,      # 超时，重试
        # ValueError 不重试（参数错误）
        # AuthenticationError 不重试（认证错误）
    ]
)
```

### 2. 为不同服务使用独立熔断器

```python
# 每个外部服务一个熔断器
db_circuit = CircuitBreaker(failure_threshold=5)
api_circuit = CircuitBreaker(failure_threshold=3)
llm_circuit = CircuitBreaker(failure_threshold=2)
```

### 3. 监控熔断器状态

```python
def health_check():
    stats = llm_circuit.get_stats()
    if stats['state'] == 'open':
        return HealthStatus.DEGRADED, "LLM 服务熔断中"
    return HealthStatus.HEALTHY, "正常"
```

### 4. 合理设置超时

```python
# 超时时间应小于重试总时间
# 例如：3 次重试，base_delay=1s
# 最大等待 ≈ 1 + 2 + 4 = 7s
# 超时应设置为 10s 以上
```

---

## 十一、常见问题

### Q: 重试和熔断器有什么区别？

**A:**
- **重试**：针对单次请求失败，自动重试
- **熔断器**：针对服务整体状态，连续失败后阻止所有请求

### Q: jitter 是什么？

**A:** jitter（抖动）在重试延迟中添加随机性，防止多个客户端同时重试造成"惊群效应"。

```python
# 无 jitter
delay = 2.0  # 所有客户端都在 2 秒后重试

# 有 jitter
delay = 2.0 * (0.5 + random())  # 延迟在 1-3 秒之间随机
```

### Q: 什么时候应该使用限流器？

**A:** 调用有速率限制的 API（如 OpenAI、百度千帆）时，使用限流器防止被限流。

---

## 十二、小结

`retry.py` 提供生产级容错能力：

- ✅ 指数退避重试
- ✅ 熔断器保护
- ✅ 令牌桶限流
- ✅ 装饰器便捷使用
- ✅ 全局预配置实例

掌握容错机制，让系统稳如泰山！
