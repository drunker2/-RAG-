# 日志模块 (logger.py)

## 一、模块概述

`logger.py` 提供 RAG 系统的**统一日志管理**，支持彩色控制台输出和文件日志，便于开发调试和生产环境问题追踪。

> **核心价值**：让系统运行状态可视化，便于调试、监控和问题定位。

---

## 二、为什么需要日志？

### 2.1 日志的作用

```
┌─────────────────────────────────────────────────────────────┐
│                     日志的作用                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  开发阶段                                                    │
│  ────────                                                   │
│  - 调试代码执行流程                                          │
│  - 查看变量值                                                │
│  - 定位问题                                                  │
│                                                             │
│  生产阶段                                                    │
│  ────────                                                   │
│  - 监控系统运行状态                                          │
│  - 追踪错误和异常                                            │
│  - 性能分析                                                  │
│  - 审计记录                                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 日志级别

| 级别 | 数值 | 用途 |
|------|------|------|
| DEBUG | 10 | 详细调试信息 |
| INFO | 20 | 一般信息 |
| WARNING | 30 | 警告信息 |
| ERROR | 40 | 错误信息 |
| CRITICAL | 50 | 严重错误 |

---

## 三、核心类

### 3.1 Colors - 颜色定义

```python
class Colors:
    """ANSI 颜色代码，用于终端彩色输出"""

    RESET = "\033[0m"
    RED = "\033[91m"      # 错误
    GREEN = "\033[92m"    # 信息
    YELLOW = "\033[93m"   # 警告
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"  # 严重错误
    CYAN = "\033[96m"     # 调试
    WHITE = "\033[97m"
```

### 3.2 ColoredFormatter - 彩色格式化器

```python
class ColoredFormatter(logging.Formatter):
    """
    自定义格式化器，根据日志级别使用不同颜色

    颜色对应:
    - DEBUG: 青色
    - INFO: 绿色
    - WARNING: 黄色
    - ERROR: 红色
    - CRITICAL: 紫色
    """

    LEVEL_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.MAGENTA,
    }
```

### 3.3 RAGLogger - 日志管理器

```python
class RAGLogger:
    """
    集中式日志管理器

    特性:
    - 单例模式
    - 控制台彩色输出
    - 文件日志（可选）
    - 可配置日志级别
    """

    _instance: Optional['RAGLogger'] = None
    _logger: Optional[logging.Logger] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """获取日志器实例"""

    @classmethod
    def set_level(cls, level: str) -> None:
        """
        设置日志级别

        Args:
            level: DEBUG, INFO, WARNING, ERROR, CRITICAL
        """
```

---

## 四、日志输出

### 4.1 控制台输出

```
[2024-01-15 10:30:00] [INFO   ] RAG 系统初始化完成
[2024-01-15 10:30:01] [WARNING] 向量存储为空，建议索引文档
[2024-01-15 10:30:02] [ERROR  ] API 调用失败: Connection timeout
[2024-01-15 10:30:03] [DEBUG  ] 检索到 4 个相关文档
```

### 4.2 文件输出

日志文件位置：`./logs/rag_system.log`

```
2024-01-15 10:30:00 - RAGSystem - INFO - RAG 系统初始化完成
2024-01-15 10:30:01 - RAGSystem - WARNING - 向量存储为空
2024-01-15 10:30:02 - RAGSystem - ERROR - API 调用失败
```

---

## 五、便捷函数

### 5.1 get_logger()

```python
def get_logger(name: str = "RAGSystem") -> logging.Logger:
    """
    获取日志器实例

    Args:
        name: 日志器名称（可选）

    Returns:
        logging.Logger 实例
    """
```

### 5.2 快捷日志函数

```python
def debug(msg: str) -> None:
    """记录 DEBUG 级别日志"""

def info(msg: str) -> None:
    """记录 INFO 级别日志"""

def warning(msg: str) -> None:
    """记录 WARNING 级别日志"""

def error(msg: str) -> None:
    """记录 ERROR 级别日志"""

def critical(msg: str) -> None:
    """记录 CRITICAL 级别日志"""
```

---

## 六、使用示例

### 示例 1：基本使用

```python
from logger import get_logger

# 获取日志器
logger = get_logger()

# 记录日志
logger.debug("这是调试信息")
logger.info("系统启动成功")
logger.warning("内存使用率较高")
logger.error("API 调用失败")
logger.critical("数据库连接断开")
```

### 示例 2：使用便捷函数

```python
from logger import debug, info, warning, error

# 直接调用函数
debug("调试信息")
info("一般信息")
warning("警告信息")
error("错误信息")
```

### 示例 3：设置日志级别

```python
from logger import RAGLogger

# 设置为 DEBUG 级别（显示所有日志）
RAGLogger.set_level("DEBUG")

# 设置为 WARNING 级别（只显示警告及以上）
RAGLogger.set_level("WARNING")

# 设置为 ERROR 级别（只显示错误及以上）
RAGLogger.set_level("ERROR")
```

### 示例 4：在类中使用

```python
from logger import get_logger

class RAGSystem:
    def __init__(self):
        self.logger = get_logger()
        self.logger.info("RAG 系统初始化...")

    def index_documents(self, path):
        self.logger.debug(f"开始索引文档: {path}")
        try:
            # ... 索引逻辑
            self.logger.info(f"成功索引 {count} 个文档")
        except Exception as e:
            self.logger.error(f"索引失败: {e}")
            raise

    def query(self, question):
        self.logger.info(f"收到查询: {question}")
        # ... 查询逻辑
```

### 示例 5：记录异常

```python
from logger import get_logger

logger = get_logger()

try:
    result = risky_operation()
except Exception as e:
    # 记录异常堆栈
    logger.exception("操作失败")
    # 或者
    logger.error(f"操作失败: {e}", exc_info=True)
```

---

## 七、日志格式

### 7.1 控制台格式

```
[时间戳] [级别] 消息

示例:
[2024-01-15 10:30:00] [INFO   ] 系统启动
```

### 7.2 文件格式

```
时间戳 - 日志器名称 - 级别 - 消息

示例:
2024-01-15 10:30:00 - RAGSystem - INFO - 系统启动
```

---

## 八、日志级别选择

### 8.1 开发环境

```python
# 显示所有日志
RAGLogger.set_level("DEBUG")
```

| 级别 | 使用场景 |
|------|----------|
| DEBUG | 详细调试信息（变量值、执行路径） |
| INFO | 关键操作（初始化、请求处理） |

### 8.2 生产环境

```python
# 只显示警告和错误
RAGLogger.set_level("WARNING")
```

| 级别 | 使用场景 |
|------|----------|
| WARNING | 潜在问题（配置缺失、性能下降） |
| ERROR | 错误但不影响整体运行 |
| CRITICAL | 严重错误（服务不可用） |

---

## 九、最佳实践

### 9.1 有意义的日志消息

```python
# 不推荐
logger.info("处理中...")
logger.error("错误")

# 推荐
logger.info(f"开始处理文档: {file_path}")
logger.error(f"API 调用失败: {error_code} - {error_message}")
```

### 9.2 适当的日志级别

```python
# DEBUG: 详细调试信息
logger.debug(f"检索参数: k={k}, alpha={alpha}")

# INFO: 关键操作
logger.info(f"索引完成: {count} 个文档")

# WARNING: 潜在问题
logger.warning(f"缓存命中率低: {hit_rate}%")

# ERROR: 错误但不影响整体
logger.error(f"单个文档加载失败: {file_path}")

# CRITICAL: 严重错误
logger.critical("数据库连接断开，服务不可用")
```

### 9.3 使用 f-string 格式化

```python
# 推荐
logger.info(f"处理完成: {count} 个文档，耗时 {duration:.2f}s")

# 不推荐
logger.info("处理完成: {} 个文档，耗时 {}s".format(count, duration))
```

### 9.4 异常日志

```python
try:
    process()
except Exception as e:
    # 记录完整堆栈
    logger.exception("处理失败")
    raise
```

### 9.5 避免敏感信息

```python
# 不推荐：记录敏感信息
logger.info(f"用户登录: {username}, 密码: {password}")

# 推荐：脱敏处理
logger.info(f"用户登录: {username}")
```

---

## 十、日志文件管理

### 10.1 日志轮转（推荐生产使用）

如需日志轮转，可修改 `logger.py`：

```python
from logging.handlers import RotatingFileHandler

# 按大小轮转（每个文件最大 10MB，保留 5 个）
file_handler = RotatingFileHandler(
    log_file,
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)

# 或按时间轮转（每天一个文件）
from logging.handlers import TimedRotatingFileHandler

file_handler = TimedRotatingFileHandler(
    log_file,
    when='midnight',  # 每天午夜轮转
    backupCount=7     # 保留 7 天
)
```

### 10.2 日志清理

```bash
# 手动清理旧日志
find ./logs -name "*.log" -mtime +30 -delete
```

---

## 十一、常见问题

### Q: 为什么日志文件没有生成？

**A:** 检查：
1. `./logs` 目录是否有写入权限
2. 是否有异常被静默捕获

```python
# logger.py 中已有异常处理
try:
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_file)
except Exception:
    pass  # 跳过文件日志
```

### Q: 如何禁用彩色输出？

**A:** 修改 `ColoredFormatter.format` 方法，去掉颜色代码。

### Q: 如何同时输出到多个文件？

```python
import logging

logger = logging.getLogger("RAGSystem")

# 添加多个文件处理器
handler1 = logging.FileHandler("app.log")
handler2 = logging.FileHandler("error.log")
handler2.setLevel(logging.ERROR)

logger.addHandler(handler1)
logger.addHandler(handler2)
```

### Q: Jupyter 中日志颜色不显示？

**A:** Jupyter 可能不支持 ANSI 颜色代码。可以使用：

```python
import logging
logger = logging.getLogger("RAGSystem")
logger.handlers[0].setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
```

---

## 十二、小结

`logger.py` 提供统一日志管理：

- ✅ 彩色控制台输出
- ✅ 文件日志支持
- ✅ 日志级别控制
- ✅ 单例模式
- ✅ 便捷函数

掌握日志模块，让系统运行状态一目了然！
