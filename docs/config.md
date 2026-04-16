# 配置管理模块 (config.py)

## 一、模块概述

`config.py` 提供 RAG 系统的**集中式配置管理**，支持环境变量、.env 文件和默认值，实现配置的统一管理和类型安全访问。

> **核心价值**：让配置管理变得简单、统一、类型安全，支持多环境部署。

---

## 二、为什么需要配置管理？

### 2.1 配置的来源

```
┌─────────────────────────────────────────────────────────────┐
│                     配置优先级                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 代码中显式设置（最高优先级）                              │
│     ↓                                                       │
│  2. 环境变量                                                │
│     ↓                                                       │
│  3. .env 文件                                               │
│     ↓                                                       │
│  4. 默认值（最低优先级）                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 配置的挑战

| 挑战 | 解决方案 |
|------|----------|
| 多环境配置 | 使用不同的 .env 文件 |
| 敏感信息保护 | 环境变量，不写入代码 |
| 类型转换 | 提供 get_int、get_float 等方法 |
| 默认值 | 统一的 DEFAULTS 字典 |

---

## 三、核心类：Config

### 3.1 类定义

```python
class Config:
    """
    集中式配置管理

    特性:
    - 支持环境变量
    - 支持 .env 文件
    - 类型安全访问
    - 默认值支持
    """

    # 默认配置值
    DEFAULTS = {
        # 向量存储
        "VECTOR_DB_PATH": "./chroma_db",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",

        # LLM
        "LLM_MODEL": "gpt-3.5-turbo",
        "LLM_TEMPERATURE": "0.7",

        # 检索
        "SEARCH_K": "4",
        "CHUNK_SIZE": "1000",
        "CHUNK_OVERLAP": "200",

        # 混合检索
        "HYBRID_ALPHA": "0.5",
        "BM25_K1": "1.5",
        "BM25_B": "0.75",

        # 系统配置
        "LOG_LEVEL": "INFO",
        "MAX_HISTORY_TURNS": "3",
    }
```

### 3.2 核心方法

#### get() - 获取配置值

```python
@classmethod
def get(cls, key: str, default: Optional[str] = None) -> str:
    """
    获取配置值

    查找顺序:
    1. 环境变量
    2. 默认值

    Args:
        key: 配置键名
        default: 自定义默认值

    Returns:
        配置值字符串
    """
```

#### get_int() - 获取整数配置

```python
@classmethod
def get_int(cls, key: str, default: Optional[int] = None) -> int:
    """获取整数类型的配置值"""
```

#### get_float() - 获取浮点数配置

```python
@classmethod
def get_float(cls, key: str, default: Optional[float] = None) -> float:
    """获取浮点数类型的配置值"""
```

#### get_bool() - 获取布尔配置

```python
@classmethod
def get_bool(cls, key: str, default: Optional[bool] = None) -> bool:
    """
    获取布尔类型的配置值

    True: 'true', '1', 'yes', 'on'
    False: 'false', '0', 'no', 'off', ''
    """
```

#### set() - 设置配置值

```python
@classmethod
def set(cls, key: str, value: str) -> None:
    """
    设置配置值

    实际上是设置环境变量
    """
```

#### get_all() - 获取所有配置

```python
@classmethod
def get_all(cls) -> Dict[str, str]:
    """获取所有配置值（敏感信息会脱敏）"""
```

---

## 四、配置项详解

### 4.1 向量存储配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `VECTOR_DB_PATH` | ./chroma_db | 向量数据库存储路径 |
| `EMBEDDING_MODEL` | all-MiniLM-L6-v2 | 嵌入模型名称 |

### 4.2 LLM 配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_MODEL` | gpt-3.5-turbo | 默认 LLM 模型 |
| `LLM_TEMPERATURE` | 0.7 | 生成温度 |

### 4.3 检索配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `SEARCH_K` | 4 | 检索返回文档数 |
| `CHUNK_SIZE` | 1000 | 文档分块大小 |
| `CHUNK_OVERLAP` | 200 | 分块重叠大小 |

### 4.4 混合检索配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `HYBRID_ALPHA` | 0.5 | 向量检索权重 |
| `BM25_K1` | 1.5 | BM25 k1 参数 |
| `BM25_B` | 0.75 | BM25 b 参数 |

### 4.5 系统配置

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LOG_LEVEL` | INFO | 日志级别 |
| `MAX_HISTORY_TURNS` | 3 | 对话历史轮数 |
| `HF_ENDPOINT` | https://hf-mirror.com | HuggingFace 镜像地址 |

---

## 五、环境变量

### 5.1 API 密钥

```bash
# OpenAI
OPENAI_API_KEY=sk-xxx

# 阿里云通义千问
DASHSCOPE_API_KEY=sk-xxx

# 本地模型
USE_LOCAL_MODEL=true
```

### 5.2 配置覆盖

```bash
# 覆盖默认配置
export LLM_MODEL="qwen-plus"
export LLM_TEMPERATURE="0.5"
export SEARCH_K="6"
export LOG_LEVEL="DEBUG"
```

---

## 六、.env 文件

### 6.1 文件格式

在项目根目录创建 `.env` 文件：

```env
# API Keys
OPENAI_API_KEY=sk-your-openai-key
DASHSCOPE_API_KEY=sk-your-dashscope-key

# LLM 配置
LLM_MODEL=qwen-plus
LLM_TEMPERATURE=0.7

# 向量存储
VECTOR_DB_PATH=./chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2

# 检索配置
SEARCH_K=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# 日志级别
LOG_LEVEL=INFO
```

### 6.2 多环境配置

```bash
# 开发环境
.env.development

# 生产环境
.env.production

# 测试环境
.env.test
```

```python
# 加载指定环境配置
from config import Config

Config(env_file=".env.production")
```

---

## 七、便捷函数

### 7.1 向量存储配置

```python
from config import get_vector_db_path, get_embedding_model

db_path = get_vector_db_path()      # "./chroma_db"
model = get_embedding_model()       # "all-MiniLM-L6-v2"
```

### 7.2 LLM 配置

```python
from config import get_llm_model, get_llm_temperature, has_openai_key

model = get_llm_model()             # "gpt-3.5-turbo"
temp = get_llm_temperature()        # 0.7
has_key = has_openai_key()          # True/False
```

### 7.3 检索配置

```python
from config import (
    get_search_k,
    get_chunk_size,
    get_chunk_overlap,
    get_hybrid_alpha,
    get_bm25_params
)

k = get_search_k()                  # 4
chunk_size = get_chunk_size()       # 1000
overlap = get_chunk_overlap()       # 200
alpha = get_hybrid_alpha()          # 0.5
bm25 = get_bm25_params()            # {"k1": 1.5, "b": 0.75}
```

---

## 八、使用示例

### 示例 1：基本使用

```python
from config import Config

# 获取配置
model = Config.get("LLM_MODEL")
temperature = Config.get_float("LLM_TEMPERATURE")
search_k = Config.get_int("SEARCH_K")

print(f"模型: {model}")
print(f"温度: {temperature}")
print(f"检索数量: {search_k}")
```

### 示例 2：带默认值

```python
from config import Config

# 使用默认值
timeout = Config.get("API_TIMEOUT", default="30")
debug = Config.get_bool("DEBUG", default=False)
```

### 示例 3：运行时修改

```python
from config import Config

# 运行时修改配置
Config.set("LLM_MODEL", "qwen-max")
Config.set("LLM_TEMPERATURE", "0.5")

# 后续调用会使用新值
model = Config.get("LLM_MODEL")  # "qwen-max"
```

### 示例 4：打印所有配置

```python
from config import Config

# 打印所有配置（敏感信息自动脱敏）
Config.print_config()

# 输出:
# ============================================================
#   RAG System Configuration
# ============================================================
#   VECTOR_DB_PATH: ./chroma_db
#   EMBEDDING_MODEL: all-MiniLM-L6-v2
#   LLM_MODEL: qwen-plus
#   LLM_TEMPERATURE: 0.7
#   OPENAI_API_KEY: ***
# ============================================================
```

### 示例 5：条件配置

```python
from config import Config, has_openai_key

# 根据配置选择模型
if has_openai_key():
    model = "gpt-4"
else:
    model = Config.get("DASHSCOPE_MODEL", "qwen-plus")
```

---

## 九、最佳实践

### 9.1 敏感信息处理

```python
# .env 文件
OPENAI_API_KEY=sk-xxx

# .gitignore
.env
.env.local
.env.*.local

# 代码中通过环境变量访问
import os
api_key = os.getenv("OPENAI_API_KEY")
```

### 9.2 类型安全

```python
from config import Config

# 推荐：使用类型安全方法
temperature = Config.get_float("LLM_TEMPERATURE")
search_k = Config.get_int("SEARCH_K")
debug = Config.get_bool("DEBUG")

# 不推荐：手动转换
temperature = float(Config.get("LLM_TEMPERATURE"))  # 可能抛出异常
```

### 9.3 配置验证

```python
def validate_config():
    """验证关键配置"""
    from config import Config

    errors = []

    if not os.getenv("OPENAI_API_KEY") and not os.getenv("DASHSCOPE_API_KEY"):
        errors.append("未配置任何 LLM API Key")

    temp = Config.get_float("LLM_TEMPERATURE")
    if temp < 0 or temp > 1:
        errors.append(f"Temperature 值无效: {temp}")

    if errors:
        raise ValueError("\n".join(errors))
```

### 9.4 配置文档化

```python
# config.py 中使用清晰的命名和注释
DEFAULTS = {
    # 向量数据库路径（存储嵌入向量）
    "VECTOR_DB_PATH": "./chroma_db",

    # 嵌入模型（用于将文本转换为向量）
    "EMBEDDING_MODEL": "all-MiniLM-L6-v2",

    # LLM 模型（用于生成回答）
    "LLM_MODEL": "gpt-3.5-turbo",
}
```

---

## 十、常见问题

### Q: 配置不生效怎么办？

**A:** 检查优先级：

1. 确认 .env 文件在项目根目录
2. 确认环境变量已设置
3. 确认配置键名正确（大小写敏感）

```python
# 调试：打印实际值
from config import Config
print(Config.get("LLM_MODEL"))
print(Config.get_all())
```

### Q: 如何区分开发和生产环境？

**A:** 使用不同的 .env 文件：

```python
import os
from config import Config

env = os.getenv("ENV", "development")
Config(env_file=f".env.{env}")
```

### Q: 如何查看当前所有配置？

```python
from config import Config

# 方式1：打印
Config.print_config()

# 方式2：获取字典
all_config = Config.get_all()
for key, value in all_config.items():
    print(f"{key}: {value}")
```

---

## 十一、小结

`config.py` 提供集中式配置管理：

- ✅ 环境变量支持
- ✅ .env 文件支持
- ✅ 类型安全访问
- ✅ 默认值机制
- ✅ 敏感信息保护

掌握配置模块，让系统配置更规范！
