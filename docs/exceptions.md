# 异常处理模块 (exceptions.py)

## 一、模块概述

`exceptions.py` 提供 RAG 系统的**结构化异常处理**机制，定义了完整的异常层次结构，支持错误码、HTTP 状态码和上下文信息。

> **核心价值**：让错误可追溯、可调试、可展示，提升系统可维护性。

---

## 二、为什么需要自定义异常？

### 2.1 结构化错误的好处

```
普通异常:
  ValueError("文件不存在")
  ↓
  无法区分错误类型，无法自动处理

自定义异常:
  FileNotFoundError("/path/to/file")
  ↓
  error_code: "FILE_NOT_FOUND"
  http_status: 404
  context: {"file_path": "/path/to/file"}
  ↓
  可以自动转换为 API 响应
```

### 2.2 核心优势

| 特性 | 好处 |
|------|------|
| 错误码 | 快速定位问题类型 |
| HTTP 状态码 | 自动转换为 API 响应 |
| 上下文信息 | 便于调试和日志 |
| 层次结构 | 统一捕获特定类型错误 |

---

## 三、异常层次结构

```
┌─────────────────────────────────────────────────────────────┐
│                     异常层次结构                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  RAGException (基类)                                        │
│  ├── ConfigurationError (配置错误)                          │
│  │   └── MissingAPIKeyError (缺少 API Key)                  │
│  │                                                         │
│  ├── DocumentError (文档错误)                               │
│  │   ├── FileNotFoundError (文件未找到)                     │
│  │   ├── UnsupportedFileTypeError (不支持的文件类型)         │
│  │   └── DocumentLoadingError (文档加载失败)                │
│  │                                                         │
│  ├── VectorStoreError (向量存储错误)                        │
│  │   ├── EmbeddingError (嵌入错误)                          │
│  │   ├── CollectionNotFoundError (集合未找到)               │
│  │   └── VectorStoreInitError (初始化失败)                  │
│  │                                                         │
│  ├── LLMError (大模型错误)                                  │
│  │   ├── LLMNotAvailableError (模型不可用)                  │
│  │   ├── LLMResponseError (响应错误)                        │
│  │   └── RateLimitError (速率限制)                          │
│  │                                                         │
│  ├── RetrievalError (检索错误)                              │
│  │   └── NoDocumentsRetrievedError (未检索到文档)           │
│  │                                                         │
│  └── ValidationError (验证错误)                             │
│      ├── EmptyQueryError (空查询)                           │
│      └── InvalidParameterError (无效参数)                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 四、RAGException 基类

### 4.1 类定义

```python
class RAGException(Exception):
    """RAG 系统异常基类"""

    error_code: str = "RAG_ERROR"   # 错误码
    http_status: int = 500          # HTTP 状态码

    def __init__(
        self,
        message: str,                      # 错误消息
        error_code: Optional[str] = None,  # 自定义错误码
        context: Optional[Dict] = None     # 上下文信息
    ):
        self.message = message
        self.error_code = error_code or self.error_code
        self.context = context or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式，用于 API 响应"""
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context
        }
```

### 4.2 属性说明

| 属性 | 类型 | 说明 |
|------|------|------|
| `error_code` | str | 错误码，唯一标识错误类型 |
| `http_status` | int | 对应的 HTTP 状态码 |
| `message` | str | 错误消息 |
| `context` | dict | 上下文信息（文件路径、参数等） |

---

## 五、异常类型详解

### 5.1 ConfigurationError - 配置错误

配置相关问题，通常由缺少环境变量或配置错误引起：

```python
class ConfigurationError(RAGException):
    """配置错误"""
    error_code = "CONFIG_ERROR"

class MissingAPIKeyError(ConfigurationError):
    """缺少 API Key"""
    error_code = "MISSING_API_KEY"

    def __init__(self, key_name: str = "API_KEY"):
        super().__init__(
            message=f"Required API key '{key_name}' not found",
            context={"key_name": key_name}
        )
```

**使用示例：**

```python
from exceptions import MissingAPIKeyError

if not os.getenv("OPENAI_API_KEY"):
    raise MissingAPIKeyError("OPENAI_API_KEY")
```

### 5.2 DocumentError - 文档错误

文档处理相关错误：

```python
class DocumentError(RAGException):
    """文档错误"""
    error_code = "DOCUMENT_ERROR"

class FileNotFoundError(DocumentError):
    """文件未找到"""
    error_code = "FILE_NOT_FOUND"

class UnsupportedFileTypeError(DocumentError):
    """不支持的文件类型"""
    error_code = "UNSUPPORTED_FILE_TYPE"

class DocumentLoadingError(DocumentError):
    """文档加载失败"""
    error_code = "DOCUMENT_LOADING_ERROR"
```

**使用示例：**

```python
from exceptions import UnsupportedFileTypeError

if file_type not in ['.pdf', '.txt', '.md']:
    raise UnsupportedFileTypeError(
        file_type=".docx",
        supported_types=[".pdf", ".txt", ".md"]
    )
```

### 5.3 VectorStoreError - 向量存储错误

向量存储相关错误：

```python
class VectorStoreError(RAGException):
    """向量存储错误"""
    error_code = "VECTOR_STORE_ERROR"

class EmbeddingError(VectorStoreError):
    """嵌入错误"""
    error_code = "EMBEDDING_ERROR"

class CollectionNotFoundError(VectorStoreError):
    """集合未找到"""
    error_code = "COLLECTION_NOT_FOUND"

class VectorStoreInitError(VectorStoreError):
    """初始化失败"""
    error_code = "VECTOR_STORE_INIT_ERROR"
```

**使用示例：**

```python
from exceptions import CollectionNotFoundError

if collection_name not in vector_store.list_collections():
    raise CollectionNotFoundError(collection_name)
```

### 5.4 LLMError - 大模型错误

大语言模型相关错误：

```python
class LLMError(RAGException):
    """大模型错误"""
    error_code = "LLM_ERROR"

class LLMNotAvailableError(LLMError):
    """模型不可用"""
    error_code = "LLM_NOT_AVAILABLE"

class LLMResponseError(LLMError):
    """响应错误"""
    error_code = "LLM_RESPONSE_ERROR"

class RateLimitError(LLMError):
    """速率限制"""
    error_code = "RATE_LIMIT_EXCEEDED"
    http_status = 429  # 特殊状态码
```

**使用示例：**

```python
from exceptions import RateLimitError

try:
    response = llm.invoke(prompt)
except Exception as e:
    if "rate limit" in str(e).lower():
        raise RateLimitError(retry_after=60)
    raise
```

### 5.5 RetrievalError - 检索错误

文档检索相关错误：

```python
class RetrievalError(RAGException):
    """检索错误"""
    error_code = "RETRIEVAL_ERROR"

class NoDocumentsRetrievedError(RetrievalError):
    """未检索到文档"""
    error_code = "NO_DOCUMENTS_RETRIEVED"
```

**使用示例：**

```python
from exceptions import NoDocumentsRetrievedError

docs = retriever.invoke(query)
if not docs:
    raise NoDocumentsRetrievedError(query)
```

### 5.6 ValidationError - 验证错误

输入验证相关错误，通常返回 HTTP 400：

```python
class ValidationError(RAGException):
    """验证错误"""
    error_code = "VALIDATION_ERROR"
    http_status = 400  # 客户端错误

class EmptyQueryError(ValidationError):
    """空查询"""
    error_code = "EMPTY_QUERY"

class InvalidParameterError(ValidationError):
    """无效参数"""
    error_code = "INVALID_PARAMETER"
```

**使用示例：**

```python
from exceptions import EmptyQueryError, InvalidParameterError

if not question.strip():
    raise EmptyQueryError()

if temperature < 0 or temperature > 1:
    raise InvalidParameterError(
        param_name="temperature",
        value=temperature,
        expected="float between 0 and 1"
    )
```

---

## 六、HTTP 状态码对应

| 异常类型 | HTTP 状态码 | 说明 |
|---------|------------|------|
| RAGException | 500 | 通用服务器错误 |
| ConfigurationError | 500 | 配置错误 |
| DocumentError | 500 | 文档处理错误 |
| VectorStoreError | 500 | 向量存储错误 |
| LLMError | 500 | 模型错误 |
| RateLimitError | 429 | 请求过多 |
| ValidationError | 400 | 客户端错误 |

---

## 七、辅助函数

### handle_exception()

将任意异常转换为标准化格式：

```python
def handle_exception(e: Exception) -> Dict[str, Any]:
    """
    将任意异常转换为标准化错误响应

    Args:
        e: 任意异常

    Returns:
        {
            'error': True,
            'error_code': '...',
            'message': '...',
            'context': {...}
        }
    """
    if isinstance(e, RAGException):
        return e.to_dict()

    # 包装未知异常
    return {
        "error": True,
        "error_code": "INTERNAL_ERROR",
        "message": str(e),
        "context": {"exception_type": type(e).__name__}
    }
```

---

## 八、使用示例

### 示例 1：抛出异常

```python
from exceptions import DocumentError, FileNotFoundError

# 简单使用
raise DocumentError("无法加载文档")

# 带上下文信息
raise DocumentError(
    "无法加载文档",
    context={
        "file_path": "/path/to/file.pdf",
        "reason": "文件已损坏"
    }
)

# 使用特定异常
raise FileNotFoundError("/path/to/missing.txt")
```

### 示例 2：捕获和处理异常

```python
from exceptions import RAGException, handle_exception

try:
    result = load_document(file_path)
except RAGException as e:
    # 处理 RAG 系统异常
    error_dict = e.to_dict()
    print(f"错误码: {error_dict['error_code']}")
    print(f"消息: {error_dict['message']}")
    print(f"上下文: {error_dict['context']}")
except Exception as e:
    # 处理其他异常
    error_dict = handle_exception(e)
    print(error_dict)
```

### 示例 3：FastAPI 集成

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from exceptions import RAGException, handle_exception

app = FastAPI()

@app.exception_handler(RAGException)
async def rag_exception_handler(request: Request, exc: RAGException):
    """统一处理 RAG 异常"""
    return JSONResponse(
        status_code=exc.http_status,
        content=exc.to_dict()
    )

@app.post("/query")
async def query(question: str):
    if not question.strip():
        raise EmptyQueryError()  # 自动返回 400

    # ... 处理逻辑
```

### 示例 4：在 RAG 系统中使用

```python
from exceptions import ConfigurationError, DocumentError

class RAGSystem:
    def setup(self):
        if not self.config:
            raise ConfigurationError("系统配置缺失")

    def index_documents(self, path):
        if not os.path.exists(path):
            raise DocumentError(f"路径不存在: {path}")
```

---

## 九、最佳实践

### 9.1 使用具体的异常类型

```python
# 不推荐
raise RAGException("文件不存在")

# 推荐
raise FileNotFoundError(file_path)
```

### 9.2 提供有意义的上下文

```python
# 不推荐
raise DocumentError("加载失败")

# 推荐
raise DocumentLoadingError(
    file_path="/path/to/file.pdf",
    reason="PDF 解析错误: 第 5 页数据损坏"
)
```

### 9.3 在 API 层统一处理

```python
@app.exception_handler(RAGException)
async def rag_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.http_status,
        content=exc.to_dict()
    )
```

### 9.4 区分客户端错误和服务端错误

```python
# 客户端错误（400）- 不应该重试
class ValidationError(RAGException):
    http_status = 400

# 服务端错误（500）- 可以重试
class VectorStoreError(RAGException):
    http_status = 500

# 速率限制（429）- 应该等待后重试
class RateLimitError(LLMError):
    http_status = 429
```

---

## 十、常见问题

### Q: 如何判断异常类型？

```python
from exceptions import DocumentError, VectorStoreError

try:
    process()
except DocumentError as e:
    print("文档相关错误")
except VectorStoreError as e:
    print("向量存储相关错误")
except RAGException as e:
    print("其他 RAG 错误")
```

### Q: 如何添加新的异常类型？

```python
from exceptions import RAGException

class MyCustomError(RAGException):
    """自定义异常"""
    error_code = "MY_CUSTOM_ERROR"
    http_status = 500

    def __init__(self, detail: str):
        super().__init__(
            message=f"自定义错误: {detail}",
            context={"detail": detail}
        )
```

### Q: 如何在日志中记录异常？

```python
import logging
from exceptions import RAGException

logger = logging.getLogger(__name__)

try:
    process()
except RAGException as e:
    logger.error(
        f"RAG 错误: {e.error_code} - {e.message}",
        extra={"context": e.context}
    )
    raise
```

---

## 十一、小结

`exceptions.py` 提供结构化异常处理：

- ✅ 完整的异常层次结构
- ✅ 错误码和 HTTP 状态码
- ✅ 上下文信息支持
- ✅ 统一的转换方法
- ✅ FastAPI 集成支持

掌握异常模块，让错误处理更专业！
