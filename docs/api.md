# REST API 模块 (api.py)

## 一、模块概述

`api.py` 提供 RAG 系统的 **REST API 接口**，基于 FastAPI 框架实现，让其他应用能够通过 HTTP 请求调用 RAG 功能。

> **核心价值**：将 RAG 系统包装为 Web 服务，支持跨平台、跨语言调用。

---

## 二、快速开始

### 2.1 安装依赖

```bash
pip install fastapi uvicorn
```

### 2.2 启动服务

```bash
# 方式1：直接运行
python api.py

# 方式2：使用 uvicorn
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 2.3 访问文档

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## 三、API 端点列表

### 3.1 基础端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | API 基本信息 |
| GET | `/health` | 健康检查 |
| GET | `/info` | 系统信息 |

### 3.2 核心功能

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/query` | 提交问题 |
| POST | `/batch-query` | 批量问答 |
| POST | `/index` | 索引文档 |
| POST | `/setup` | 配置系统 |

### 3.3 监控管理

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/stats` | 系统统计 |
| GET | `/cache` | 缓存统计 |
| DELETE | `/cache` | 清空缓存 |
| GET | `/circuit-breaker` | 熔断器状态 |
| POST | `/circuit-breaker/reset` | 重置熔断器 |

---

## 四、核心端点详解

### 4.1 POST /query - 提交问题

**请求体：**

```json
{
    "question": "什么是人工智能？",
    "show_sources": true,
    "max_sources": 3
}
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| question | string | 必填 | 问题内容 (1-2000字符) |
| show_sources | bool | true | 是否显示来源 |
| max_sources | int | 3 | 最大来源数量 (1-10) |

**响应示例：**

```json
{
    "answer": "人工智能是模拟人类智能的技术...",
    "sources": [
        {
            "source_id": 1,
            "content": "相关文档内容...",
            "metadata": {"source": "doc.txt"}
        }
    ],
    "original_question": "什么是人工智能？",
    "search_question": "什么是人工智能？它的定义和特点是什么？",
    "query_was_optimized": true,
    "llm_type": "dashscope"
}
```

### 4.2 POST /batch-query - 批量问答

**请求体：**

```json
{
    "questions": [
        "什么是人工智能？",
        "什么是机器学习？",
        "什么是深度学习？"
    ]
}
```

**响应示例：**

```json
{
    "total": 3,
    "successful": 3,
    "results": [
        {
            "question": "什么是人工智能？",
            "answer": "人工智能是...",
            "success": true
        }
    ]
}
```

### 4.3 POST /index - 索引文档

**请求体：**

```json
{
    "source_path": "./documents",
    "collection_name": "rag_collection"
}
```

**响应示例：**

```json
{
    "success": true,
    "message": "Indexing started in background",
    "documents_indexed": 0
}
```

### 4.4 POST /setup - 配置系统

**查询参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | string | ERNIE-3.5-8K | 模型名称 |
| provider | string | auto | LLM 提供商 |
| temperature | float | 0.7 | 温度参数 |
| use_conversation | bool | false | 启用对话 |
| optimize_query | bool | false | 启用问题优化 |
| use_hybrid | bool | false | 启用混合检索 |
| hybrid_alpha | float | 0.5 | 混合检索权重 |

**示例：**

```bash
curl -X POST "http://localhost:8000/setup?model=qwen-plus&provider=dashscope&temperature=0.7"
```

---

## 五、使用示例

### 5.1 curl 命令

```bash
# 健康检查
curl http://localhost:8000/health

# 提交问题
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RAG？", "show_sources": true}'

# 批量查询
curl -X POST http://localhost:8000/batch-query \
  -H "Content-Type: application/json" \
  -d '{"questions": ["问题1", "问题2"]}'

# 清空缓存
curl -X DELETE http://localhost:8000/cache
```

### 5.2 Python requests

```python
import requests

# 健康检查
response = requests.get("http://localhost:8000/health")
print(response.json())

# 配置系统
response = requests.post(
    "http://localhost:8000/setup",
    params={"model": "qwen-plus", "provider": "dashscope"}
)

# 提交问题
response = requests.post(
    "http://localhost:8000/query",
    json={"question": "什么是人工智能？"}
)
result = response.json()
print(f"回答: {result['answer']}")
```

### 5.3 JavaScript Fetch

```javascript
async function queryRAG(question) {
    const response = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question: question,
            show_sources: true
        })
    });
    return response.json();
}

// 使用
const result = await queryRAG('什么是人工智能？');
console.log(result.answer);
```

---

## 六、错误处理

### 6.1 错误响应格式

```json
{
    "error": true,
    "error_code": "VALIDATION_ERROR",
    "message": "Query cannot be empty",
    "context": {}
}
```

### 6.2 常见错误码

| 错误码 | HTTP 状态码 | 说明 |
|--------|------------|------|
| EMPTY_QUERY | 400 | 查询为空 |
| INVALID_PARAMETER | 400 | 参数无效 |
| RATE_LIMIT_EXCEEDED | 429 | 请求过多 |
| LLM_NOT_AVAILABLE | 503 | 模型不可用 |

---

## 七、部署指南

### 7.1 使用 uvicorn

```bash
# 开发模式
uvicorn api:app --reload

# 生产模式
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

### 7.2 Docker 部署

**Dockerfile:**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**构建和运行：**

```bash
docker build -t rag-api .
docker run -p 8000:8000 \
  -e DASHSCOPE_API_KEY=xxx \
  rag-api
```

---

## 八、环境变量

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| RAG_DB_PATH | 向量数据库路径 | ./chroma_db |
| RAG_COLLECTION | 集合名称 | rag_collection |
| DASHSCOPE_API_KEY | 阿里云 API Key | - |
| OPENAI_API_KEY | OpenAI API Key | - |

---

## 九、小结

`api.py` 让 RAG 系统能够：

- ✅ 提供 REST API 接口
- ✅ 支持跨平台调用
- ✅ 自动生成 API 文档
- ✅ 支持生产部署
- ✅ 完善的错误处理

掌握 API 模块，就能将 RAG 系统集成到任何应用中！
