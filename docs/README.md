# RAG 系统文档中心

## 欢迎来到 RAG 系统文档

本文档库提供 RAG（检索增强生成）系统的**完整技术文档**，从入门到精通，帮助你全面掌握系统的设计理念、核心原理和实战应用。

> **适用人群**：初学者、开发者、架构师、技术管理者

---

## 文档版本

| 项目 | 信息 |
|------|------|
| 文档版本 | 2.0.0 |
| 系统版本 | 2.0.0 |
| 更新日期 | 2024年 |

---

## 一、快速导航

### 我该从哪里开始？

| 你的角色 | 推荐阅读路径 |
|----------|-------------|
| **初学者** | main.md → document_loader.md → vector_store.md → rag_qa.md |
| **开发者** | 全部核心模块 + api.md + 部署实践 |
| **架构师** | 全部文档 + hybrid_retriever.md + 性能优化章节 |
| **运维人员** | metrics.md + retry.md + api.md + 部署章节 |

### 5分钟快速上手

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（二选一）
# 方式A：阿里云通义千问（推荐国内用户）
export DASHSCOPE_API_KEY=your_api_key

# 方式B：OpenAI
export OPENAI_API_KEY=sk-your-key

# 3. 运行演示
python main.py --mode demo

# 4. 索引文档
python main.py --mode index --source ./documents/

# 5. 开始问答
python main.py --mode interactive
```

---

## 二、核心模块文档

### 2.1 系统架构层

#### [main.md](main.md) - RAG 系统编排器

**一句话介绍**：整个系统的"大脑"，协调各模块完成端到端的问答流程。

**你将学到**：
- RAG 技术的核心原理与应用场景
- RAGSystem 类的完整 API
- 四种运行模式（demo/index/query/interactive）
- 命令行参数详解

**核心流程**：
```
用户问题 → 文档检索 → 上下文构建 → LLM生成 → 返回答案
```

**适合谁看**：所有用户必读，理解系统全貌

---

### 2.2 数据处理层

#### [document_loader.md](document_loader.md) - 文档加载器

**一句话介绍**：将各种格式的文档转换为可检索的文本块。

**你将学到**：
- Document 对象结构
- 文本分割策略（RecursiveCharacterTextSplitter）
- chunk_size 和 chunk_overlap 参数调优
- 多格式支持（PDF/TXT/Markdown）
- 编码自动检测

**关键概念**：
```
原始文档 → 文本分割 → Document对象列表 → 向量化
```

**适合谁看**：需要处理知识库文档的用户

---

#### [vector_store.md](vector_store.md) - 向量存储

**一句话介绍**：将文本转化为向量，实现语义检索。

**你将学到**：
- 向量嵌入（Embedding）原理
- ChromaDB 使用方法
- 相似度检索 vs MM检索
- 混合检索集成
- 持久化与加载

**关键概念**：
```
文本 → 嵌入模型 → 向量(1536维) → 存储 → 相似度搜索
```

**适合谁看**：需要理解检索原理的用户

---

### 2.3 问答生成层

#### [rag_qa.md](rag_qa.md) - RAG 问答核心

**一句话介绍**：整合检索结果与 LLM，生成准确、有据可查的回答。

**你将学到**：
- RAGQA 类完整 API
- 多 LLM 提供商支持（OpenAI/通义千问/本地）
- 对话记忆管理
- Temperature 参数调优
- 演示模式（无 API Key 也可运行）

**LLM 提供商选择**：
| 提供商 | 推荐模型 | 适用场景 |
|--------|----------|----------|
| 阿里云通义千问 | qwen-plus | 中文问答（推荐） |
| OpenAI | gpt-3.5-turbo | 英文问答 |
| 本地模型 | 自定义 | 私有化部署 |

**适合谁看**：所有用户必读，掌握问答配置

---

#### [dashscope_llm.md](dashscope_llm.md) - 阿里云通义千问适配器

**一句话介绍**：接入阿里云大模型，国内用户首选。

**你将学到**：
- 通义千问模型选择（turbo/plus/max/long）
- API Key 获取与配置
- DashScopeLLM 类使用
- 嵌入模型集成
- 流式输出实现

**模型对比**：
| 模型 | 特点 | 适用场景 |
|------|------|----------|
| qwen-turbo | 快速响应 | 简单问答 |
| qwen-plus | 性价比高 | 日常使用 |
| qwen-max | 最强能力 | 复杂推理 |
| qwen-long | 长上下文 | 长文档理解 |

**适合谁看**：国内用户必读

---

### 2.4 检索增强层

#### [hybrid_retriever.md](hybrid_retriever.md) - 混合检索

**一句话介绍**：结合关键词检索和语义检索，大幅提升检索质量。

**你将学到**：
- BM25 算法原理与参数
- 向量检索 vs BM25 对比
- Rerank 重排序原理
- RRF 融合算法
- HybridRetriever 使用方法

**检索方式对比**：
| 方式 | 优点 | 缺点 |
|------|------|------|
| 纯向量 | 理解语义 | 遗漏精确匹配 |
| 纯 BM25 | 精确匹配 | 不理解语义 |
| 混合+Rerank | 最佳效果 | 延迟稍高 |

**适合谁看**：追求检索质量的用户

---

#### [query_optimizer.md](query_optimizer.md) - 问题优化

**一句话介绍**：使用 LLM 将模糊问题改写得更清晰，提升检索准确性。

**你将学到**：
- QueryOptimizer 类使用
- QueryExpander 类使用
- 多轮对话代词解析
- Temperature 参数选择
- 启用/禁用策略

**优化效果**：
```
输入: "它是什么？"
输出: "人工智能（AI）是什么？它的定义和主要特征是什么？"
```

**适合谁看**：处理用户模糊提问的场景

---

### 2.5 服务接口层

#### [api.md](api.md) - REST API 服务

**一句话介绍**：将 RAG 系统包装为 HTTP 服务，支持跨语言调用。

**你将学到**：
- FastAPI 框架使用
- API 端点设计
- 请求/响应格式
- 健康检查集成
- Docker 部署方案

**核心端点**：
| 端点 | 方法 | 说明 |
|------|------|------|
| `/query` | POST | 提交问题 |
| `/batch-query` | POST | 批量问答 |
| `/index` | POST | 索引文档 |
| `/health` | GET | 健康检查 |
| `/metrics` | GET | 系统指标 |

**适合谁看**：需要提供 HTTP 服务的用户

---

## 三、生产级模块文档

### 3.1 可靠性保障

#### [exceptions.md](exceptions.md) - 异常处理

**一句话介绍**：结构化的异常体系，让错误可追溯、可调试。

**你将学到**：
- 异常层次结构设计
- 错误码与 HTTP 状态码
- 上下文信息记录
- FastAPI 集成

**异常层次**：
```
RAGException
├── ConfigurationError
├── DocumentError
├── VectorStoreError
├── LLMError
└── ValidationError
```

---

#### [retry.md](retry.md) - 重试与容错

**一句话介绍**：三大容错模式，让系统稳如泰山。

**你将学到**：
- 指数退避重试策略
- 熔断器模式（Circuit Breaker）
- 限流器（Rate Limiter）
- 与 LLM 调用集成

**三大组件**：
| 组件 | 功能 | 解决问题 |
|------|------|----------|
| RetryExecutor | 自动重试 | 临时故障 |
| CircuitBreaker | 快速失败 | 级联故障 |
| RateLimiter | 控制速率 | API 限流 |

---

### 3.2 性能优化

#### [cache.md](cache.md) - 缓存系统

**一句话介绍**：LRU 缓存 + TTL 过期，性能提升数千倍。

**你将学到**：
- LRU 缓存原理
- TTL 过期机制
- 全局缓存实例
- @cached 装饰器

**性能收益**：
```
无缓存: 2400ms
缓存命中: 1ms
提升: 2400倍
```

---

#### [async_support.md](async_support.md) - 异步并发

**一句话介绍**：并发处理，吞吐量提升 10 倍以上。

**你将学到**：
- AsyncExecutor 使用
- AsyncBatchProcessor 批量处理
- 并发控制
- 性能对比

**性能对比**：
```
同步 10 个查询: 20秒
异步并发: 2秒
```

---

### 3.3 可观测性

#### [metrics.md](metrics.md) - 监控指标

**一句话介绍**：让系统运行状态"可见"。

**你将学到**：
- 健康检查系统
- 指标收集（Counter/Gauge/Timer）
- 分位数统计（p50/p90/p99）
- 系统资源监控

**健康状态**：
| 状态 | 说明 | HTTP 状态码 |
|------|------|------------|
| HEALTHY | 正常 | 200 |
| DEGRADED | 降级 | 200 |
| UNHEALTHY | 异常 | 503 |

---

### 3.4 基础设施

#### [config.md](config.md) - 配置管理

**一句话介绍**：集中式配置管理，支持多环境部署。

**你将学到**：
- 环境变量配置
- .env 文件使用
- 类型安全访问
- 配置优先级

---

#### [logger.md](logger.md) - 日志系统

**一句话介绍**：统一日志管理，彩色输出，便于调试。

**你将学到**：
- 日志级别选择
- 彩色控制台输出
- 文件日志
- 最佳实践

---

## 四、模块依赖关系图

```
┌─────────────────────────────────────────────────────────────────────┐
│                           用户接口层                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    命令行 CLI              REST API              Web UI             │
│    (main.py)               (api.py)             (app.py)            │
│                                                                     │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────────────┐
│                           核心业务层                                │
├─────────────────────────────────┼───────────────────────────────────┤
│                                 │                                   │
│    ┌────────────────────────────┼────────────────────────────┐     │
│    │                     RAGSystem                            │     │
│    │                    (main.py)                             │     │
│    │                     系统编排器                            │     │
│    └────────────────────────────┬────────────────────────────┘     │
│                                 │                                   │
│         ┌───────────────────────┼───────────────────────┐          │
│         │                       │                       │          │
│         ▼                       ▼                       ▼          │
│  ┌─────────────┐        ┌─────────────┐        ┌─────────────┐    │
│  │document_    │        │ vector_     │        │   rag_qa    │    │
│  │loader       │        │ store       │        │             │    │
│  │文档加载     │        │ 向量存储    │        │ 问答生成    │    │
│  └─────────────┘        └──────┬──────┘        └──────┬──────┘    │
│                                │                      │            │
│                                ▼                      │            │
│                        ┌─────────────┐                │            │
│                        │hybrid_      │                │            │
│                        │retriever    │◄───────────────┘            │
│                        │ 混合检索    │                             │
│                        └─────────────┘                             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────────────┐
│                           LLM 提供商层                              │
├─────────────────────────────────┼───────────────────────────────────┤
│                                 │                                   │
│    ┌─────────────┐        ┌─────┴─────┐        ┌─────────────┐    │
│    │dashscope_llm│        │  OpenAI   │        │  本地模型   │    │
│    │ 通义千问    │        │   GPT     │        │  (自定义)   │    │
│    │ ⭐推荐      │        │           │        │             │    │
│    └─────────────┘        └───────────┘        └─────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────┼───────────────────────────────────┐
│                           生产级能力层                              │
├─────────────────────────────────┼───────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │
│  │   cache     │  │   retry     │  │  metrics    │  │  async_   │ │
│  │   缓存      │  │  容错       │  │   监控      │  │  support  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘ │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │
│  │ exceptions  │  │   config    │  │   logger    │                │
│  │   异常      │  │   配置      │  │   日志      │                │
│  └─────────────┘  └─────────────┘  └─────────────┘                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 五、典型应用场景

### 场景 1：企业知识库问答系统

```python
from main import RAGSystem

# 1. 初始化系统
rag = RAGSystem(vector_db_path="./company_kb")
rag.setup()

# 2. 索引企业文档
rag.index_documents("./company_documents/")

# 3. 配置问答（使用通义千问）
rag.create_qa_system(
    model_name="qwen-plus",
    llm_provider="dashscope"
)

# 4. 提问
result = rag.ask("公司的请假制度是怎样的？")
print(result["answer"])
```

### 场景 2：技术文档助手

```python
# 启用问题优化和混合检索
rag.create_qa_system(
    model_name="qwen-max",
    optimize_query=True,    # 优化模糊问题
    use_hybrid=True         # 混合检索
)

# 多轮对话
rag.ask("如何部署这个服务？")
rag.ask("需要什么配置？")  # 系统知道"配置"指什么
```

### 场景 3：REST API 服务

```bash
# 启动服务
python api.py

# 配置模型
curl -X POST "http://localhost:8000/setup?model=qwen-plus&provider=dashscope"

# 提问
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RAG？", "show_sources": true}'

# 查看健康状态
curl http://localhost:8000/health
```

### 场景 4：批量文档处理

```python
from async_support import AsyncBatchProcessor

# 并发处理大量文档
processor = AsyncBatchProcessor(batch_size=20)

def index_doc(doc_path):
    return rag.index_documents(doc_path)

results = processor.process_sync(
    items=document_paths,
    process_func=index_doc,
    progress_callback=lambda c, t: print(f"进度: {c}/{t}")
)
```

### 场景 5：监控与告警

```python
from metrics import health_checker, metrics
from retry import llm_circuit_breaker

# 检查系统健康
health = health_checker.check()
if health["status"] == "unhealthy":
    send_alert("RAG 系统异常")

# 检查熔断器状态
if llm_circuit_breaker.state.value == "open":
    send_alert("LLM 服务熔断")

# 查看性能指标
stats = metrics.get_histogram_stats("query_duration")
if stats["p99"] > 5000:  # P99 超过 5 秒
    send_alert("查询性能下降")
```

---

## 六、命令行参考

### 基础命令

```bash
# 演示模式（无需配置 API Key）
python main.py --mode demo

# 索引文档
python main.py --mode index --source ./documents/

# 单次查询
python main.py --mode query --question "什么是人工智能？"

# 交互模式
python main.py --mode interactive
```

### 高级选项

```bash
# 指定 LLM 提供商
python main.py --mode interactive --provider dashscope --model qwen-plus

# 启用问题优化
python main.py --mode interactive --optimize-query

# 启用混合检索
python main.py --mode interactive --hybrid

# 指定向量数据库路径
python main.py --mode interactive --db-path ./my_vectordb

# 完整参数示例
python main.py --mode interactive \
    --provider dashscope \
    --model qwen-max \
    --optimize-query \
    --hybrid \
    --temperature 0.3
```

---

## 七、故障排除指南

### 常见问题速查表

| 问题现象 | 可能原因 | 解决方案 | 参考文档 |
|----------|----------|----------|----------|
| 演示模式提示 | 未配置 API Key | 设置环境变量 | config.md |
| API 调用失败 | 网络问题/密钥错误 | 检查配置/网络 | dashscope_llm.md |
| 检索结果不相关 | 参数不当 | 调整 k 值/启用混合检索 | hybrid_retriever.md |
| 响应速度慢 | 未启用缓存 | 启用缓存/优化参数 | cache.md |
| 内存占用高 | 缓存过大 | 调整缓存大小/TTL | cache.md |
| 服务熔断 | 连续失败 | 检查上游服务/重置熔断器 | retry.md |

### 日志排查

```bash
# 查看日志
tail -f ./logs/rag_system.log

# 查看错误日志
grep "ERROR" ./logs/rag_system.log
```

### 系统诊断

```python
from metrics import health_checker

# 运行健康检查
result = health_checker.check()

for check in result["checks"]:
    print(f"{check['name']}: {check['status']}")
    if check['status'] != 'healthy':
        print(f"  问题: {check['message']}")
```

---

## 八、学习路径建议

### 初学者路径（1-2 周）

```
第1天: 阅读 main.md，运行 demo 模式
第2-3天: 阅读 document_loader.md，索引自己的文档
第4-5天: 阅读 vector_store.md，理解检索原理
第6-7天: 阅读 rag_qa.md，配置问答系统
第2周: 实践一个完整的知识库问答项目
```

### 开发者路径（2-3 周）

```
第1周: 完成初学者路径
第2周: 阅读 api.md、cache.md、retry.md
第3周: 阅读 hybrid_retriever.md、metrics.md，优化系统性能
```

### 架构师路径（3-4 周）

```
第1-2周: 完成开发者路径
第3周: 深入所有模块源码，理解设计模式
第4周: 设计并实现自己的扩展功能
```

---

## 九、相关资源

### 官方文档

- [阿里云通义千问](https://help.aliyun.com/zh/dashscope/)
- [LangChain 文档](https://python.langchain.com/)
- [ChromaDB 文档](https://docs.trychroma.com/)
- [FastAPI 文档](https://fastapi.tiangolo.com/)

### 技术原理

- [RAG 技术综述](https://arxiv.org/abs/2312.10997)
- [BM25 算法详解](https://en.wikipedia.org/wiki/Okapi_BM25)
- [向量检索原理](https://www.pinecone.io/learn/vector-search/)

---

## 十、贡献与反馈

如果你在使用过程中遇到问题或有改进建议，欢迎：

1. 查看各模块文档的"常见问题"章节
2. 检查日志文件 `./logs/rag_system.log`
3. 运行健康检查诊断系统状态

---

**祝你使用愉快！掌握 RAG 系统，构建智能问答应用！**
