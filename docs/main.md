# RAG 系统主模块 (main.py)

## 一、模块概述

`main.py` 是整个 RAG（检索增强生成）系统的**核心入口文件**，扮演着"指挥官"的角色。它将文档加载、向量存储、问答系统等模块有机地组织在一起，提供完整的工作流程。

> **为什么这个文件很重要？**
> 想象一下，一个大型项目就像一个复杂的机器，每个模块是机器的零件。`main.py` 就是控制面板，让用户能够方便地操作这台机器。

---

## 二、核心概念：什么是 RAG？

RAG = Retrieval（检索）+ Augmented（增强）+ Generation（生成）

```
┌─────────────────────────────────────────────────────────┐
│                     RAG 工作流程                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  用户问题 ──→ 检索相关文档 ──→ 构建上下文 ──→ LLM生成回答   │
│               (向量搜索)      (合并文档)     (基于事实)    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**RAG 解决了什么问题？**

| 问题 | 传统 LLM | RAG 系统 |
|------|----------|----------|
| 知识过时 | 无法更新 | 可随时更新文档库 |
| 幻觉问题 | 经常编造 | 基于检索的事实回答 |
| 专业领域 | 知识有限 | 可注入专业知识 |
| 来源追溯 | 无法溯源 | 提供参考文档 |

---

## 三、核心类：RAGSystem

### 3.1 类定义

```python
class RAGSystem:
    """Complete RAG system orchestrator with production features."""
```

这是整个系统的"大脑"，负责协调各个组件。

### 3.2 初始化参数

```python
def __init__(self,
             vector_db_path: str = "./chroma_db",
             embedding_model: str = "all-MiniLM-L6-v2"):
    """
    初始化 RAG 系统

    Args:
        vector_db_path: 向量数据库存储路径
        embedding_model: 嵌入模型名称
    """
```

**参数详解：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vector_db_path` | `./chroma_db` | 向量数据库保存位置 |
| `embedding_model` | `all-MiniLM-L6-v2` | 将文本转换为向量的模型 |

### 3.3 核心属性

```python
self.vector_db_path = vector_db_path      # 数据库路径
self.embedding_model = embedding_model    # 嵌入模型名称
self.document_loader: Optional[DocumentLoader] = None  # 文档加载器
self.vector_store: Optional[VectorStore] = None        # 向量存储
self.qa_system: Optional[RAGQA] = None                 # 问答系统
```

---

## 四、核心方法详解

### 4.1 setup() - 系统初始化

```python
def setup(self,
          chunk_size: int = 1000,
          chunk_overlap: int = 200) -> None:
    """
    Setup document loader and vector store.

    Args:
        chunk_size: 文档分块大小（字符数）
        chunk_overlap: 分块之间的重叠大小
    """
```

**为什么要分块？**

```
原始文档（10000字符）
    │
    ├── 分块1 (0-1000字符)
    ├── 分块2 (800-1800字符)    ← 重叠200字符
    ├── 分块3 (1600-2600字符)   ← 重叠200字符
    └── ...
```

**分块参数最佳实践：**

| 文档类型 | 推荐 chunk_size | 推荐 overlap |
|----------|-----------------|--------------|
| 技术文档 | 1000-1500 | 200 |
| 新闻文章 | 500-800 | 100 |
| 代码文件 | 300-500 | 50 |
| FAQ/问答 | 200-400 | 50 |

### 4.2 index_documents() - 文档索引

```python
def index_documents(self,
                    source_path: str,
                    collection_name: str = "rag_collection") -> bool:
    """
    将文档索引到向量数据库

    Args:
        source_path: 源文件或目录路径
        collection_name: 集合名称

    Returns:
        True 表示成功，False 表示失败
    """
```

**工作流程：**

```
source_path ──→ 检测文件类型 ──→ 加载文档 ──→ 分块 ──→ 向量化 ──→ 存储
    │
    ├── .pdf 文件 ──→ load_pdf()
    ├── .txt 文件 ──→ load_text()
    └── 目录   ──→ load_directory()
```

### 4.3 create_qa_system() - 创建问答系统

```python
def create_qa_system(self,
                     model_name: str = "gpt-3.5-turbo",
                     temperature: float = 0.7,
                     use_conversation: bool = False,
                     use_local_model: bool = False,
                     search_k: int = 10,
                     optimize_query: bool = False,
                     use_hybrid: bool = True,
                     hybrid_alpha: float = 0.5,
                     candidate_k: int = 20,
                     use_rerank: bool = True,
                     llm_provider: str = "auto") -> bool:
```

**关键参数解释：**

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `model_name` | LLM 模型名称 | qwen-plus (阿里云) |
| `temperature` | 输出随机性 (0-1) | 0.7 (平衡) |
| `use_conversation` | 是否启用对话记忆 | True/False |
| `use_hybrid` | 是否使用混合检索 | True |
| `use_rerank` | 是否使用重排序 | True |
| `search_k` | 最终返回文档数 | 4-10 |
| `candidate_k` | 每路召回候选数 | 20 |
| `llm_provider` | LLM 提供商 | auto |

**llm_provider 取值：**

| 值 | 说明 |
|----|------|
| `auto` | 自动选择（OpenAI → DashScope → Demo） |
| `openai` | 强制使用 OpenAI |
| `dashscope` | 强制使用阿里云通义千问 |
| `local` | 使用本地模型 |
| `demo` | 演示模式（无需 API Key） |

### 4.4 ask() - 提问方法

```python
def ask(self,
        question: str,
        show_sources: bool = True,
        max_sources: int = 3) -> Optional[dict]:
    """
    向 RAG 系统提问

    Args:
        question: 用户问题
        show_sources: 是否显示来源文档
        max_sources: 最大显示来源数量

    Returns:
        包含答案和来源的字典
    """
```

**返回值结构：**

```python
{
    "answer": "这是生成的答案...",
    "sources": [Document, Document, ...],
    "formatted_sources": [...],
    "original_question": "原始问题",
    "search_question": "优化后的搜索问题",
    "query_was_optimized": True/False,
    "llm_type": "dashscope"
}
```

---

## 五、运行模式

### 5.1 Demo 模式

```bash
python main.py --mode demo
```

适用于快速测试，会创建示例文档并演示基本功能。

### 5.2 索引模式

```bash
python main.py --mode index --source ./my_documents
```

将指定目录的文档索引到向量数据库。

### 5.3 查询模式

```bash
python main.py --mode query --question "什么是人工智能？"
```

单次问答模式，适合脚本调用。

### 5.4 交互模式

```bash
python main.py --mode interactive
```

进入交互式问答界面，支持多轮对话。

**交互模式命令：**

| 命令 | 功能 |
|------|------|
| `quit/exit/q` | 退出程序 |
| `help/h/?` | 显示帮助 |
| `clear` | 清除对话记忆 |
| `sources on/off` | 开关来源显示 |
| `info` | 显示系统信息 |
| `history` | 显示对话历史 |
| `stats` | 显示向量库统计 |
| `health` | 显示系统健康状态 |
| `cache` | 显示缓存统计 |

---

## 六、命令行参数完整列表

```bash
python main.py [OPTIONS]

运行模式:
  --mode {index,query,interactive,demo}
                        运行模式 (默认: demo)

索引参数:
  --source PATH         要索引的文件或目录路径
  --collection NAME     集合名称 (默认: rag_collection)

查询参数:
  --question TEXT       要问的问题

系统配置:
  --db PATH             向量数据库路径 (默认: ./chroma_db)
  --model NAME          LLM 模型名称 (默认: qwen-plus)
  --provider PROVIDER   LLM 提供商 (默认: auto)
  --temp FLOAT          温度参数 (默认: 0.7)

检索配置:
  --search-k N          最终返回文档数 (默认: 10)
  --candidate-k N       各路召回候选数 (默认: 20)
  --no-hybrid           禁用混合检索
  --no-rerank           禁用重排序
  --hybrid-alpha FLOAT  RRF 权重 (默认: 0.5)

文档处理:
  --chunk-size N        分块大小 (默认: 1000)
  --chunk-overlap N     分块重叠 (默认: 200)
  --optimize-query      启用查询优化
```

---

## 七、完整使用示例

### 示例 1：快速开始

```python
from main import RAGSystem

# 1. 创建系统实例
rag = RAGSystem(vector_db_path="./my_db")

# 2. 初始化
rag.setup(chunk_size=500, chunk_overlap=100)

# 3. 索引文档
rag.index_documents("./documents")

# 4. 创建问答系统
rag.create_qa_system(
    model_name="qwen-plus",
    use_hybrid=True,
    use_rerank=True
)

# 5. 提问
result = rag.ask("什么是机器学习？")
print(result["answer"])
```

### 示例 2：加载已有索引

```python
from main import RAGSystem

rag = RAGSystem(vector_db_path="./my_db")

# 加载已有向量库
rag.load_existing_index()

# 创建问答系统
rag.create_qa_system(use_conversation=True)

# 交互式问答
rag.interactive_mode()
```

### 示例 3：混合检索配置

```python
# 高精度检索配置
rag.create_qa_system(
    use_hybrid=True,
    candidate_k=30,     # 每路召回更多候选
    search_k=5,         # 最终返回5篇
    use_rerank=True     # 使用重排序
)

# 快速检索配置
rag.create_qa_system(
    use_hybrid=False,   # 仅向量检索
    search_k=3          # 返回3篇
)
```

---

## 八、架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        RAGSystem                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │DocumentLoader│  │ VectorStore │  │      RAGQA          │ │
│  │  文档加载器  │  │  向量存储   │  │    问答系统         │ │
│  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘ │
│         │                │                    │            │
│         ▼                ▼                    ▼            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ 文档分块    │  │ ChromaDB    │  │ HybridRetriever    │ │
│  │ 编码检测    │  │ Embeddings  │  │ QueryOptimizer     │ │
│  │ 多格式支持  │  │ 相似度搜索  │  │ LLM Provider       │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 九、常见问题

### Q1: 为什么我的回答是 Demo Mode？

**原因**：未配置 LLM API Key。

**解决方案**：
```bash
# 方法1: 设置环境变量
export DASHSCOPE_API_KEY=your_key_here

# 方法2: 创建 .env 文件
echo "DASHSCOPE_API_KEY=your_key_here" > .env
```

### Q2: 如何选择分块大小？

| 文档类型 | 推荐 chunk_size | 推荐 overlap |
|----------|-----------------|--------------|
| 技术文档 | 1000-1500 | 200 |
| 新闻文章 | 500-800 | 100 |
| 代码文件 | 300-500 | 50 |
| FAQ/问答 | 200-400 | 50 |

### Q3: 混合检索和纯向量检索怎么选？

- **混合检索**：适合需要精确关键词匹配的场景（如技术文档、代码）
- **纯向量检索**：适合语义理解更重要的场景（如客服对话）

### Q4: 如何更新知识库？

```python
# 方法1: 重新索引（覆盖）
rag.index_documents("./new_documents")

# 方法2: 删除旧数据后重建
rag.vector_store.delete_collection()
rag.index_documents("./new_documents")
```

---

## 十、最佳实践

1. **文档预处理**：索引前清洗文档，去除无关内容
2. **合理分块**：根据文档特点选择合适的分块参数
3. **定期更新**：知识库变化时及时重新索引
4. **监控指标**：关注缓存命中率、查询延迟等指标
5. **错误处理**：做好异常捕获和降级策略

---

## 十一、小结

`main.py` 作为 RAG 系统的入口，整合了所有核心功能：

- ✅ 文档加载与分块
- ✅ 向量存储与管理
- ✅ 混合检索（BM25 + 向量 + Rerank）
- ✅ 多 LLM 提供商支持
- ✅ 对话记忆管理
- ✅ 命令行交互界面
- ✅ 生产级监控和缓存

掌握了这个模块，你就理解了整个 RAG 系统的工作流程！
