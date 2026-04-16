# 向量存储模块 (vector_store.py)

## 一、模块概述

`vector_store.py` 是 RAG 系统的**记忆中心**，负责将文档转换为向量并存储，支持高效的语义检索。可以把它理解为系统的"图书馆"，但这个图书馆不是按书名排列，而是按"意思"排列。

> **核心价值**：让计算机理解"意思相近"的内容，而不是只匹配"字面相同"。

---

## 二、核心概念

### 2.1 什么是向量嵌入？

**向量嵌入**是将文本转换为数字数组的过程：

```
"人工智能很有趣" → [0.12, -0.34, 0.56, 0.78, ...]
```

**为什么需要向量？**

```
传统搜索（关键词匹配）:
  查询: "小狗"
  文档: "犬类宠物护理" ❌ 不匹配（字面不同）

向量搜索（语义匹配）:
  查询: "小狗" → [0.1, 0.2, 0.3]
  文档: "犬类宠物护理" → [0.11, 0.19, 0.31]
  相似度: 0.98 ✅ 匹配（语义相近）
```

### 2.2 向量空间示意

```
                    ┌─────────────────────────────────────┐
                    │          向量空间                    │
                    │                                     │
                    │    🐕 宠物相关                       │
                    │       ↗︎                            │
                    │      /                              │
                    │     /  🐱 猫咪                       │
                    │    /                               │
                    │   🐕 小狗 ───────── 🚗 汽车          │
                    │              不同主题                │
                    │                                     │
                    │ 距离近 = 语义相似                    │
                    │ 距离远 = 语义不同                    │
                    └─────────────────────────────────────┘
```

### 2.3 ChromaDB 简介

ChromaDB 是一个开源的向量数据库，专为 AI 应用设计。

**优点：**
- 轻量级，无需额外服务器
- 支持持久化存储
- 与 LangChain 无缝集成
- 支持元数据过滤

---

## 三、核心类：VectorStore

### 3.1 类定义

```python
class VectorStore:
    """Vector store manager for RAG system."""
```

### 3.2 初始化参数

```python
def __init__(self,
             persist_directory: str = "./chroma_db",
             embedding_model: str = "all-MiniLM-L6-v2"):
    """
    Initialize vector store with embedding model.

    Args:
        persist_directory: 向量数据库持久化目录
        embedding_model: 嵌入模型名称
    """
```

**参数详解：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `persist_directory` | `./chroma_db` | 向量数据保存位置 |
| `embedding_model` | `all-MiniLM-L6-v2` | 嵌入模型 |

### 3.3 嵌入模型优先级

系统自动尝试以下嵌入模型，直到成功：

```
1. DashScope 嵌入（中文最佳，需要 API Key）
      ↓ 失败
2. HuggingFace 本地模型
      ↓ 失败
3. FakeEmbeddings（降级模式，仅供测试）
```

---

## 四、核心方法详解

### 4.1 创建向量存储

```python
def create_from_documents(self,
                          documents: List[Document],
                          collection_name: str = "rag_collection") -> None:
    """
    Create vector store from documents.

    Args:
        documents: Document 对象列表
        collection_name: 集合名称
    """
```

**执行流程：**

```
Documents ──→ 预处理 ──→ 生成嵌入向量 ──→ 存储到 ChromaDB
    │
    ├── 清理旧数据
    ├── 调用嵌入模型
    └── 持久化到磁盘
```

### 4.2 加载已有存储

```python
def load_existing(self, collection_name: str = "rag_collection") -> bool:
    """
    Load existing vector store from disk.

    Args:
        collection_name: 要加载的集合名称

    Returns:
        True 加载成功，False 加载失败
    """
```

### 4.3 相似度搜索

```python
def similarity_search(self,
                      query: str,
                      k: int = 4) -> List[Document]:
    """
    Search for similar documents.

    Args:
        query: 搜索查询
        k: 返回结果数量

    Returns:
        相似文档列表
    """
```

**搜索原理：**

```
查询: "什么是机器学习？"
    │
    ▼
转换为向量: [0.15, -0.23, 0.41, ...]
    │
    ▼
计算与所有文档的距离（余弦相似度）
    │
    ▼
返回距离最近的 k 个文档
```

### 4.4 带分数的搜索

```python
def similarity_search_with_scores(self,
                                   query: str,
                                   k: int = 4) -> List[tuple]:
    """
    Search with similarity scores.

    Returns:
        List of (document, score) tuples
    """
```

**分数含义：**

| 分数范围 | 含义 |
|----------|------|
| > 0.7 | 高度相关 |
| 0.4 - 0.7 | 部分相关 |
| < 0.4 | 相关性较低 |

### 4.5 获取检索器

```python
def get_retriever(self, search_kwargs: Optional[dict] = None):
    """
    Get a retriever for use with LangChain chains.

    Args:
        search_kwargs: 搜索参数 {"k": 4}
    """
```

### 4.6 获取混合检索器

```python
def get_hybrid_retriever(self,
                         k: int = 4,
                         alpha: float = 0.5,
                         candidate_k: int = 20,
                         use_rerank: bool = True) -> Any:
    """
    Get a hybrid retriever combining BM25 and vector search.

    Args:
        k: 最终返回文档数
        alpha: 向量搜索权重（仅 RRF 模式）
        candidate_k: 每路召回的候选数
        use_rerank: 是否使用重排序
    """
```

---

## 五、混合检索架构

### 5.1 为什么需要混合检索？

```
纯向量搜索的问题:
  查询: "Python 3.11"
  可能错过: "Python 3.10 与 3.11 版本差异"（版本号精确匹配差）

纯关键词搜索的问题:
  查询: "如何养小狗"
  可能错过: "幼犬护理指南"（语义相近但字面不同）

混合检索 = 两者的优势结合
```

### 5.2 混合检索流程

```
                    ┌─────────────────────────────────────┐
                    │           用户查询                   │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────┴───────────────────┐
                    │                                     │
                    ▼                                     ▼
            ┌───────────────┐                   ┌───────────────┐
            │   BM25 检索    │                   │   向量检索     │
            │  (关键词匹配)   │                   │  (语义匹配)    │
            └───────┬───────┘                   └───────┬───────┘
                    │                                   │
                    │  候选20篇                          │  候选20篇
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                            ┌───────────────────┐
                            │      合并去重      │
                            │   (约30篇候选)     │
                            └─────────┬─────────┘
                                      │
                                      ▼
                            ┌───────────────────┐
                            │   Rerank 重排序    │
                            │  (语义精排 top10)  │
                            └─────────┬─────────┘
                                      │
                                      ▼
                            ┌───────────────────┐
                            │    返回最终结果    │
                            └───────────────────┘
```

---

## 六、使用示例

### 示例 1：基本使用

```python
from vector_store import VectorStore
from document_loader import DocumentLoader

# 1. 加载文档
loader = DocumentLoader(chunk_size=500)
documents = loader.load_text("./my_doc.txt")

# 2. 创建向量存储
vs = VectorStore(persist_directory="./my_db")
vs.create_from_documents(documents)

# 3. 搜索
results = vs.similarity_search("人工智能的发展", k=3)
for doc in results:
    print(doc.page_content[:100])
```

### 示例 2：加载已有存储

```python
from vector_store import VectorStore

vs = VectorStore(persist_directory="./my_db")

if vs.load_existing():
    print("加载成功！")

    # 查看信息
    info = vs.get_collection_info()
    print(f"文档数: {info['document_count']}")
    print(f"模型: {info['embedding_model']}")
```

### 示例 3：混合检索

```python
# 获取混合检索器
retriever = vs.get_hybrid_retriever(
    k=10,              # 最终返回10篇
    candidate_k=20,    # 每路召回20篇
    use_rerank=True    # 使用 Rerank
)

# 执行检索
results = retriever.invoke("什么是深度学习？")
```

### 示例 4：带分数搜索

```python
# 带相似度分数的搜索
results = vs.similarity_search_with_scores("机器学习", k=5)

for doc, score in results:
    print(f"相似度: {score:.4f}")
    print(f"内容: {doc.page_content[:80]}...")
    print()

# 筛选高相似度结果
high_score = [(doc, score) for doc, score in results if score > 0.7]
```

---

## 七、嵌入模型详解

### 7.1 DashScope 嵌入（推荐中文使用）

```python
# 自动启用条件：
# 1. 安装了 dashscope 包
# 2. 设置了 DASHSCOPE_API_KEY 环境变量

# 特点：
# - 中文效果最佳
# - API 调用，无需本地资源
# - 1536 维向量
```

### 7.2 HuggingFace 嵌入

```python
# 默认模型: all-MiniLM-L6-v2
# 特点：
# - 本地运行，无 API 调用
# - 384 维向量
# - 英文效果好，中文一般

# 推荐中文模型:
# - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# - shibing624/text2vec-base-chinese
```

### 7.3 网络问题解决方案

```python
# 设置 HuggingFace 镜像（中国用户）
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 或在命令行设置：
# Windows: set HF_ENDPOINT=https://hf-mirror.com
# Linux/Mac: export HF_ENDPOINT=https://hf-mirror.com
```

---

## 八、存储结构

### 8.1 目录结构

```
./chroma_db/                    # 持久化目录
├── chroma.sqlite3              # SQLite 数据库
│   ├── embeddings 表           # 向量数据
│   ├── documents 表            # 原文内容
│   └── metadata 表             # 元数据
└── index/                      # 向量索引文件
```

### 8.2 数据存储格式

```python
# 每个文档存储的信息
{
    "id": "unique-id-123",           # 唯一标识
    "embedding": [0.1, 0.2, ...],    # 向量 (384/1536维)
    "document": "原文内容...",        # 文本
    "metadata": {                    # 元数据
        "source": "file.txt",
        "file_type": ".txt"
    }
}
```

---

## 九、性能优化

### 9.1 批量处理

```python
# 大规模数据时，系统内部已优化批量处理
vs.create_from_documents(documents)  # 自动分批
```

### 9.2 搜索调优

```python
# 精确匹配场景
results = vs.similarity_search(query, k=4)

# 广泛探索场景
results = vs.similarity_search(query, k=20)

# 带分数过滤
results_with_scores = vs.similarity_search_with_scores(query, k=10)
filtered = [(doc, score) for doc, score in results_with_scores if score > 0.7]
```

---

## 十、错误处理

### 10.1 常见错误

| 错误 | 原因 | 解决方案 |
|------|------|----------|
| `Model not found` | 网络问题/模型不存在 | 检查网络或使用镜像 |
| `Collection not found` | 集合不存在 | 先创建或检查路径 |
| `API key not set` | 未配置密钥 | 设置环境变量 |

### 10.2 错误处理示例

```python
from vector_store import VectorStore

try:
    vs = VectorStore()
    vs.create_from_documents(docs)
except ValueError as e:
    print(f"参数错误: {e}")
except RuntimeError as e:
    print(f"运行时错误: {e}")
```

---

## 十一、最佳实践

### 11.1 模型选择

| 场景 | 推荐模型 | 维度 |
|------|----------|------|
| 中文为主 | DashScope text-embedding-v2 | 1536 |
| 英文为主 | all-MiniLM-L6-v2 | 384 |
| 多语言 | paraphrase-multilingual-MiniLM-L12-v2 | 384 |
| 离线环境 | 本地 HuggingFace 模型 | - |

### 11.2 检索数量调优

```python
# 一般问答: k=3-5
retriever = vs.get_retriever(search_kwargs={"k": 4})

# 复杂问题: k=6-10
retriever = vs.get_retriever(search_kwargs={"k": 8})

# 简单查询: k=2-3
retriever = vs.get_retriever(search_kwargs={"k": 2})
```

---

## 十二、架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                        VectorStore                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入层                        存储层             输出层        │
│  ┌──────────┐               ┌──────────┐       ┌──────────┐   │
│  │ Documents │───嵌入──────→│ ChromaDB │───检索──→│ Results  │   │
│  │  (文本)   │               │ (向量库) │         │ (文档)   │   │
│  └──────────┘               └──────────┘       └──────────┘   │
│       │                           │                   │        │
│       ▼                           ▼                   ▼        │
│  ┌──────────┐               ┌──────────┐       ┌──────────┐   │
│  │ Embeddings│              │ Metadata │       │Retriever │   │
│  │  (向量)   │              │ (元数据) │       │(检索器)  │   │
│  └──────────┘               └──────────┘       └──────────┘   │
│       │                                               │        │
│       ├─ DashScope (中文最佳)                         │        │
│       ├─ HuggingFace (通用)                           │        │
│       └─ FakeEmbeddings (降级)                        │        │
│                                                      │        │
│                                           ┌──────────┴──────┐ │
│                                           │                  │ │
│                                      Vector Retriever  Hybrid │
│                                                        Retriever│
└─────────────────────────────────────────────────────────────────┘
```

---

## 十三、小结

`vector_store.py` 是 RAG 系统的核心存储模块：

- ✅ 多种嵌入模型支持（DashScope/HuggingFace/降级）
- ✅ 持久化向量存储（ChromaDB）
- ✅ 高效语义搜索
- ✅ 混合检索支持
- ✅ LangChain 兼容接口

理解向量存储，就理解了 RAG 系统如何"记住"知识！
