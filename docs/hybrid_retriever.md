# 混合检索模块 (hybrid_retriever.py)

## 一、模块概述

`hybrid_retriever.py` 实现了**混合检索**功能，将 BM25（关键词检索）和向量检索（语义检索）结合起来，配合 Rerank 重排序，提供更准确、更全面的检索结果。

> **核心价值**：同时利用关键词精确匹配和语义理解的优势，大幅提升检索质量。

---

## 二、为什么需要混合检索？

### 2.1 单一检索的局限性

| 检索方式 | 优点 | 缺点 |
|----------|------|------|
| **向量检索** | 理解语义、同义词、近义词 | 可能遗漏精确关键词匹配 |
| **BM25检索** | 精确关键词匹配、速度快 | 不理解语义、同义词 |

### 2.2 混合检索的优势

```
查询: "Python 3.11 新特性"

向量检索结果:
  1. "Python最新版本带来了很多改进..." (语义相关)
  2. "编程语言的版本更新..." (语义相关)

BM25检索结果:
  1. "Python 3.11 新特性包括异常组..." (精确匹配)
  2. "Python 3.10 与 3.11 性能对比..." (精确匹配)

混合检索结果:
  1. "Python 3.11 新特性包括异常组..." (BM25高分)
  2. "Python最新版本带来了很多改进..." (向量高分)
  3. "Python 3.10 与 3.11 性能对比..." (两者兼顾)
```

---

## 三、核心概念

### 3.1 BM25 算法

BM25 (Best Matching 25) 是一种基于概率的排序函数，是搜索引擎中最经典的算法之一。

**核心公式：**

```
Score(D, Q) = Σ IDF(qi) × (f(qi, D) × (k1 + 1)) / (f(qi, D) + k1 × (1 - b + b × |D|/avgdl))
```

**参数说明：**

| 参数 | 说明 | 默认值 | 调优建议 |
|------|------|--------|----------|
| `k1` | 词频饱和参数 | 1.5 | 较高值使高频词影响更大 |
| `b` | 文档长度归一化 | 0.75 | 较高值对长文档惩罚更重 |

### 3.2 Rerank 重排序

Rerank 是对检索结果的二次精排：

```
初始检索结果（20-30篇候选）
      ↓
  Rerank 模型
  (语义相关性打分)
      ↓
精排后的 Top-K 结果
```

**为什么需要 Rerank？**

- 初步检索（BM25/向量）速度快但精度有限
- Rerank 模型专门针对"查询-文档相关性"训练
- 在小规模候选集上精排，平衡速度和效果

### 3.3 Reciprocal Rank Fusion (RRF)

当 Rerank 不可用时的降级方案：

```
RRF(d) = Σ 1 / (k + rank(d))

示例:
文档A: BM25排名2, 向量排名5
RRF(A) = 1/(60+2) + 1/(60+5) = 0.0315

文档B: BM25排名1, 向量排名10
RRF(B) = 1/(60+1) + 1/(60+10) = 0.0307

最终: A > B
```

---

## 四、核心类详解

### 4.1 DashScopeReranker

阿里云 DashScope 的 Rerank 服务封装：

```python
class DashScopeReranker:
    """阿里云 DashScope Rerank 服务"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 Reranker

        Args:
            api_key: DashScope API Key
        """

    def rerank(self, query: str, documents: List[str], top_n: int = 10) -> List[Tuple[int, float]]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回前 N 个结果

        Returns:
            List of (original_index, rerank_score) tuples
        """
```

### 4.2 BM25 类

自实现的 BM25 算法，支持中英文：

```python
class BM25:
    """BM25 算法实现"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents: List[List[str]] = []
        self.idf: Dict[str, float] = {}

    def fit(self, documents: List[str]) -> None:
        """在文档集合上训练"""

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """搜索并返回 top_k 结果"""
```

### 4.3 HybridRetriever

核心混合检索器：

```python
class HybridRetriever(BaseRetriever):
    """混合检索器，结合 BM25 + 向量 + Rerank"""

    def __init__(
        self,
        documents: List[Document],
        vector_store: Any,
        k: int = 4,
        candidate_k: int = 20,
        use_rerank: bool = True,
        alpha: float = 0.5,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75
    ):
        """
        Args:
            documents: Document 对象列表
            vector_store: 向量存储实例
            k: 最终返回文档数
            candidate_k: 每路召回的候选数
            use_rerank: 是否使用 Rerank
            alpha: 向量检索权重（仅 RRF 模式）
            bm25_k1: BM25 k1 参数
            bm25_b: BM25 b 参数
        """
```

---

## 五、检索流程图

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
                    │  Top-20 候选                       │  Top-20 候选
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                            ┌───────────────────┐
                            │      合并去重      │
                            │   (约30篇候选)     │
                            └─────────┬─────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
                    ▼                                   ▼
            ┌───────────────┐                   ┌───────────────┐
            │  Rerank 可用?  │──→ No ──→        │  RRF 融合排序  │
            └───────┬───────┘                   └───────┬───────┘
                    │ Yes                               │
                    ▼                                   │
            ┌───────────────┐                          │
            │ Rerank 重排序  │                          │
            └───────┬───────┘                          │
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                            ┌───────────────────┐
                            │    返回 Top-K     │
                            └───────────────────┘
```

---

## 六、使用示例

### 示例 1：基本使用

```python
from hybrid_retriever import HybridRetriever
from vector_store import VectorStore
from document_loader import DocumentLoader

# 1. 准备数据
loader = DocumentLoader()
documents = loader.load_directory("./docs/")

# 2. 创建向量存储
vs = VectorStore()
vs.create_from_documents(documents)

# 3. 获取混合检索器
retriever = vs.get_hybrid_retriever(
    k=10,              # 最终返回10篇
    candidate_k=20,    # 每路召回20篇
    use_rerank=True    # 使用 Rerank
)

# 4. 执行检索
results = retriever.invoke("什么是机器学习？")
for doc in results:
    print(doc.page_content[:100])
```

### 示例 2：查看详细分数

```python
# 获取带分数的搜索结果
results = retriever.search_with_scores("深度学习原理", k=5)

for doc, scores in results:
    print(f"内容: {doc.page_content[:50]}...")
    print(f"  BM25 分数: {scores.get('bm25_score', 0):.4f}")
    print(f"  向量分数: {scores.get('vector_score', 0):.4f}")
    print(f"  Rerank 分数: {scores.get('rerank_score', 'N/A')}")
    print()
```

### 示例 3：与 RAGQA 集成

```python
from rag_qa import RAGQA

# 使用混合检索器创建问答系统
qa = RAGQA(
    retriever=retriever,
    model_name="qwen-plus",
    temperature=0.7
)

# 提问
result = qa.ask("Python 有哪些优势？")
print(result["answer"])
```

### 示例 4：仅使用 RRF（无 Rerank）

```python
# 适用于没有 DashScope API Key 的场景
retriever = vs.get_hybrid_retriever(
    k=10,
    candidate_k=20,
    use_rerank=False,  # 禁用 Rerank
    alpha=0.5          # RRF 权重
)

results = retriever.invoke("查询内容")
```

---

## 七、参数调优指南

### 7.1 检索数量参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `k` | 最终返回数 | 4-10 |
| `candidate_k` | 每路召回候选数 | 20-30 |

**调优建议：**

```python
# 高精度场景：候选多，最终少
retriever = vs.get_hybrid_retriever(k=5, candidate_k=30)

# 快速响应场景：候选少
retriever = vs.get_hybrid_retriever(k=4, candidate_k=10)
```

### 7.2 BM25 参数

| 文档类型 | k1 | b |
|----------|-----|-----|
| 短文档（标题、摘要） | 1.2 | 0.5 |
| 一般文档 | 1.5 | 0.75 |
| 长文档（文章、报告） | 1.8 | 0.8 |

### 7.3 Alpha 权重（RRF 模式）

```python
# 偏向关键词匹配
retriever.set_alpha(0.3)  # BM25 权重 70%

# 平衡
retriever.set_alpha(0.5)

# 偏向语义匹配
retriever.set_alpha(0.7)  # 向量权重 70%
```

---

## 八、性能分析

### 8.1 检索延迟对比

| 方式 | 延迟 | 质量 |
|------|------|------|
| 纯向量 | ~50ms | 中 |
| 纯 BM25 | ~10ms | 中 |
| 混合（无 Rerank） | ~60ms | 高 |
| 混合 + Rerank | ~200ms | 最高 |

### 8.2 资源消耗

```
BM25: 纯内存计算，几乎无额外资源消耗
向量检索: 依赖向量数据库
Rerank: API 调用，有网络延迟
```

---

## 九、常见问题

### Q1: Rerank 初始化失败怎么办？

```python
# 检查 API Key
import os
print(os.getenv("DASHSCOPE_API_KEY"))

# 如果没有，系统会自动降级到 RRF
```

### Q2: 混合检索比纯向量慢多少？

- 无 Rerank：增加约 10-20%
- 有 Rerank：增加约 200-300%（因为 API 调用）

### Q3: 如何选择是否使用 Rerank？

| 场景 | 推荐 |
|------|------|
| 实时性要求高 | 不使用 Rerank |
| 检索质量要求高 | 使用 Rerank |
| 离线批处理 | 使用 Rerank |

---

## 十、小结

`hybrid_retriever.py` 提供了生产级的混合检索方案：

- ✅ BM25 关键词检索
- ✅ 向量语义检索
- ✅ DashScope Rerank 精排
- ✅ RRF 降级方案
- ✅ 参数可配置

掌握了混合检索，就掌握了现代搜索引擎的核心技术！
