# 问题优化模块 (query_optimizer.py)

## 一、模块概述

`query_optimizer.py` 提供 RAG 系统的**智能问题优化**能力，使用大语言模型将用户模糊、简短或指代不明的问题改写得更清晰、更具体，从而提高检索准确性。

> **核心价值**：让模糊问题变清晰，让简短问题变具体，大幅提升检索质量和回答准确性。

---

## 二、为什么需要问题优化？

### 2.1 用户问题的常见问题

| 问题类型 | 示例 | 问题 |
|----------|------|------|
| 指代不明 | "它是什么？" | 不知道"它"指什么 |
| 过于简短 | "怎么用" | 缺乏上下文 |
| 表述模糊 | "有什么好处" | 不知道问什么的好处 |
| 信息不足 | "怎么提高" | 不知道要提高什么 |

### 2.2 优化效果示例

```
┌─────────────────────────────────────────────────────────────┐
│                     问题优化示例                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户输入: "它是什么？"                                       │
│  优化后: "人工智能（AI）是什么？它的定义和主要特征是什么？"     │
│                                                             │
│  用户输入: "怎么用"                                          │
│  优化后: "如何使用RAG系统？请介绍具体的使用方法和步骤"          │
│                                                             │
│  用户输入: "有什么好处"                                       │
│  优化后: "RAG系统有哪些优点和好处？请列举主要优势"              │
│                                                             │
│  用户输入: "如何提高效果"                                     │
│  优化后: "如何提高RAG系统的检索效果和回答质量？"                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 工作流程

```
┌─────────────────────────────────────────────────────────────┐
│                     问题优化流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户输入: "它是什么？"                                       │
│         ↓                                                   │
│  ┌─────────────────┐                                        │
│  │ 检查问题长度    │ → 太短（<5字符）则不优化                │
│  └────────┬────────┘                                        │
│           ↓                                                 │
│  ┌─────────────────┐                                        │
│  │ 构建优化提示词   │                                        │
│  │ + 对话上下文    │ (如果有)                                │
│  └────────┬────────┘                                        │
│           ↓                                                 │
│  ┌─────────────────┐                                        │
│  │   LLM 优化      │                                        │
│  └────────┬────────┘                                        │
│           ↓                                                 │
│  优化后: "人工智能（AI）是什么？它的定义和主要特征是什么？"     │
│           ↓                                                 │
│  ┌─────────────────┐                                        │
│  │ 使用优化后问题   │                                        │
│  │ 进行检索        │                                        │
│  └─────────────────┘                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 三、优化策略

### 3.1 QueryOptimizer - 问题优化

将问题改写得更清晰、更具体：

```python
class QueryOptimizer:
    """
    使用 LLM 优化用户问题

    特点:
    - 保持原意
    - 补充上下文
    - 添加关键词
    - 使用较低 Temperature 保证稳定
    """
```

### 3.2 QueryExpander - 问题扩展

生成多个问题变体，扩大检索范围：

```python
class QueryExpander:
    """
    生成问题的多个变体

    用途:
    - 多角度检索
    - 提高召回率
    - 合并多个检索结果
    """
```

### 3.3 策略对比

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| QueryOptimizer | 改写得清晰具体 | 问题模糊、指代不明 |
| QueryExpander | 生成多个变体 | 需要全面检索 |

---

## 四、QueryOptimizer 类详解

### 4.1 类定义

```python
class QueryOptimizer:
    """使用 LLM 优化用户问题，使检索更准确"""

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.3,
                 enabled: bool = True):
        """
        初始化问题优化器

        Args:
            model_name: LLM 模型名称
            temperature: 温度参数（建议 0.1-0.5，保证稳定）
            enabled: 是否启用优化
        """
```

### 4.2 核心方法

#### optimize() - 优化问题

```python
def optimize(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    优化用户问题

    Args:
        question: 原始问题
        context: 对话上下文（可选，用于解析代词指代）

    Returns:
        {
            'original_question': '原始问题',
            'optimized_question': '优化后的问题',
            'was_optimized': True/False,
            'optimizer_status': 'openai' | 'disabled'
        }
    """
```

#### is_enabled() - 检查状态

```python
def is_enabled(self) -> bool:
    """检查优化器是否可用"""
    return self.enabled and self.llm is not None
```

#### get_status() - 获取状态

```python
def get_status(self) -> Dict[str, Any]:
    """获取优化器状态信息"""
    return {
        "enabled": self.enabled,
        "llm_type": self._llm_type,
        "model_name": self.model_name,
        "temperature": self.temperature
    }
```

### 4.3 自动降级

当 LLM 不可用时，优化器自动禁用：

```python
optimizer = QueryOptimizer(enabled=True)

# 没有 API Key 时
result = optimizer.optimize("它是什么？")
# result["optimized_question"] = "它是什么？"（原样返回）
# result["was_optimized"] = False
```

---

## 五、QueryExpander 类详解

### 5.1 类定义

```python
class QueryExpander:
    """生成问题的多个变体，用于更全面的检索"""

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.5,
                 num_variations: int = 3):
        """
        初始化问题扩展器

        Args:
            model_name: LLM 模型名称
            temperature: 温度参数（建议 0.3-0.7，允许一定多样性）
            num_variations: 变体数量
        """
```

### 5.2 expand() 方法

```python
def expand(self, question: str) -> Dict[str, Any]:
    """
    生成问题的多个变体

    Returns:
        {
            'original_question': '原始问题',
            'expanded_queries': ['原始', '变体1', '变体2', ...],
            'was_expanded': True/False
        }
    """
```

---

## 六、使用示例

### 示例 1：在 RAG 系统中启用

```python
from rag_qa import RAGQA
from vector_store import VectorStore

# 准备检索器
vector_store = VectorStore()
vector_store.load_existing()
retriever = vector_store.get_retriever()

# 创建问答系统，启用问题优化
qa = RAGQA(
    retriever=retriever,
    model_name="qwen-plus",
    optimize_query=True  # 启用问题优化
)

# 提问
result = qa.ask("它是什么？")

print(f"原始问题: {result['original_question']}")
print(f"搜索问题: {result['search_question']}")
print(f"是否优化: {result['query_was_optimized']}")
print(f"回答: {result['answer']}")
```

### 示例 2：命令行使用

```bash
# 启用问题优化的交互模式
python main.py --mode interactive --optimize-query

# 启用问题优化的单次查询
python main.py --mode query --question "它是什么" --optimize-query

# 运行演示
python main.py --mode demo --optimize-query
```

### 示例 3：多轮对话优化

```python
from rag_qa import RAGQA

qa = RAGQA(
    retriever=retriever,
    use_conversation=True,    # 启用对话记忆
    optimize_query=True       # 启用问题优化
)

# 第一轮
r1 = qa.ask("什么是RAG？")
print(f"问题优化为: {r1['search_question']}")

# 第二轮（优化器会利用对话上下文）
r2 = qa.ask("它有什么优点？")  # "它"会被解析为"RAG"
print(f"问题优化为: {r2['search_question']}")
# 输出: "RAG（检索增强生成）系统有哪些优点和好处？"
```

### 示例 4：独立使用优化器

```python
from query_optimizer import QueryOptimizer

# 创建优化器
optimizer = QueryOptimizer(
    model_name="qwen-plus",
    temperature=0.3
)

# 测试问题
test_questions = [
    "它是什么？",
    "怎么用",
    "有什么好处",
    "如何提高效率",
    "能不能讲一下"
]

print("问题优化测试:\n")
for q in test_questions:
    result = optimizer.optimize(q)
    print(f"原始: {q}")
    if result['was_optimized']:
        print(f"优化: {result['optimized_question']}")
    else:
        print("(未优化)")
    print()
```

### 示例 5：使用问题扩展

```python
from query_optimizer import QueryExpander

expander = QueryExpander(num_variations=3)

result = expander.expand("如何学习AI")

print("问题变体:")
for i, query in enumerate(result['expanded_queries'], 1):
    print(f"{i}. {query}")

# 可以用这些变体进行多次检索
all_docs = []
for query in result['expanded_queries']:
    docs = retriever.invoke(query)
    all_docs.extend(docs)

# 去重并排序
# ...
```

---

## 七、优化提示词

### 7.1 默认提示词

```python
优化提示词 = """你是一个问题优化专家。你的任务是将用户的问题优化得更加清晰、具体和易于检索。

优化规则:
1. 保持原问题的核心意图不变
2. 补充必要的上下文信息（如果有对话历史）
3. 使问题更加具体和明确
4. 添加相关的关键词以提高检索准确性
5. 如果原问题已经很清晰，可以直接返回原问题
6. 只返回优化后的问题，不要添加任何解释

示例:
用户输入: "它是什么？"
优化后: "人工智能（AI）是什么？它的定义和主要特征是什么？"

用户输入: "怎么用"
优化后: "如何使用RAG系统？请介绍具体的使用方法和步骤"
"""
```

### 7.2 自定义提示词

```python
class CustomQueryOptimizer(QueryOptimizer):
    def _build_optimization_prompt(self, question, context=None):
        return f"""你是一个技术文档助手。
请将用户的问题改写得更具体、更技术化。

原始问题: {question}
优化后:"""
```

---

## 八、Temperature 设置建议

| 值 | 特点 | 适用场景 |
|----|------|----------|
| 0.1-0.3 | 非常稳定 | 问题优化（推荐） |
| 0.3-0.5 | 稳定 | 问题扩展 |
| 0.5-0.7 | 一定多样性 | 创意场景 |

```python
# 问题优化：使用低 Temperature
optimizer = QueryOptimizer(temperature=0.3)

# 问题扩展：使用中等 Temperature
expander = QueryExpander(temperature=0.5)
```

---

## 九、最佳实践

### 9.1 何时启用问题优化

| 场景 | 是否推荐 | 原因 |
|------|----------|------|
| 多轮对话 | 推荐 | 可以解析代词指代 |
| 用户问题简短 | 推荐 | 补充上下文信息 |
| 专业领域问答 | 推荐 | 添加专业术语 |
| 单次精确查询 | 可选 | 可能不需要优化 |
| 实时性要求高 | 可选 | 会增加 LLM 调用延迟 |

### 9.2 性能考虑

```
无优化: 用户问题 → 检索 → 生成回答 (1次LLM调用)
有优化: 用户问题 → 优化 → 检索 → 生成回答 (2次LLM调用)

优化增加的延迟: 0.5-2 秒
```

**建议：**
- 重要查询：启用优化
- 简单查询：可禁用优化
- 使用更快的模型（如 qwen-turbo）进行优化

### 9.3 检查优化器状态

```python
optimizer = QueryOptimizer(enabled=True)

if not optimizer.is_enabled():
    print("警告: 问题优化器不可用，将使用原始问题")
```

---

## 十、常见问题

### Q1: 问题优化需要额外的 API Key 吗？

不需要额外配置。问题优化使用与主问答系统相同的 LLM API Key。

### Q2: 问题优化会增加多少延迟？

通常增加 0.5-2 秒的延迟，取决于模型和网络。对于模糊问题，这个延迟是值得的。

### Q3: 所有问题都会被优化吗？

不是。以下情况不会优化：
- 问题太短（<5 个字符）
- 问题已经很清晰
- LLM 返回与原问题相同的内容

### Q4: 可以只对某些问题进行优化吗？

```python
def should_optimize(question: str) -> bool:
    """判断是否需要优化"""
    # 包含代词的问题
    if any(word in question for word in ['它', '这个', '那个', '怎么', '如何']):
        return True
    # 过短的问题
    if len(question) < 10:
        return True
    return False

# 使用
if should_optimize(question):
    result = optimizer.optimize(question)
    search_question = result['optimized_question']
else:
    search_question = question
```

### Q5: 如何关闭问题优化？

```python
# 方法 1: 初始化时禁用
qa = RAGQA(retriever=retriever, optimize_query=False)

# 方法 2: 命令行不使用参数
python main.py --mode interactive  # 不加 --optimize-query
```

---

## 十一、小结

`query_optimizer.py` 为 RAG 系统提供智能问题优化：

- ✅ 问题改写优化
- ✅ 问题扩展变体
- ✅ 对话上下文利用
- ✅ 自动降级机制
- ✅ 低延迟配置

掌握问题优化，让检索更精准，回答更准确！
