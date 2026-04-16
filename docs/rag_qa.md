# RAG 问答模块 (rag_qa.py)

## 一、模块概述

`rag_qa.py` 是 RAG 系统的**问答核心**，负责将检索到的文档与用户问题结合，调用大语言模型（LLM）生成准确的回答。

> **核心价值**：让 AI 基于真实文档回答问题，减少幻觉，提供可追溯的来源。

---

## 二、核心概念

### 2.1 RAG 问答流程

```
┌─────────────────────────────────────────────────────────────┐
│                     RAG 问答流程                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  用户问题: "什么是机器学习？"                                 │
│         ↓                                                   │
│  ┌─────────────┐                                            │
│  │   检索器    │ → 检索相关文档                              │
│  └──────┬──────┘                                            │
│         ↓                                                   │
│  相关文档: ["机器学习是AI的子集...", "机器学习包括..."]       │
│         ↓                                                   │
│  ┌─────────────┐                                            │
│  │  构建提示词  │ → 整合文档和问题                            │
│  └──────┬──────┘                                            │
│         ↓                                                   │
│  提示词: "根据以下内容回答问题...\n文档内容...\n问题: ..."    │
│         ↓                                                   │
│  ┌─────────────┐                                            │
│  │     LLM     │ → 生成回答                                  │
│  └──────┬──────┘                                            │
│         ↓                                                   │
│  回答: "机器学习是人工智能的一个子集，它使系统能够..."         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 提示词构建

```python
提示词模板 = """
Based on the following context, please answer the question.
If the context doesn't contain relevant information, say 
"I don't have enough information to answer this question."

Context:
{context}

Question: {question}

Please provide a clear and concise answer:
"""
```

### 2.3 Temperature 参数

控制回答的随机性：

| 值 | 特点 | 适用场景 |
|----|------|----------|
| 0.0-0.3 | 保守、一致 | 事实问答、技术文档 |
| 0.4-0.7 | 平衡 | 一般对话 |
| 0.8-1.0 | 创造、多样 | 创意写作 |

---

## 三、核心类：RAGQA

### 3.1 类定义

```python
class RAGQA:
    """RAG Question Answering system with multiple LLM provider support."""

    def __init__(self,
                 retriever,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.7,
                 use_conversation: bool = False,
                 system_prompt: Optional[str] = None,
                 optimize_query: bool = False,
                 llm_provider: str = "auto"):
        """
        Initialize RAG QA system.

        Args:
            retriever: 向量存储检索器
            model_name: LLM 模型名称
            temperature: 生成温度 (0-1)
            use_conversation: 是否启用对话记忆
            system_prompt: 自定义系统提示词
            optimize_query: 是否启用查询优化
            llm_provider: LLM 提供商
        """
```

### 3.2 LLM 提供商优先级

```
llm_provider = "auto" 时的自动选择顺序:

1. OpenAI (需要 OPENAI_API_KEY)
      ↓ 不可用
2. DashScope/通义千问 (需要 DASHSCOPE_API_KEY)
      ↓ 不可用
3. 本地模型 (需要 USE_LOCAL_MODEL=true)
      ↓ 不可用
4. 演示模式 (Demo Mode)
```

### 3.3 支持的 LLM 提供商

| 值 | 说明 | 配置要求 |
|----|------|----------|
| `auto` | 自动选择 | 按优先级尝试 |
| `openai` | OpenAI GPT | OPENAI_API_KEY |
| `dashscope` | 阿里云通义千问 | DASHSCOPE_API_KEY |
| `local` | 本地模型 | USE_LOCAL_MODEL=true |
| `demo` | 演示模式 | 无需配置 |

---

## 四、核心方法详解

### 4.1 ask() - 提问方法

```python
def ask(self, question: str) -> Dict[str, Any]:
    """
    Ask a question and get an answer with sources.

    Args:
        question: 用户问题

    Returns:
        包含答案和元数据的字典
    """
```

**返回值结构：**

```python
{
    "answer": "AI 生成的回答...",
    "sources": [Document, Document, ...],
    "question": "原始问题",
    "original_question": "原始问题",
    "search_question": "优化后的搜索问题",
    "query_was_optimized": True/False,
    "llm_type": "dashscope"
}
```

### 4.2 ask_with_sources() - 带详细来源

```python
def ask_with_sources(self, question: str, k: int = 3) -> Dict[str, Any]:
    """
    Ask a question and include detailed source information.

    Args:
        question: 问题
        k: 最大来源数量

    Returns:
        包含格式化来源的字典
    """
```

**返回值结构：**

```python
{
    "answer": "回答内容...",
    "sources": [Document, ...],
    "formatted_sources": [
        {
            "source_id": 1,
            "content": "内容预览...",
            "full_content": "完整内容",
            "metadata": {"source": "文件路径"}
        }
    ]
}
```

### 4.3 对话记忆管理

```python
def clear_memory(self):
    """清除对话记忆"""

def get_memory_contents(self) -> Dict[str, Any]:
    """获取对话历史"""
```

---

## 五、对话记忆功能

### 5.1 启用对话记忆

```python
qa = RAGQA(
    retriever=retriever,
    use_conversation=True  # 启用
)
```

### 5.2 工作原理

```
第1轮:
用户: 什么是AI？
系统: AI是人工智能的缩写...

第2轮:
用户: 它有哪些应用？
                    ↑
系统知道"它"指AI（从上下文推断）
系统: AI的应用包括图像识别、自然语言处理...
```

### 5.3 对话历史限制

系统默认保留最近 3 轮对话，避免上下文过长：

```python
# 内部实现
for h in self.conversation_history[-3:]:  # 只取最近3轮
    ...
```

---

## 六、使用示例

### 示例 1：基本问答

```python
from rag_qa import RAGQA
from vector_store import VectorStore

# 加载向量存储
vs = VectorStore()
vs.load_existing()

# 创建问答系统
qa = RAGQA(
    retriever=vs.get_retriever(),
    model_name="qwen-plus",
    llm_provider="dashscope"
)

# 提问
result = qa.ask("什么是机器学习？")
print(result["answer"])
```

### 示例 2：多轮对话

```python
qa = RAGQA(
    retriever=retriever,
    use_conversation=True
)

# 第一轮
r1 = qa.ask("什么是深度学习？")
print(f"回答: {r1['answer']}")

# 第二轮（系统记得上下文）
r2 = qa.ask("它和机器学习有什么关系？")
print(f"回答: {r2['answer']}")

# 清除记忆
qa.clear_memory()
```

### 示例 3：带来源显示

```python
result = qa.ask_with_sources("RAG系统有什么优势？", k=3)

print(f"回答: {result['answer']}")
print("\n参考来源:")

for source in result.get('formatted_sources', []):
    print(f"\n[来源 {source['source_id']}]")
    print(f"文件: {source['metadata'].get('source')}")
    print(f"内容: {source['content']}")
```

### 示例 4：自定义提示词

```python
custom_prompt = """你是一个专业的技术文档助手。
请根据以下文档内容回答问题。
回答要准确、专业，并在最后标注信息来源。

文档内容:
{context}

问题: {question}

请回答:"""

qa = RAGQA(
    retriever=retriever,
    system_prompt=custom_prompt,
    temperature=0.3
)
```

### 示例 5：批量问答

```python
questions = [
    "什么是RAG？",
    "RAG有什么优势？",
    "如何实现RAG系统？"
]

for question in questions:
    result = qa.ask(question)
    print(f"\nQ: {question}")
    print(f"A: {result['answer'][:150]}...")
    print(f"LLM: {result['llm_type']}")
```

---

## 七、演示模式

### 7.1 什么是演示模式？

当没有配置任何 LLM API Key 时，系统自动进入演示模式。

### 7.2 演示模式特点

- 不需要任何 API Key
- 仍然会检索相关文档
- 返回预设的模拟回答
- 适合测试和学习

### 7.3 演示模式输出

```python
# 演示模式的回答
"[Demo Mode] This is a simulated response.

To get real AI responses:
1. Get an OpenAI API key from https://platform.openai.com/
2. Add it to your .env file: OPENAI_API_KEY=your_key_here

The retrieval system is working correctly -
you can see the relevant document sources below."
```

### 7.4 检测当前模式

```python
info = qa.get_llm_info()

if info['llm_type'] == 'demo':
    print("当前使用演示模式")
elif info['llm_type'] == 'dashscope':
    print("当前使用阿里云通义千问")
elif info['llm_type'] == 'openai':
    print("当前使用 OpenAI")
```

---

## 八、配置说明

### 8.1 环境变量

| 变量名 | 说明 |
|--------|------|
| `OPENAI_API_KEY` | OpenAI API Key |
| `DASHSCOPE_API_KEY` | 阿里云 DashScope API Key |
| `USE_LOCAL_MODEL` | 是否使用本地模型 |

### 8.2 .env 文件示例

```env
# OpenAI
OPENAI_API_KEY=sk-xxx

# 阿里云通义千问（推荐中文用户）
DASHSCOPE_API_KEY=sk-xxx

# 本地模型
USE_LOCAL_MODEL=false
```

---

## 九、错误处理

```python
from rag_qa import RAGQA

qa = RAGQA(retriever=retriever)

try:
    result = qa.ask("问题")

    if 'error' in result:
        print(f"发生错误: {result['error']}")
    else:
        print(result['answer'])

except Exception as e:
    print(f"系统错误: {e}")
```

---

## 十、最佳实践

### 10.1 Temperature 设置

| 应用场景 | 推荐 Temperature |
|----------|------------------|
| 事实问答 | 0.0 - 0.3 |
| 技术文档 | 0.2 - 0.4 |
| 一般对话 | 0.5 - 0.7 |
| 创意写作 | 0.8 - 1.0 |

### 10.2 检索数量设置

```python
# 简单问题: k=2-3
result = qa.ask_with_sources("定义是什么？", k=2)

# 复杂问题: k=4-6
result = qa.ask_with_sources("详细解释原理", k=5)

# 全面回答: k=8-10
result = qa.ask_with_sources("请全面分析", k=10)
```

### 10.3 LLM 选择建议

| 场景 | 推荐模型 |
|------|----------|
| 中文问答 | 通义千问 qwen-plus |
| 英文问答 | GPT-3.5-turbo / GPT-4 |
| 复杂推理 | 通义千问 qwen-max / GPT-4 |
| 快速响应 | 通义千问 qwen-turbo |

---

## 十一、小结

`rag_qa.py` 是 RAG 系统的问答核心：

- ✅ 多 LLM 提供商支持（OpenAI、通义千问、本地）
- ✅ 对话记忆管理
- ✅ 查询优化集成
- ✅ 演示模式降级
- ✅ 来源追溯

掌握了这个模块，就理解了 RAG 系统如何生成高质量回答！
