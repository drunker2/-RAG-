# 阿里云通义千问适配器 (dashscope_llm.py)

## 一、模块概述

`dashscope_llm.py` 是阿里云通义千问（Qwen）大模型的**适配器模块**，让 RAG 系统能够使用阿里云的大模型服务。

> **核心价值**：为中文场景提供最佳的大模型支持，性价比高，无需翻墙。

---

## 二、为什么选择通义千问？

| 特点 | 说明 |
|------|------|
| **中文优化** | 专为中文场景训练，理解更准确 |
| **性价比高** | 比 GPT-4 便宜很多 |
| **国内服务** | 无需翻墙，访问稳定 |
| **多模型选择** | turbo/plus/max 满足不同需求 |

---

## 三、支持的模型

```python
SUPPORTED_MODELS = {
    "qwen-turbo": {
        "name": "通义千问-Turbo",
        "description": "快速响应模型，适合简单任务",
        "context_length": 8192,
        "recommended_for": ["快速问答", "简单对话"]
    },
    "qwen-plus": {
        "name": "通义千问-Plus",
        "description": "通用模型，性价比高",
        "context_length": 32768,
        "recommended_for": ["日常对话", "文本处理", "信息抽取"]
    },
    "qwen-max": {
        "name": "通义千问-Max",
        "description": "最强模型，适合复杂推理",
        "context_length": 32768,
        "recommended_for": ["复杂问答", "推理任务", "内容创作"]
    },
    "qwen-long": {
        "name": "通义千问-Long",
        "description": "长上下文模型",
        "context_length": 1000000,
        "recommended_for": ["长文档理解", "文档摘要"]
    }
}
```

---

## 四、核心类

### 4.1 DashScopeLLM

```python
class DashScopeLLM:
    """阿里云通义千问大模型适配器"""

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "qwen-plus",
                 temperature: float = 0.7,
                 top_p: float = 0.8,
                 max_tokens: int = 2048):
        """
        初始化通义千问模型

        Args:
            api_key: 阿里云 DashScope API Key
            model: 模型名称
            temperature: 温度参数 (0-1)
            top_p: 核采样参数 (0-1)
            max_tokens: 最大输出 token 数
        """
```

### 4.2 DashScopeEmbeddings

```python
class DashScopeEmbeddings:
    """阿里云通义千问嵌入模型适配器"""

    def __init__(self,
                 api_key: Optional[str] = None,
                 model: str = "text-embedding-v2"):
        """
        初始化嵌入模型

        Args:
            api_key: DashScope API Key
            model: 嵌入模型名称
        """

    def embed_query(self, text: str) -> List[float]:
        """将文本转换为嵌入向量"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量嵌入文档"""
```

---

## 五、配置指南

### 5.1 获取 API Key

1. 注册阿里云账号：https://www.aliyun.com/
2. 开通 DashScope 服务：https://dashscope.console.aliyun.com/
3. 创建 API Key

### 5.2 配置环境变量

```bash
# Windows CMD
set DASHSCOPE_API_KEY=your_api_key_here

# Windows PowerShell
$env:DASHSCOPE_API_KEY="your_api_key_here"

# Linux/Mac
export DASHSCOPE_API_KEY=your_api_key_here
```

### 5.3 .env 文件配置

```env
DASHSCOPE_API_KEY=sk-your-api-key-here
```

---

## 六、使用示例

### 示例 1：基本调用

```python
from dashscope_llm import DashScopeLLM

# 初始化
llm = DashScopeLLM(
    model="qwen-plus",
    temperature=0.7
)

# 调用
response = llm.invoke("什么是人工智能？")
print(response.content)
```

### 示例 2：多轮对话

```python
llm = DashScopeLLM(model="qwen-plus")

messages = [
    {"role": "user", "content": "什么是机器学习？"},
    {"role": "assistant", "content": "机器学习是人工智能的一个子集..."},
    {"role": "user", "content": "它有哪些应用？"}
]

response = llm.chat(messages)
print(response)
```

### 示例 3：流式输出

```python
llm = DashScopeLLM(model="qwen-plus")

for chunk in llm.stream_call("写一首关于春天的诗"):
    print(chunk, end="", flush=True)
```

### 示例 4：嵌入向量

```python
from dashscope_llm import DashScopeEmbeddings

embeddings = DashScopeEmbeddings()

# 单个文本
vector = embeddings.embed_query("机器学习很有趣")
print(f"向量维度: {len(vector)}")  # 1536

# 批量处理
texts = ["人工智能", "机器学习", "深度学习"]
vectors = embeddings.embed_documents(texts)
print(f"处理了 {len(vectors)} 个文本")
```

### 示例 5：与 RAGQA 集成

```python
from rag_qa import RAGQA
from vector_store import VectorStore

# 加载向量存储
vs = VectorStore()
vs.load_existing()

# 使用通义千问
qa = RAGQA(
    retriever=vs.get_retriever(),
    model_name="qwen-plus",
    llm_provider="dashscope"
)

# 提问
result = qa.ask("什么是深度学习？")
print(result["answer"])
```

---

## 七、参数说明

### 7.1 Temperature

| 值 | 特点 | 适用场景 |
|----|------|----------|
| 0.0-0.3 | 保守、一致 | 事实问答、代码生成 |
| 0.4-0.7 | 平衡 | 一般对话 |
| 0.8-1.0 | 创造、多样 | 创意写作 |

### 7.2 Top_p

核采样参数，控制输出多样性：

- 较小的值（0.1-0.5）：更确定的输出
- 较大的值（0.5-1.0）：更多样的输出

### 7.3 Max_tokens

最大输出 token 数，根据需求调整：

- 简单问答：512-1024
- 一般对话：1024-2048
- 长文本生成：2048-4096

---

## 八、错误处理

```python
from dashscope_llm import DashScopeLLM

try:
    llm = DashScopeLLM(model="qwen-plus")
    response = llm.invoke("问题")
except ValueError as e:
    print(f"配置错误: {e}")
except RuntimeError as e:
    print(f"API 调用失败: {e}")
```

---

## 九、最佳实践

### 9.1 模型选择

| 场景 | 推荐模型 |
|------|----------|
| 快速响应 | qwen-turbo |
| 日常使用 | qwen-plus |
| 复杂任务 | qwen-max |
| 长文档 | qwen-long |

### 9.2 成本优化

```python
# 简单任务用 turbo
llm = DashScopeLLM(model="qwen-turbo")

# 复杂任务用 max
llm = DashScopeLLM(model="qwen-max")

# 限制输出长度
llm = DashScopeLLM(model="qwen-plus", max_tokens=1024)
```

---

## 十、小结

`dashscope_llm.py` 让 RAG 系统能够使用阿里云通义千问：

- ✅ 多模型支持（turbo/plus/max/long）
- ✅ 中文优化
- ✅ 嵌入模型支持
- ✅ 流式输出
- ✅ 多轮对话

这是中文 RAG 系统的最佳 LLM 选择！
