# 文档加载模块 (document_loader.py)

## 一、模块概述

`document_loader.py` 是 RAG 系统的**入口模块**，负责加载各种格式的文档并将其分割成适合检索的文本块。可以把它理解为系统的"食材处理区"。

> **核心价值**：将各种格式的原始文档转换为结构化的文本块，为后续的向量化做准备。

---

## 二、核心概念

### 2.1 Document 对象

LangChain 中的标准文档结构：

```python
from langchain_core.documents import Document

doc = Document(
    page_content="这是文档的文本内容...",  # 文本内容
    metadata={                          # 元数据
        'source': '/path/to/file.txt',   # 来源文件
        'file_type': '.txt',             # 文件类型
        'file_name': 'file.txt'          # 文件名
    }
)
```

### 2.2 文本分块（Chunking）

**为什么需要分块？**

```
问题: 
- LLM 有上下文长度限制
- 太长的文档检索效率低
- 需要精确定位相关内容

解决: 将长文档分割成小块

原始文档 (50000字)
    ↓ 分割
[块1, 块2, 块3, ..., 块N] (每块1000字)
```

### 2.3 分块参数

```python
chunk_size = 1000    # 每块最大字符数
chunk_overlap = 200  # 相邻块重叠字符数
```

**重叠的作用：** 确保上下文连续性，避免重要信息被截断。

```
文档: "ABCDEFGHIJ"

chunk_size=5, overlap=2:
  块1: "ABCDE"
  块2: "DEFGH"  # DE 是重叠部分
  块3: "HIJ"
```

### 2.4 分隔符优先级

系统按优先级尝试分隔符，确保语义完整性：

```python
separators = [
    "\n\n",    # 1. 优先按段落分割
    "\n",      # 2. 其次按行分割
    "。",      # 3. 中文句号
    "！",      # 4. 中文感叹号
    "？",      # 5. 中文问号
    ". ",      # 6. 英文句号
    "! ",      # 7. 英文感叹号
    "? ",      # 8. 英文问号
    " ",       # 9. 空格
    ""         # 10. 最后按字符分割
]
```

---

## 三、核心类：DocumentLoader

### 3.1 类定义

```python
class DocumentLoader:
    """Load and split documents for RAG system."""

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None):
        """
        Initialize document loader with text splitter.

        Args:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻块之间的重叠字符数
            separators: 自定义分隔符列表
        """
```

### 3.2 核心属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `chunk_size` | int | 文本块大小 |
| `chunk_overlap` | int | 重叠大小 |
| `text_splitter` | RecursiveCharacterTextSplitter | 文本分割器 |

---

## 四、核心方法详解

### 4.1 load_pdf() - 加载 PDF

```python
def load_pdf(self, file_path: str) -> List[Document]:
    """
    Load PDF document and split into chunks.

    Args:
        file_path: PDF 文件路径

    Returns:
        Document 对象列表
    """
```

**元数据示例：**

```python
{
    'source': '/path/to/document.pdf',
    'file_type': 'pdf',
    'page': 0  # 页码（从0开始）
}
```

### 4.2 load_text() - 加载文本

```python
def load_text(self, file_path: str) -> List[Document]:
    """
    Load text document and split into chunks.

    Args:
        file_path: 文本文件路径

    Returns:
        Document 对象列表
    """
```

**支持的编码：** UTF-8, UTF-8-BOM, GBK, GB2312, GB18030, Latin-1

### 4.3 load_directory() - 加载目录

```python
def load_directory(self,
                   directory_path: str,
                   recursive: bool = False,
                   extensions: Optional[List[str]] = None) -> List[Document]:
    """
    Load all supported documents from a directory.

    Args:
        directory_path: 目录路径
        recursive: 是否递归搜索子目录
        extensions: 要加载的文件扩展名列表

    Returns:
        所有文档块的列表
    """
```

### 4.4 load_file() - 自动识别类型

```python
def load_file(self, file_path: str) -> List[Document]:
    """
    Load a single file based on its extension.

    Args:
        file_path: 文件路径

    Returns:
        Document 对象列表
    """
```

### 4.5 load_string() - 加载字符串

```python
def load_string(self,
                text: str,
                metadata: Optional[dict] = None) -> List[Document]:
    """
    Load text from a string and split into chunks.

    Args:
        text: 文本内容
        metadata: 可选的元数据

    Returns:
        Document 对象列表
    """
```

---

## 五、文本分割器原理

### 5.1 RecursiveCharacterTextSplitter

使用递归方式分割文本，优先尝试大分隔符：

```
文本
  ↓ 尝试 "\n\n" 分割
段落列表
  ↓ 某段落太长？尝试 "\n" 分割
句子列表
  ↓ 某句子太长？尝试 "。" 分割
...
  ↓ 最终按字符分割
文本块列表
```

### 5.2 分割流程图

```
┌─────────────────────────────────────────────────────────────┐
│                      原始文本                                │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  1. 尝试按 "\n\n" 分割（段落）                               │
│     - 如果段落 <= chunk_size: 保持                           │
│     - 如果段落 > chunk_size: 继续分割                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  2. 尝试按 "\n" 分割（行）                                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  3. 尝试按句号、问号等分割（句子）                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  4. 最终按字符分割                                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      文本块列表                              │
│  [块1, 块2, 块3, ...]                                       │
│  每块包含: page_content + metadata                           │
└─────────────────────────────────────────────────────────────┘
```

---

## 六、使用示例

### 示例 1：加载单个文件

```python
from document_loader import DocumentLoader

loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)

# 加载 PDF
pdf_chunks = loader.load_pdf("./document.pdf")

# 加载文本
txt_chunks = loader.load_text("./article.txt")

# 自动识别类型
chunks = loader.load_file("./notes.md")

print(f"加载了 {len(chunks)} 个文本块")
```

### 示例 2：批量加载目录

```python
loader = DocumentLoader(chunk_size=800, chunk_overlap=150)

# 加载整个目录
chunks = loader.load_directory(
    "./knowledge_base/",
    recursive=True,  # 递归加载子目录
    extensions=['.pdf', '.txt', '.md']
)

print(f"共加载 {len(chunks)} 个文本块")

# 统计来源
from collections import Counter
sources = Counter(chunk.metadata.get('source') for chunk in chunks)
for source, count in sources.most_common():
    print(f"  {source}: {count} 块")
```

### 示例 3：处理用户输入

```python
loader = DocumentLoader(chunk_size=500, chunk_overlap=100)

user_text = """
用户输入的长文本内容...
可以是问题、笔记、或任何文本。
"""

chunks = loader.load_string(
    user_text,
    metadata={'source': 'user_input', 'timestamp': '2024-01-01'}
)

print(f"分割成 {len(chunks)} 块")
```

### 示例 4：自定义分隔符

```python
# 针对代码文档的分隔符
code_separators = [
    "\nclass ",    # 类定义
    "\ndef ",      # 函数定义
    "\n\n",        # 空行
    "\n",          # 换行
    " ",           # 空格
    ""             # 字符
]

loader = DocumentLoader(
    chunk_size=1500,
    chunk_overlap=200,
    separators=code_separators
)

# 针对中文文档的分隔符
chinese_separators = [
    "\n\n", "\n", "。", "；", "，", " ", ""
]

loader = DocumentLoader(separators=chinese_separators)
```

---

## 七、编码检测机制

### 7.1 自动检测流程

```python
def _detect_encoding(self, file_path: str) -> str:
    """Try to detect file encoding."""
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'latin-1']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1024)  # 尝试读取前 1KB
            return encoding
        except (UnicodeDecodeError, UnicodeError):
            continue

    return 'utf-8'  # 默认回退
```

### 7.2 支持的编码

| 编码 | 说明 |
|------|------|
| UTF-8 | 标准编码 |
| UTF-8-BOM | 带 BOM 的 UTF-8 |
| GBK | 简体中文 |
| GB2312 | 简体中文（旧） |
| GB18030 | 中文超集 |
| Latin-1 | 西欧语言 |

---

## 八、最佳实践

### 8.1 分块参数选择

| 文档类型 | 推荐 chunk_size | 推荐 overlap |
|----------|-----------------|--------------|
| 问答系统 | 500-1000 | 100-200 |
| 文档摘要 | 1500-3000 | 300-500 |
| 代码文件 | 800-1500 | 200-300 |
| 新闻文章 | 500-800 | 100-150 |
| 技术文档 | 1000-1500 | 200-300 |
| FAQ/问答对 | 200-400 | 50-100 |

### 8.2 错误处理

```python
from document_loader import DocumentLoader

loader = DocumentLoader()

try:
    chunks = loader.load_file("./document.pdf")
except FileNotFoundError as e:
    print(f"文件不存在: {e}")
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("PDF 支持需要: pip install pypdf")
except ValueError as e:
    print(f"不支持的文件类型: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

### 8.3 质量检查

```python
# 检查分割质量
chunks = loader.load_text("./document.txt")

for i, chunk in enumerate(chunks):
    content = chunk.page_content

    # 检查是否截断句子
    if not content.endswith(('.', '。', '!', '！', '?', '？')):
        print(f"块 {i} 可能截断了句子")

    # 检查是否太短
    if len(content) < 100:
        print(f"块 {i} 内容过短: {len(content)} 字符")
```

---

## 九、常见问题

### Q1: PDF 加载失败？

```python
# 确保安装了 pypdf
pip install pypdf

# 或者使用 pypdf2
pip install pypdf2
```

### Q2: 中文乱码？

系统会自动检测编码。如果仍有问题：

```python
# 方法1: 确保文件是 UTF-8 编码
# 方法2: 手动读取后用 load_string
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

chunks = loader.load_string(text, metadata={'source': file_path})
```

### Q3: 超大文件处理？

```python
# 使用更小的 chunk_size 和更少的 overlap
loader = DocumentLoader(
    chunk_size=500,   # 较小的块
    chunk_overlap=50  # 较小的重叠
)

# 或分批处理
import os
for filename in os.listdir("./large_docs/"):
    chunks = loader.load_file(f"./large_docs/{filename}")
    # 立即处理每个文件
    process_chunks(chunks)
```

---

## 十、与其他模块的关系

```
┌─────────────────┐
│ document_loader │
└────────┬────────┘
         │ 输出 List[Document]
         ↓
┌─────────────────┐
│  vector_store   │ ← 创建向量索引
└────────┬────────┘
         │ 输出 Retriever
         ↓
┌─────────────────┐
│     rag_qa      │ ← 生成回答
└─────────────────┘
```

### 完整流程示例

```python
from document_loader import DocumentLoader
from vector_store import VectorStore
from rag_qa import RAGQA

# 1. 加载文档
loader = DocumentLoader(chunk_size=1000, chunk_overlap=200)
documents = loader.load_directory("./documents/")

# 2. 创建向量存储
vector_store = VectorStore(persist_directory="./db")
vector_store.create_from_documents(documents)

# 3. 创建问答系统
retriever = vector_store.get_retriever(search_kwargs={"k": 4})
qa = RAGQA(retriever=retriever)

# 4. 提问
result = qa.ask("文档的主要内容是什么？")
print(result["answer"])
```

---

## 十一、小结

`document_loader.py` 是 RAG 系统的第一步：

- ✅ 支持 PDF、TXT、Markdown 格式
- ✅ 自动检测文件编码
- ✅ 智能文本分割
- ✅ 保留文档元数据
- ✅ 支持批量处理

掌握了文档加载，就掌握了 RAG 系统的数据输入！
