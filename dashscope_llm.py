#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阿里云通义千问大模型适配器
支持 Qwen 系列模型的 API 调用
使 RAG 系统能够使用阿里云的大模型服务

阿里云 DashScope: https://dashscope.console.aliyun.com/
"""

import os
import time
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass

# 尝试导入 dashscope SDK
try:
    import dashscope
    from dashscope import Generation
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    print("警告: dashscope SDK 未安装。请运行: pip install dashscope")


@dataclass
class DashScopeConfig:
    """DashScope 配置信息"""
    api_key: str
    model: str = "qwen-plus"  # 默认使用 qwen-plus
    temperature: float = 0.7
    top_p: float = 0.8
    max_tokens: int = 2048


class DashScopeLLM:
    """
    阿里云通义千问大模型适配器

    支持的模型列表:
    - qwen-turbo: 快速响应，适合简单任务
    - qwen-plus: 通用模型，性价比高
    - qwen-max: 最强模型，适合复杂任务
    - qwen-long: 长上下文模型

    使用前需要:
    1. 注册阿里云账号: https://www.aliyun.com/
    2. 开通 DashScope 服务: https://dashscope.console.aliyun.com/
    3. 创建 API Key
    4. 设置环境变量或直接传入密钥
    """

    # 支持的模型列表
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
            "description": "最强模型，适合复杂推理任务",
            "context_length": 32768,
            "recommended_for": ["复杂问答", "推理任务", "内容创作"]
        },
        "qwen-long": {
            "name": "通义千问-Long",
            "description": "长上下文模型，支持超长文档",
            "context_length": 1000000,
            "recommended_for": ["长文档理解", "文档摘要"]
        }
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen-plus",
        temperature: float = 0.7,
        top_p: float = 0.8,
        max_tokens: int = 2048
    ):
        """
        初始化通义千问模型

        Args:
            api_key: 阿里云 DashScope API Key（可从环境变量获取）
            model: 模型名称
            temperature: 温度参数，控制输出随机性（0-1）
            top_p: 核采样参数（0-1）
            max_tokens: 最大输出token数
        """
        if not DASHSCOPE_AVAILABLE:
            raise ImportError(
                "dashscope SDK 未安装。请运行: pip install dashscope\n"
                "文档: https://help.aliyun.com/zh/dashscope/developer-reference/api-details"
            )

        # 获取密钥（优先使用传入参数，其次使用环境变量）
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "缺少阿里云 DashScope API Key。\n"
                "请设置环境变量:\n"
                "  export DASHSCOPE_API_KEY=your_api_key\n"
                "或在代码中传入参数:\n"
                "  llm = DashScopeLLM(api_key='your_api_key')"
            )

        # 设置模型参数
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

        # 配置 dashscope
        dashscope.api_key = self.api_key

        print(f"通义千问模型初始化成功: {self.model}")

    def invoke(self, prompt: str, **kwargs) -> Any:
        """
        调用模型生成回复（兼容 LangChain 接口）

        Args:
            prompt: 输入提示词
            **kwargs: 额外参数

        Returns:
            包含 content 属性的响应对象
        """
        return self._call(prompt, **kwargs)

    def _call(self, prompt: str, **kwargs) -> Any:
        """
        内部调用方法

        Args:
            prompt: 输入文本
            **kwargs: 额外参数

        Returns:
            响应对象
        """
        # 合并参数
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        try:
            # 调用 DashScope API
            response = Generation.call(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                result_format='message'
            )

            if response.status_code == 200:
                # 提取回复内容
                content = response.output.choices[0].message.content
                return DashScopeResponse(content)
            else:
                raise RuntimeError(
                    f"DashScope API 调用失败: {response.code} - {response.message}"
                )

        except Exception as e:
            raise RuntimeError(f"通义千问 API 调用失败: {e}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        多轮对话接口

        Args:
            messages: 对话消息列表，格式如:
                [{"role": "user", "content": "你好"}]
            **kwargs: 额外参数

        Returns:
            模型回复文本
        """
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        try:
            response = Generation.call(
                model=self.model,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                result_format='message'
            )

            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                raise RuntimeError(
                    f"DashScope API 调用失败: {response.code} - {response.message}"
                )

        except Exception as e:
            raise RuntimeError(f"通义千问对话调用失败: {e}")

    def stream_call(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        流式调用（逐字返回）

        Args:
            prompt: 输入文本
            **kwargs: 额外参数

        Yields:
            逐个字符
        """
        temperature = kwargs.get("temperature", self.temperature)
        top_p = kwargs.get("top_p", self.top_p)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        try:
            responses = Generation.call(
                model=self.model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                result_format='message',
                stream=True
            )

            for response in responses:
                if response.status_code == 200:
                    content = response.output.choices[0].message.content
                    if content:
                        yield content

        except Exception as e:
            raise RuntimeError(f"通义千问流式调用失败: {e}")

    def __call__(self, prompt: str, **kwargs) -> Any:
        """支持直接调用"""
        return self.invoke(prompt, **kwargs)

    @classmethod
    def list_models(cls) -> Dict[str, Dict]:
        """列出所有支持的模型"""
        return cls.SUPPORTED_MODELS

    @classmethod
    def print_models(cls):
        """打印模型列表"""
        print("阿里云通义千问支持的模型:")
        print("=" * 60)
        for model_id, info in cls.SUPPORTED_MODELS.items():
            print(f"\n模型: {model_id}")
            print(f"  名称: {info['name']}")
            print(f"  描述: {info['description']}")
            print(f"  上下文长度: {info['context_length']} tokens")
            print(f"  推荐场景: {', '.join(info['recommended_for'])}")


class DashScopeResponse:
    """DashScope 响应包装器，兼容 LangChain 接口"""

    def __init__(self, content: str):
        self.content = content

    def __str__(self):
        return self.content


class DashScopeEmbeddings:
    """
    阿里云通义千问嵌入模型适配器

    用于将文本转换为向量，支持语义搜索

    支持的嵌入模型:
    - text-embedding-v1: 通用文本嵌入
    - text-embedding-v2: 新版嵌入模型
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-v2"
    ):
        """
        初始化嵌入模型

        Args:
            api_key: 阿里云 DashScope API Key
            model: 嵌入模型名称
        """
        if not DASHSCOPE_AVAILABLE:
            raise ImportError("dashscope SDK 未安装。请运行: pip install dashscope")

        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.model = model

        if not self.api_key:
            raise ValueError("缺少阿里云 DashScope API Key")

        dashscope.api_key = self.api_key

    def embed_query(self, text: str) -> List[float]:
        """
        将文本转换为嵌入向量

        Args:
            text: 输入文本

        Returns:
            嵌入向量（浮点数列表）
        """
        try:
            from dashscope import TextEmbedding

            response = TextEmbedding.call(
                model=self.model,
                input=text
            )

            if response.status_code == 200:
                return response.output['embeddings'][0]['embedding']
            else:
                raise RuntimeError(
                    f"嵌入 API 调用失败: {response.code} - {response.message}"
                )

        except Exception as e:
            raise RuntimeError(f"通义千问嵌入调用失败: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档

        Args:
            texts: 文本列表

        Returns:
            嵌入向量列表
        """
        try:
            from dashscope import TextEmbedding

            # 分批处理（每批最多25个）
            all_embeddings = []
            batch_size = 25

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                response = TextEmbedding.call(
                    model=self.model,
                    input=batch
                )

                if response.status_code == 200:
                    for item in response.output['embeddings']:
                        all_embeddings.append(item['embedding'])
                else:
                    raise RuntimeError(
                        f"批量嵌入失败: {response.code} - {response.message}"
                    )

            return all_embeddings

        except Exception as e:
            raise RuntimeError(f"通义千问批量嵌入失败: {e}")


# ============================================
# 辅助函数
# ============================================

def create_dashscope_llm(
    model: str = "qwen-plus",
    temperature: float = 0.7
) -> Optional[DashScopeLLM]:
    """
    创建通义千问 LLM 实例的便捷函数

    Args:
        model: 模型名称
        temperature: 温度参数

    Returns:
        DashScopeLLM 实例或 None
    """
    try:
        return DashScopeLLM(model=model, temperature=temperature)
    except Exception as e:
        print(f"创建通义千问 LLM 失败: {e}")
        return None


def check_dashscope_credentials() -> bool:
    """
    检查 DashScope 凭证是否配置

    Returns:
        True 如果凭证已配置
    """
    return bool(os.getenv("DASHSCOPE_API_KEY"))


def get_dashscope_config_help() -> str:
    """获取 DashScope 配置帮助信息"""
    return """
========================================
阿里云通义千问配置指南
========================================

1. 注册阿里云账号
   访问: https://www.aliyun.com/

2. 开通 DashScope 服务
   访问: https://dashscope.console.aliyun.com/
   点击"开通服务"

3. 创建 API Key
   - 进入"API-KEY管理"页面
   - 创建新的 API Key

4. 配置环境变量

   Windows CMD:
   set DASHSCOPE_API_KEY=your_api_key

   Windows PowerShell:
   $env:DASHSCOPE_API_KEY="your_api_key"

   Linux/Mac:
   export DASHSCOPE_API_KEY=your_api_key

   或在 .env 文件中添加:
   DASHSCOPE_API_KEY=your_api_key

5. 安装 SDK
   pip install dashscope

========================================
"""


# ============================================
# 示例用法
# ============================================

if __name__ == "__main__":
    print("阿里云通义千问大模型适配器")
    print("=" * 60)

    # 打印支持的模型
    DashScopeLLM.print_models()

    # 检查凭证
    print("\n" + "=" * 60)
    if check_dashscope_credentials():
        print("✓ DashScope 凭证已配置")

        # 测试调用
        try:
            print("\n测试调用通义千问模型...")
            llm = create_dashscope_llm(model="qwen-plus", temperature=0.7)

            response = llm.invoke("你好，请用一句话介绍什么是人工智能。")
            print(f"回复: {response.content}")

        except Exception as e:
            print(f"调用失败: {e}")
    else:
        print("✗ DashScope 凭证未配置")
        print(get_dashscope_config_help())
