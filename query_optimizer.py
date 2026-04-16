#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Optimizer Module for RAG System.
Uses LLM to optimize and clarify user questions before retrieval.
"""

import os
import warnings
from typing import Dict, Any, Optional

from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()


class QueryOptimizer:
    """
    Optimizes user questions using LLM before retrieval.
    Makes questions clearer, more specific, and easier to search.
    """

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.3,
                 enabled: bool = True):
        """
        Initialize query optimizer.

        Args:
            model_name: LLM model name for optimization
            temperature: Model temperature (lower = more consistent)
            enabled: Whether to enable query optimization
        """
        self.model_name = model_name
        self.temperature = temperature
        self.enabled = enabled
        self.llm = None
        self._llm_type = "none"

        if self.enabled:
            self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the language model for query optimization."""
        openai_api_key = os.getenv("OPENAI_API_KEY")

        # Try OpenAI first
        if openai_api_key:
            llm = self._try_openai(openai_api_key)
            if llm:
                self.llm = llm
                return

        # If no LLM available, disable optimizer
        self.enabled = False
        self._llm_type = "none"

    def _try_openai(self, api_key: str):
        """Try to initialize OpenAI model."""
        try:
            from langchain_openai import ChatOpenAI
            self._llm_type = "openai"
            print(f"Query Optimizer: Using OpenAI model {self.model_name}")
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=api_key
            )
        except ImportError:
            pass
        except Exception as e:
            print(f"Query Optimizer: OpenAI initialization error: {e}")

        return None

    def optimize(self, question: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize a user question for better retrieval.

        Args:
            question: Original user question
            context: Optional conversation context

        Returns:
            Dictionary with optimized question and metadata
        """
        result = {
            "original_question": question,
            "optimized_question": question,  # Default to original
            "was_optimized": False,
            "optimizer_status": self._llm_type if self.enabled else "disabled"
        }

        # If optimizer is disabled or no LLM, return original
        if not self.enabled or not self.llm:
            return result

        # Don't optimize very short or already clear questions
        if len(question.strip()) < 5:
            return result

        try:
            # Build optimization prompt
            prompt = self._build_optimization_prompt(question, context)

            # Get optimized question from LLM
            response = self._get_llm_response(prompt)

            if response and response.strip():
                optimized = response.strip()

                # Remove quotes if the LLM added them
                if optimized.startswith('"') and optimized.endswith('"'):
                    optimized = optimized[1:-1]
                if optimized.startswith("'") and optimized.endswith("'"):
                    optimized = optimized[1:-1]

                # Only use if it's different from original
                if optimized.lower() != question.lower():
                    result["optimized_question"] = optimized
                    result["was_optimized"] = True

                    print(f"\n[Query Optimizer]")
                    print(f"  原始问题: {question}")
                    print(f"  优化后: {optimized}")

        except Exception as e:
            print(f"Query optimization error: {e}")

        return result

    def _build_optimization_prompt(self, question: str, context: Optional[str] = None) -> str:
        """Build the prompt for query optimization."""
        base_prompt = """你是一个问题优化专家。你的任务是将用户的问题优化得更加清晰、具体和易于检索。

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

用户输入: "有什么好处"
优化后: "RAG（检索增强生成）系统有哪些优点和好处？请列举主要优势"
"""

        if context:
            base_prompt += f"\n\n对话上下文:\n{context}"

        base_prompt += f"\n\n用户输入: {question}\n优化后:"

        return base_prompt

    def _get_llm_response(self, prompt: str) -> str:
        """Get response from the LLM."""
        try:
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    return response.content
                return str(response)
            elif callable(self.llm):
                result = self.llm(prompt)
                if hasattr(result, 'content'):
                    return result.content
                return str(result)
            else:
                return str(self.llm)
        except Exception as e:
            print(f"LLM response error: {e}")
            return ""

    def is_enabled(self) -> bool:
        """Check if optimizer is enabled and working."""
        return self.enabled and self.llm is not None

    def get_status(self) -> Dict[str, Any]:
        """Get optimizer status information."""
        return {
            "enabled": self.enabled,
            "llm_type": self._llm_type,
            "model_name": self.model_name,
            "temperature": self.temperature
        }


class QueryExpander:
    """
    Alternative approach: Expands query with related terms and variations.
    Useful when the original query might miss relevant documents.
    """

    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.5,
                 num_variations: int = 3):
        """
        Initialize query expander.

        Args:
            model_name: LLM model name
            temperature: Model temperature
            num_variations: Number of query variations to generate
        """
        self.model_name = model_name
        self.temperature = temperature
        self.num_variations = num_variations
        self.llm = None

        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the language model."""
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if openai_api_key:
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=self.model_name,
                    temperature=self.temperature,
                    openai_api_key=openai_api_key
                )
            except Exception as e:
                print(f"Query Expander initialization error: {e}")

    def expand(self, question: str) -> Dict[str, Any]:
        """
        Generate multiple variations of the query.

        Args:
            question: Original question

        Returns:
            Dictionary with original and expanded queries
        """
        result = {
            "original_question": question,
            "expanded_queries": [question],
            "was_expanded": False
        }

        if not self.llm:
            return result

        try:
            prompt = f"""请为以下问题生成{self.num_variations}个不同的表达方式。
这些表达应该:
1. 保持相同的核心含义
2. 使用不同的词汇和句式
3. 有助于检索到更多相关文档

问题: {question}

请只输出{self.num_variations}个问题，每行一个，不要编号或其他内容:"""

            response = self._get_llm_response(prompt)

            if response:
                variations = [
                    line.strip()
                    for line in response.strip().split('\n')
                    if line.strip() and line.strip() != question
                ]

                if variations:
                    result["expanded_queries"] = [question] + variations[:self.num_variations]
                    result["was_expanded"] = True

                    print(f"\n[Query Expander]")
                    print(f"  原始: {question}")
                    for i, q in enumerate(result["expanded_queries"][1:], 1):
                        print(f"  变体{i}: {q}")

        except Exception as e:
            print(f"Query expansion error: {e}")

        return result

    def _get_llm_response(self, prompt: str) -> str:
        """Get response from the LLM."""
        try:
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke(prompt)
                if hasattr(response, 'content'):
                    return response.content
                return str(response)
            return ""
        except Exception as e:
            print(f"LLM response error: {e}")
            return ""


# Example usage
if __name__ == "__main__":
    import tempfile
    from document_loader import DocumentLoader
    from vector_store import VectorStore

    print("=" * 60)
    print("  Query Optimizer Demo")
    print("=" * 60)

    # Create optimizer
    optimizer = QueryOptimizer(enabled=True)
    print(f"\nOptimizer Status: {optimizer.get_status()}")

    # Test questions
    test_questions = [
        "它是什么？",
        "怎么用",
        "有什么好处",
        "人工智能的发展历史是怎样的",
        "如何提高效率"
    ]

    print("\n" + "=" * 60)
    print("  Testing Query Optimization")
    print("=" * 60)

    for question in test_questions:
        print(f"\n原始问题: {question}")
        result = optimizer.optimize(question)
        if result["was_optimized"]:
            print(f"优化后: {result['optimized_question']}")
        else:
            print("(问题未优化或优化器未启用)")

    print("\n" + "=" * 60)
    print("  Demo completed!")
    print("=" * 60)
