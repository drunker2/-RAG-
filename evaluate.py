#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 召回率评估工具

用法:
    python evaluate.py                          # 交互模式
    python evaluate.py --batch                  # 批量测试（使用默认测试集）
    python evaluate.py --test-file tests.json   # 使用自定义测试文件
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any

# Windows 编码修复
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stdin.reconfigure(encoding='utf-8')
    except:
        pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from vector_store import VectorStore
from hybrid_retriever import HybridRetriever


class Evaluator:
    """召回率评估器"""

    def __init__(
        self,
        vector_store: VectorStore,
        k: int = 10,
        candidate_k: int = 20,
        use_rerank: bool = True
    ):
        self.k = k
        self.vector_store = vector_store

        # 初始化混合检索器
        self.retriever = vector_store.get_hybrid_retriever(
            k=k,
            candidate_k=candidate_k,
            use_rerank=use_rerank
        )
        print(f"检索配置: 候选={candidate_k}, Rerank={use_rerank}, 最终={k}")

    def search(self, query: str) -> List[Any]:
        """执行检索"""
        return self.retriever.invoke(query)

    def evaluate(self, query: str, keywords: List[str]) -> Dict[str, Any]:
        """
        评估单个查询

        Args:
            query: 查询问题
            keywords: 关键词列表

        Returns:
            评估结果
        """
        start = time.time()
        results = self.search(query)
        latency = (time.time() - start) * 1000

        # 检查关键词命中
        contents = [doc.page_content for doc in results]
        hits = sum(1 for kw in keywords if any(kw in c for c in contents))

        recall = hits / len(keywords) if keywords else 0
        precision = hits / len(results) if results else 0

        return {
            "query": query,
            "keywords": keywords,
            "recall": recall,
            "precision": precision,
            "hits": hits,
            "total_keywords": len(keywords),
            "latency_ms": latency,
            "result_count": len(results)
        }

    def batch_evaluate(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """批量评估"""
        results = []
        total_recall = 0
        total_latency = 0

        for i, case in enumerate(test_cases, 1):
            query = case["query"]
            keywords = case.get("keywords", [])

            print(f"\r进度: {i}/{len(test_cases)} - {query[:30]}...", end="", flush=True)

            r = self.evaluate(query, keywords)
            results.append(r)
            total_recall += r["recall"]
            total_latency += r["latency_ms"]

        print()

        n = len(test_cases)
        return {
            "avg_recall": total_recall / n if n else 0,
            "avg_latency_ms": total_latency / n if n else 0,
            "total": n,
            "details": results
        }


def interactive_mode(evaluator: Evaluator):
    """交互模式"""
    print("\n" + "=" * 60)
    print("交互式评估")
    print("=" * 60)
    print("\n格式: 问题 | 关键词1,关键词2,关键词3")
    print("示例: Docker是什么 | Docker,容器,镜像")
    print("输入 quit 退出\n")

    while True:
        try:
            raw = input("> ").strip()
            if raw.lower() in ["quit", "exit", "q"]:
                break
            if not raw:
                continue

            # 解析输入
            if "|" in raw:
                parts = raw.split("|", 1)
                query = parts[0].strip()
                keywords = [k.strip() for k in parts[1].split(",") if k.strip()]
            else:
                query = raw
                keywords = []

            if not keywords:
                # 无关键词，只展示检索结果
                results = evaluator.search(query)
                print(f"\n检索到 {len(results)} 条结果:")
                for i, doc in enumerate(results[:3], 1):
                    print(f"  [{i}] {doc.page_content[:100]}...\n")
                continue

            # 评估
            r = evaluator.evaluate(query, keywords)

            print(f"\n召回率: {r['recall']:.0%}")
            print(f"精确率: {r['precision']:.0%}")
            print(f"命中: {r['hits']}/{r['total_keywords']}")
            print(f"延迟: {r['latency_ms']:.0f}ms\n")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"错误: {e}\n")

    print("\n已退出")


def default_test_cases() -> List[Dict]:
    """默认测试用例"""
    return [
        {"query": "Docker 容器和虚拟机的区别", "keywords": ["容器", "虚拟机"]},
        {"query": "如何映射 Docker 端口", "keywords": ["端口", "映射"]},
        {"query": "Dockerfile COPY 和 ADD 的区别", "keywords": ["COPY", "ADD"]},
        {"query": "阿里巴巴开发规范有哪些", "keywords": ["阿里巴巴", "规范"]},
        {"query": "Java ArrayList 和 LinkedList 区别", "keywords": ["ArrayList", "LinkedList"]},
        {"query": "什么是 JVM 内存结构", "keywords": ["JVM", "内存"]},
        {"query": "volatile 关键字的作用", "keywords": ["volatile"]},
        {"query": "HashMap 在 JDK 1.7 和 1.8 的变化", "keywords": ["HashMap", "1.7", "1.8"]},
    ]


def print_report(result: Dict):
    """打印评估报告"""
    print("\n" + "=" * 60)
    print("评估报告")
    print("=" * 60)
    print(f"\n测试数量: {result['total']}")
    print(f"平均召回率: {result['avg_recall']:.1%}")
    print(f"平均延迟: {result['avg_latency_ms']:.0f}ms")

    print("\n详细结果:")
    print("-" * 60)
    for i, r in enumerate(result["details"], 1):
        print(f"{i}. {r['query'][:40]}")
        print(f"   召回率: {r['recall']:.0%} | 命中: {r['hits']}/{r['total_keywords']} | {r['latency_ms']:.0f}ms")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="RAG 召回率评估")
    parser.add_argument("--batch", action="store_true", help="批量测试模式")
    parser.add_argument("--test-file", type=str, help="测试文件路径 (JSON)")
    parser.add_argument("--k", type=int, default=10, help="最终返回数量 (默认10)")
    parser.add_argument("--candidate-k", type=int, default=20, help="各路召回候选数 (默认20)")
    parser.add_argument("--no-rerank", action="store_true", help="禁用 Rerank")

    args = parser.parse_args()

    print("=" * 60)
    print("RAG 召回率评估工具")
    print("=" * 60)

    # 加载向量存储
    print("\n加载向量存储...")
    vector_store = VectorStore(persist_directory="./chroma_db")
    if not vector_store.load_existing():
        print("错误: 向量存储不存在，请先索引文档")
        return

    # 创建评估器
    evaluator = Evaluator(
        vector_store=vector_store,
        k=args.k,
        candidate_k=args.candidate_k,
        use_rerank=not args.no_rerank
    )

    # 运行模式
    if args.batch or args.test_file:
        # 批量测试
        if args.test_file:
            with open(args.test_file, "r", encoding="utf-8") as f:
                test_cases = json.load(f)
        else:
            test_cases = default_test_cases()

        print(f"\n开始批量测试 ({len(test_cases)} 个问题)...")
        result = evaluator.batch_evaluate(test_cases)
        print_report(result)
    else:
        # 交互模式
        interactive_mode(evaluator)


if __name__ == "__main__":
    main()
