#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Retriever Module for RAG System.
Combines BM25 (keyword-based) and vector (semantic) search for better retrieval.
Supports Rerank for final result ordering.
"""

import os
import pickle
import math
import re
import warnings
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Suppress warnings
warnings.filterwarnings("ignore")


# ============================================
# Rerank 支持
# ============================================

class DashScopeReranker:
    """
    阿里云 DashScope Rerank 服务

    使用 gte-rerank 模型对文档进行重排序
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化 Reranker

        Args:
            api_key: DashScope API Key (可从环境变量获取)
        """
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")

        if not self.api_key:
            raise ValueError("缺少 DASHSCOPE_API_KEY，请设置环境变量")

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: int = 10
    ) -> List[Tuple[int, float]]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 文档列表
            top_n: 返回前 N 个结果

        Returns:
            List of (original_index, rerank_score) tuples
        """
        if not documents:
            return []

        try:
            import dashscope
            from dashscope import TextReRank

            dashscope.api_key = self.api_key

            response = TextReRank.call(
                model="gte-rerank",
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
                return_documents=False
            )

            if response.status_code == 200:
                results = []
                for item in response.output.results:
                    # item 包含 index 和 relevance_score
                    results.append((item['index'], item['relevance_score']))
                return results
            else:
                print(f"Rerank API 错误: {response.code} - {response.message}")
                return [(i, 1.0) for i in range(len(documents))]

        except ImportError:
            print("警告: dashscope 未安装或不支持 Rerank，使用原始排序")
            return [(i, 1.0) for i in range(len(documents))]
        except Exception as e:
            print(f"Rerank 错误: {e}")
            return [(i, 1.0) for i in range(len(documents))]


class BM25:
    """
    BM25 (Best Matching 25) algorithm implementation.
    A probabilistic ranking function for text retrieval.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25.

        Args:
            k1: Term frequency saturation parameter (default: 1.5)
            b: Document length normalization parameter (default: 0.75)
        """
        self.k1 = k1
        self.b = b
        self.documents: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.doc_count: int = 0
        self.avg_doc_length: float = 0.0
        self.doc_freqs: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.doc_term_freqs: List[Dict[str, int]] = []

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        Supports both English and Chinese text.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Convert to lowercase for English
        text = text.lower()

        tokens = []

        # Extract English words (continuous letters)
        english_words = re.findall(r'[a-z]+', text)
        tokens.extend(english_words)

        # Extract numbers
        numbers = re.findall(r'\d+', text)
        tokens.extend(numbers)

        # For Chinese: use character n-grams (bigrams) as fallback
        # This is a simple approach that works without jieba
        chinese_chars = re.findall(r'[\u4e00-\u9fff]+', text)
        for phrase in chinese_chars:
            # Add the whole phrase
            if len(phrase) >= 2:
                tokens.append(phrase)
            # Add bigrams for longer phrases
            if len(phrase) > 2:
                for i in range(len(phrase) - 1):
                    tokens.append(phrase[i:i+2])

        # Filter very short tokens (but keep Chinese bigrams)
        tokens = [t for t in tokens if len(t) >= 1]

        return tokens

    def fit(self, documents: List[str]) -> None:
        """
        Fit BM25 on a corpus of documents.

        Args:
            documents: List of document strings
        """
        self.documents = []
        self.doc_lengths = []
        self.doc_term_freqs = []
        self.doc_freqs = {}

        for doc in documents:
            tokens = self.tokenize(doc)
            self.documents.append(tokens)
            self.doc_lengths.append(len(tokens))

            # Calculate term frequencies for this document
            term_freqs = Counter(tokens)
            self.doc_term_freqs.append(dict(term_freqs))

            # Update document frequencies
            for term in set(tokens):
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        self.doc_count = len(documents)
        self.avg_doc_length = sum(self.doc_lengths) / self.doc_count if self.doc_count > 0 else 0

        # Calculate IDF for all terms
        self._calculate_idf()

    def _calculate_idf(self) -> None:
        """Calculate IDF (Inverse Document Frequency) for all terms."""
        self.idf = {}

        for term, df in self.doc_freqs.items():
            # Standard BM25 IDF formula
            idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = idf

    def get_scores(self, query: str) -> List[float]:
        """
        Calculate BM25 scores for all documents given a query.

        Args:
            query: Query string

        Returns:
            List of BM25 scores for each document
        """
        query_tokens = self.tokenize(query)
        scores = []

        for doc_idx, doc_tokens in enumerate(self.documents):
            score = 0.0
            doc_length = self.doc_lengths[doc_idx]
            term_freqs = self.doc_term_freqs[doc_idx]

            for term in query_tokens:
                if term not in self.idf:
                    continue

                # Get term frequency in this document
                tf = term_freqs.get(term, 0)
                if tf == 0:
                    continue

                # BM25 scoring formula
                idf = self.idf[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                score += idf * numerator / denominator

            scores.append(score)

        return scores

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search for documents matching the query.

        Args:
            query: Query string
            top_k: Number of top results to return

        Returns:
            List of (document_index, score) tuples
        """
        scores = self.get_scores(query)

        # Get top-k results
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]

    def save(self, filepath: str) -> None:
        """Save BM25 index to file."""
        data = {
            'k1': self.k1,
            'b': self.b,
            'documents': self.documents,
            'doc_lengths': self.doc_lengths,
            'doc_count': self.doc_count,
            'avg_doc_length': self.avg_doc_length,
            'doc_freqs': self.doc_freqs,
            'idf': self.idf,
            'doc_term_freqs': self.doc_term_freqs
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str) -> None:
        """Load BM25 index from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.k1 = data['k1']
        self.b = data['b']
        self.documents = data['documents']
        self.doc_lengths = data['doc_lengths']
        self.doc_count = data['doc_count']
        self.avg_doc_length = data['avg_doc_length']
        self.doc_freqs = data['doc_freqs']
        self.idf = data['idf']
        self.doc_term_freqs = data['doc_term_freqs']


class BM25Retriever(BaseRetriever):
    """
    LangChain-compatible BM25 Retriever.
    """

    documents: List[Document] = []
    bm25: BM25 = None
    k: int = 4

    def __init__(self, documents: List[Document], k: int = 4):
        """
        Initialize BM25 Retriever.

        Args:
            documents: List of Document objects
            k: Number of documents to retrieve
        """
        super().__init__()
        self.documents = documents
        self.k = k
        self.bm25 = BM25()

        # Fit BM25 on document contents
        texts = [doc.page_content for doc in documents]
        self.bm25.fit(texts)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve relevant documents using BM25."""
        results = self.bm25.search(query, top_k=self.k)
        return [self.documents[idx] for idx, score in results if score > 0]

    def save(self, filepath: str) -> None:
        """Save retriever to file."""
        data = {
            'documents': [(doc.page_content, doc.metadata) for doc in self.documents],
            'k': self.k
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        self.bm25.save(filepath + '.bm25')

    @classmethod
    def load(cls, filepath: str) -> 'BM25Retriever':
        """Load retriever from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        documents = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in data['documents']
        ]

        retriever = cls(documents=documents, k=data['k'])
        retriever.bm25.load(filepath + '.bm25')
        return retriever


class HybridRetriever(BaseRetriever):
    """
    Hybrid Retriever combining BM25 and Vector search.
    Uses Rerank for final result ordering (or RRF as fallback).
    """

    # Declare fields for Pydantic
    documents: List[Document] = []
    vector_store: Any = None
    bm25: BM25 = None
    reranker: Any = None
    k: int = 4
    candidate_k: int = 20  # 每路召回的候选数
    use_rerank: bool = True
    alpha: float = 0.5  # 仅用于 RRF fallback

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
        Initialize Hybrid Retriever.

        Args:
            documents: List of Document objects
            vector_store: Vector store instance (Chroma)
            k: Number of documents to retrieve
            candidate_k: Number of candidates from each retriever (default 20)
            use_rerank: Whether to use Rerank (default True)
            alpha: Weight for vector search (only used when Rerank disabled)
            bm25_k1: BM25 k1 parameter
            bm25_b: BM25 b parameter
        """
        super().__init__()
        self.documents = documents
        self.vector_store = vector_store
        self.k = k
        self.candidate_k = min(candidate_k, len(documents))
        self.use_rerank = use_rerank
        self.alpha = alpha

        # Initialize BM25
        self.bm25 = BM25(k1=bm25_k1, b=bm25_b)
        texts = [doc.page_content for doc in documents]
        self.bm25.fit(texts)

        # Initialize Reranker
        if self.use_rerank:
            try:
                self.reranker = DashScopeReranker()
                print("Hybrid Retriever initialized with Rerank:")
            except Exception as e:
                print(f"Reranker 初始化失败 ({e})，使用 RRF fallback")
                self.use_rerank = False
                self.reranker = None

        if not self.use_rerank:
            print("Hybrid Retriever initialized with RRF:")

        print(f"  Documents: {len(documents)}")
        print(f"  k (final): {k}")
        print(f"  candidate_k (each retriever): {self.candidate_k}")
        print(f"  use_rerank: {self.use_rerank}")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Retrieve documents using hybrid search."""

        # 1. BM25 召回
        bm25_results = self.bm25.search(query, top_k=self.candidate_k)
        bm25_indices = set(idx for idx, score in bm25_results if score > 0)

        # 2. 向量召回
        vector_indices = set()
        try:
            vector_docs = self.vector_store.similarity_search(
                query, k=self.candidate_k
            )
            for doc in vector_docs:
                for idx, orig_doc in enumerate(self.documents):
                    if orig_doc.page_content == doc.page_content:
                        vector_indices.add(idx)
                        break
        except Exception as e:
            print(f"Vector search error: {e}")

        # 3. 合并候选 (去重)
        all_candidate_indices = list(bm25_indices | vector_indices)

        if not all_candidate_indices:
            return []

        # 4. 排序
        if self.use_rerank and self.reranker:
            # 使用 Rerank 重排序
            candidate_docs = [self.documents[idx].page_content for idx in all_candidate_indices]
            rerank_results = self.reranker.rerank(query, candidate_docs, top_n=self.k)

            # 返回 Rerank 后的文档
            final_docs = []
            for orig_idx, score in rerank_results:
                if orig_idx < len(all_candidate_indices):
                    doc_idx = all_candidate_indices[orig_idx]
                    final_docs.append(self.documents[doc_idx])

            return final_docs[:self.k]
        else:
            # Fallback: RRF 融合
            bm25_scores = {idx: score for idx, score in bm25_results}
            vector_scores = {}
            try:
                vector_docs_with_scores = self.vector_store.similarity_search_with_relevance_scores(
                    query, k=self.candidate_k
                )
                for doc, score in vector_docs_with_scores:
                    for idx, orig_doc in enumerate(self.documents):
                        if orig_doc.page_content == doc.page_content:
                            vector_scores[idx] = score
                            break
            except Exception:
                pass

            combined = self._reciprocal_rank_fusion(bm25_scores, vector_scores)
            sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, score in sorted_results[:self.k]]

            return [self.documents[idx] for idx in top_indices]

    def _reciprocal_rank_fusion(
        self,
        bm25_scores: Dict[int, float],
        vector_scores: Dict[int, float],
        rrf_k: int = 60
    ) -> Dict[int, float]:
        """
        Combine scores using Reciprocal Rank Fusion (RRF).
        Used as fallback when Rerank is not available.
        """
        all_docs = set(bm25_scores.keys()) | set(vector_scores.keys())

        bm25_ranked = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        vector_ranked = sorted(vector_scores.items(), key=lambda x: x[1], reverse=True)

        bm25_ranks = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(bm25_ranked)}
        vector_ranks = {doc_idx: rank + 1 for rank, (doc_idx, _) in enumerate(vector_ranked)}

        combined = {}
        for doc_idx in all_docs:
            rrf_score = 0.0
            if doc_idx in bm25_ranks:
                rrf_score += (1 - self.alpha) / (rrf_k + bm25_ranks[doc_idx])
            if doc_idx in vector_ranks:
                rrf_score += self.alpha / (rrf_k + vector_ranks[doc_idx])
            combined[doc_idx] = rrf_score

        return combined

    def search_with_scores(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Tuple[Document, Dict[str, float]]]:
        """
        Search with detailed scores from each retriever.
        """
        k = k or self.k

        # BM25 召回
        bm25_results = self.bm25.search(query, top_k=self.candidate_k)
        bm25_indices = {idx for idx, score in bm25_results if score > 0}
        bm25_scores_dict = {idx: score for idx, score in bm25_results}

        # 向量召回
        vector_indices = set()
        vector_scores_dict = {}
        try:
            vector_docs = self.vector_store.similarity_search_with_relevance_scores(
                query, k=self.candidate_k
            )
            for doc, score in vector_docs:
                for idx, orig_doc in enumerate(self.documents):
                    if orig_doc.page_content == doc.page_content:
                        vector_indices.add(idx)
                        vector_scores_dict[idx] = score
                        break
        except Exception:
            pass

        # 合并候选
        all_candidate_indices = list(bm25_indices | vector_indices)

        if self.use_rerank and self.reranker:
            # Rerank
            candidate_docs = [self.documents[idx].page_content for idx in all_candidate_indices]
            rerank_results = self.reranker.rerank(query, candidate_docs, top_n=k)

            results = []
            for orig_idx, rerank_score in rerank_results:
                if orig_idx < len(all_candidate_indices):
                    doc_idx = all_candidate_indices[orig_idx]
                    doc = self.documents[doc_idx]
                    scores = {
                        'bm25_score': bm25_scores_dict.get(doc_idx, 0.0),
                        'vector_score': vector_scores_dict.get(doc_idx, 0.0),
                        'rerank_score': rerank_score
                    }
                    results.append((doc, scores))

            return results[:k]
        else:
            # RRF
            combined = self._reciprocal_rank_fusion(bm25_scores_dict, vector_scores_dict)
            sorted_results = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]

            results = []
            for doc_idx, combined_score in sorted_results:
                doc = self.documents[doc_idx]
                scores = {
                    'bm25_score': bm25_scores_dict.get(doc_idx, 0.0),
                    'vector_score': vector_scores_dict.get(doc_idx, 0.0),
                    'combined_score': combined_score
                }
                results.append((doc, scores))

            return results

    def set_alpha(self, alpha: float) -> None:
        """Set the weight for vector search (only for RRF fallback)."""
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = alpha
        print(f"Alpha set to {alpha} (only used when Rerank is disabled)")


class HybridRetrieverManager:
    """
    Manager class for creating and managing hybrid retrievers.
    """

    def __init__(self, persist_directory: str = "./hybrid_index"):
        """
        Initialize manager.

        Args:
            persist_directory: Directory to save hybrid index
        """
        self.persist_directory = persist_directory
        self.hybrid_retriever: Optional[HybridRetriever] = None
        self.documents: List[Document] = []
        self.bm25_index_path = os.path.join(persist_directory, "bm25_index.pkl")

    def create_from_documents(
        self,
        documents: List[Document],
        vector_store: Any,
        k: int = 4,
        alpha: float = 0.5
    ) -> HybridRetriever:
        """
        Create hybrid retriever from documents.

        Args:
            documents: List of Document objects
            vector_store: Vector store instance
            k: Number of documents to retrieve
            alpha: Vector search weight

        Returns:
            HybridRetriever instance
        """
        self.documents = documents

        self.hybrid_retriever = HybridRetriever(
            documents=documents,
            vector_store=vector_store,
            k=k,
            alpha=alpha
        )

        return self.hybrid_retriever

    def save(self) -> None:
        """Save hybrid retriever index."""
        if not self.hybrid_retriever:
            raise ValueError("No hybrid retriever to save")

        os.makedirs(self.persist_directory, exist_ok=True)

        # Save BM25 index
        self.hybrid_retriever.bm25.save(self.bm25_index_path)

        # Save document metadata
        doc_meta_path = os.path.join(self.persist_directory, "documents.pkl")
        with open(doc_meta_path, 'wb') as f:
            pickle.dump([
                (doc.page_content, doc.metadata)
                for doc in self.documents
            ], f)

        print(f"Hybrid index saved to {self.persist_directory}")

    def load(
        self,
        vector_store: Any,
        k: int = 4,
        alpha: float = 0.5
    ) -> HybridRetriever:
        """
        Load hybrid retriever from saved index.

        Args:
            vector_store: Vector store instance
            k: Number of documents to retrieve
            alpha: Vector search weight

        Returns:
            HybridRetriever instance
        """
        doc_meta_path = os.path.join(self.persist_directory, "documents.pkl")

        # Load documents
        with open(doc_meta_path, 'rb') as f:
            doc_data = pickle.load(f)

        self.documents = [
            Document(page_content=content, metadata=metadata)
            for content, metadata in doc_data
        ]

        # Create hybrid retriever
        self.hybrid_retriever = HybridRetriever(
            documents=self.documents,
            vector_store=vector_store,
            k=k,
            alpha=alpha
        )

        # Load BM25 index
        self.hybrid_retriever.bm25.load(self.bm25_index_path)

        print(f"Hybrid index loaded from {self.persist_directory}")
        return self.hybrid_retriever


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("  Hybrid Retriever Demo")
    print("=" * 60)

    # Create sample documents
    sample_docs = [
        Document(page_content="机器学习是人工智能的一个子集，它使系统能够从数据中学习。", metadata={"id": 1}),
        Document(page_content="深度学习使用多层神经网络来学习复杂的模式。", metadata={"id": 2}),
        Document(page_content="自然语言处理使计算机能够理解人类语言。", metadata={"id": 3}),
        Document(page_content="计算机视觉允许机器解释和理解视觉信息。", metadata={"id": 4}),
        Document(page_content="RAG是检索增强生成的缩写，它结合了检索和生成技术。", metadata={"id": 5}),
    ]

    print("\n1. Testing BM25 Retriever...")
    print("-" * 40)

    bm25_retriever = BM25Retriever(documents=sample_docs, k=3)
    results = bm25_retriever.invoke("什么是机器学习")

    print(f"Query: 什么是机器学习")
    print(f"BM25 Results ({len(results)}):")
    for i, doc in enumerate(results):
        print(f"  {i+1}. {doc.page_content[:50]}...")

    print("\n" + "=" * 60)
    print("  Hybrid Retriever Demo completed!")
    print("=" * 60)
