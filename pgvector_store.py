#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PostgreSQL Vector Store Module for RAG System.
Uses pgvector extension for vector storage and sentence-transformers for embeddings.
"""

import os
import warnings
from typing import List, Optional, Any, Dict

from langchain_core.documents import Document

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set Hugging Face mirror for users in China
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class PGVectorStore:
    """PostgreSQL vector store manager using pgvector extension."""

    def __init__(self,
                 connection_string: str = None,
                 embedding_model: str = "all-MiniLM-L6-v2",
                 collection_name: str = "rag_collection"):
        """
        Initialize PostgreSQL vector store.

        Args:
            connection_string: PostgreSQL connection string
                Format: postgresql://user:password@host:port/database
            embedding_model: HuggingFace model name for embeddings
            collection_name: Name of the table/collection
        """
        self.connection_string = connection_string or self._build_connection_string()
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.embeddings = None
        self.vector_store = None
        self._using_fallback = False
        self._documents: List[Document] = []

        # Initialize embedding model
        self._init_embeddings()

    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from environment variables."""
        from config import Config

        host = Config.get("PG_HOST", "localhost")
        port = Config.get("PG_PORT", "5432")
        user = Config.get("PG_USER", "postgres")
        password = Config.get("PG_PASSWORD", "123456")
        database = Config.get("PG_DATABASE", "postgres")

        return f"postgresql://{user}:{password}@{host}:{port}/{database}"

    def _init_embeddings(self):
        """Initialize embeddings with error handling."""
        print(f"Loading embedding model: {self.embedding_model}")

        # Try DashScope embeddings first
        if self._try_dashscope_embeddings():
            return

        # Try HuggingFace embeddings
        if self._try_huggingface_embeddings():
            return

        # Fallback
        self._init_fallback_embeddings()

    def _try_huggingface_embeddings(self) -> bool:
        """Try to initialize HuggingFace embeddings."""
        try:
            from langchain_huggingface import HuggingFaceEmbeddings

            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            test_embedding = self.embeddings.embed_query("test")
            if test_embedding and len(test_embedding) > 0:
                print("Embedding model loaded successfully!")
                return True

        except Exception as e:
            print(f"\nCould not load HuggingFace embeddings: {type(e).__name__}")
            self._print_network_solutions()

        return False

    def _try_dashscope_embeddings(self) -> bool:
        """Try DashScope embeddings."""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return False

        try:
            from dashscope_llm import DashScopeEmbeddings

            print("Using Alibaba DashScope embeddings (optimized for Chinese)")
            self.embeddings = DashScopeEmbeddings(api_key=api_key)

            test_embedding = self.embeddings.embed_query("测试")
            if test_embedding and len(test_embedding) > 0:
                print("DashScope embeddings loaded successfully!")
                return True

        except Exception as e:
            print(f"DashScope embeddings error: {e}")

        return False

    def _print_network_solutions(self):
        """Print network troubleshooting solutions."""
        print("\nPossible solutions:")
        print("1. Check your internet connection")
        print("2. Set HF_ENDPOINT environment variable:")
        print("   Windows CMD: set HF_ENDPOINT=https://hf-mirror.com")
        print("   PowerShell: $env:HF_ENDPOINT='https://hf-mirror.com'")

    def _init_fallback_embeddings(self):
        """Initialize fallback embeddings."""
        print("\nUsing fallback embeddings (limited semantic search capability).")
        self._using_fallback = True

        try:
            from langchain_community.embeddings import FakeEmbeddings
            self.embeddings = FakeEmbeddings(size=384)
            print("Using FakeEmbeddings for testing.")
        except Exception:
            self.embeddings = self._create_minimal_embeddings()

    def _create_minimal_embeddings(self):
        """Create minimal embedding function."""
        from langchain_core.embeddings import Embeddings

        class MinimalEmbeddings(Embeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                embeddings = []
                for text in texts:
                    hash_val = hash(text) % (10 ** 8)
                    embedding = [float((hash_val >> (i % 32)) % 100) / 100 for i in range(384)]
                    embeddings.append(embedding)
                return embeddings

            def embed_query(self, text: str) -> List[float]:
                return self.embed_documents([text])[0]

        print("Using minimal embeddings (hash-based).")
        return MinimalEmbeddings()

    def _get_pgvector_class(self):
        """Get PGVector class from langchain."""
        try:
            from langchain_community.vectorstores import PGVector
            return PGVector
        except ImportError:
            raise ImportError(
                "PGVector not found. Install with:\n"
                "  pip install psycopg2-binary pgvector langchain-community\n"
                "Also ensure PostgreSQL has pgvector extension installed."
            )

    def create_from_documents(self,
                              documents: List[Document],
                              collection_name: str = None) -> None:
        """
        Create vector store from documents.

        Args:
            documents: List of Document objects
            collection_name: Name of the table (optional)
        """
        if not documents:
            raise ValueError("No documents provided")

        if collection_name:
            self.collection_name = collection_name

        print(f"\nCreating PostgreSQL vector store with {len(documents)} documents...")
        print(f"  Collection/Table: {self.collection_name}")

        # Store documents for hybrid retrieval
        self._documents = documents

        PGVector = self._get_pgvector_class()

        try:
            # Drop existing table if exists
            self._drop_table_if_exists()

            self.vector_store = PGVector.from_documents(
                documents=documents,
                embedding=self.embeddings,
                connection_string=self.connection_string,
                collection_name=self.collection_name,
                pre_collection_name=self.collection_name,
            )

            print(f"Vector store created in PostgreSQL")
            print(f"  Table: {self.collection_name}")
            print(f"  Documents indexed: {len(documents)}")

            if self._using_fallback:
                print("  Warning: Using fallback embeddings - semantic search may be limited")

        except Exception as e:
            raise RuntimeError(f"Failed to create PostgreSQL vector store: {e}")

    def _drop_table_if_exists(self):
        """Drop existing table if it exists."""
        import psycopg2

        try:
            conn = psycopg2.connect(self.connection_string)
            cur = conn.cursor()

            # Drop the collection table
            cur.execute(f"DROP TABLE IF EXISTS {self.collection_name} CASCADE")

            # Drop the langchain_pg_collection and langchain_pg_embedding tables if they exist
            cur.execute("DROP TABLE IF EXISTS langchain_pg_collection CASCADE")
            cur.execute("DROP TABLE IF EXISTS langchain_pg_embedding CASCADE")

            conn.commit()
            cur.close()
            conn.close()

        except Exception as e:
            print(f"  Note: Could not drop existing table: {e}")

    def load_existing(self, collection_name: str = None) -> bool:
        """
        Load existing vector store.

        Args:
            collection_name: Name of the collection to load

        Returns:
            True if loaded successfully, False otherwise
        """
        if collection_name:
            self.collection_name = collection_name

        try:
            PGVector = self._get_pgvector_class()

            self.vector_store = PGVector(
                connection_string=self.connection_string,
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )

            # Get document count
            count = self._get_document_count()
            print(f"Loaded existing PostgreSQL vector store")
            print(f"  Table: {self.collection_name}")
            print(f"  Document count: {count}")

            # Load documents for hybrid retrieval
            self._load_documents_from_store()

            return True

        except Exception as e:
            print(f"Error loading PostgreSQL vector store: {e}")
            return False

    def _get_document_count(self) -> int:
        """Get document count from PostgreSQL."""
        import psycopg2

        try:
            conn = psycopg2.connect(self.connection_string)
            cur = conn.cursor()

            cur.execute(f"SELECT COUNT(*) FROM {self.collection_name}")
            count = cur.fetchone()[0]

            cur.close()
            conn.close()

            return count

        except Exception:
            return 0

    def _load_documents_from_store(self) -> None:
        """Load all documents from PostgreSQL for hybrid retrieval."""
        import psycopg2

        try:
            conn = psycopg2.connect(self.connection_string)
            cur = conn.cursor()

            # Query documents (adjust column names based on PGVector schema)
            cur.execute(f"""
                SELECT document, cmetadata
                FROM {self.collection_name}
            """)

            rows = cur.fetchall()

            self._documents = []
            for row in rows:
                content = row[0]
                metadata = row[1] if row[1] else {}
                self._documents.append(Document(
                    page_content=content,
                    metadata=metadata
                ))

            print(f"  Loaded {len(self._documents)} documents for hybrid retrieval")

            cur.close()
            conn.close()

        except Exception as e:
            print(f"  Warning: Could not load documents for hybrid retrieval: {e}")
            self._documents = []

    def similarity_search(self,
                          query: str,
                          k: int = 4) -> List[Document]:
        """
        Search for similar documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of similar documents
        """
        if not self.vector_store:
            raise RuntimeError(
                "Vector store not initialized. "
                "Call create_from_documents() or load_existing() first."
            )

        results = self.vector_store.similarity_search(query, k=k)
        return results

    def similarity_search_with_scores(self,
                                       query: str,
                                       k: int = 4) -> List[tuple]:
        """
        Search for similar documents with similarity scores.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of tuples (document, score)
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized.")

        results = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        return results

    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a retriever for use with LangChain chains.

        Args:
            search_kwargs: Additional search parameters

        Returns:
            Retriever object
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized.")

        if search_kwargs is None:
            search_kwargs = {"k": 4}

        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

    def get_hybrid_retriever(self,
                             k: int = 4,
                             alpha: float = 0.5,
                             candidate_k: int = 20,
                             use_rerank: bool = True) -> Any:
        """
        Get a hybrid retriever combining BM25 and vector search.

        Args:
            k: Number of documents to retrieve
            alpha: Weight for vector search (0-1)
            candidate_k: Number of candidates from each retriever
            use_rerank: Whether to use Rerank

        Returns:
            HybridRetriever instance
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized.")

        if not self._documents:
            raise RuntimeError("No documents stored. Create vector store first.")

        try:
            from hybrid_retriever import HybridRetriever

            hybrid_retriever = HybridRetriever(
                documents=self._documents,
                vector_store=self.vector_store,
                k=k,
                candidate_k=candidate_k,
                use_rerank=use_rerank,
                alpha=alpha
            )

            return hybrid_retriever

        except ImportError:
            print("Warning: hybrid_retriever not available. Falling back to vector retriever.")
            return self.get_retriever(search_kwargs={"k": k})

    def get_documents(self) -> List[Document]:
        """Get all stored documents."""
        return self._documents

    def get_collection_info(self) -> dict:
        """Get information about the vector store."""
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized.")

        try:
            count = self._get_document_count()
            return {
                "document_count": count,
                "collection_name": self.collection_name,
                "embedding_model": self.embedding_model,
                "using_fallback": self._using_fallback,
                "backend": "PostgreSQL (pgvector)"
            }
        except Exception:
            return {"error": "Unable to get collection information"}

    def delete_collection(self):
        """Delete the vector store collection/table."""
        import psycopg2

        try:
            conn = psycopg2.connect(self.connection_string)
            cur = conn.cursor()

            cur.execute(f"DROP TABLE IF EXISTS {self.collection_name} CASCADE")
            conn.commit()

            print(f"Table '{self.collection_name}' deleted from PostgreSQL")

            cur.close()
            conn.close()

        except Exception as e:
            print(f"Error deleting table: {e}")


def test_connection():
    """Test PostgreSQL connection and pgvector extension."""
    import psycopg2

    from config import Config

    host = Config.get("PG_HOST", "localhost")
    port = Config.get("PG_PORT", "5432")
    user = Config.get("PG_USER", "postgres")
    password = Config.get("PG_PASSWORD", "123456")
    database = Config.get("PG_DATABASE", "postgres")

    print(f"Testing PostgreSQL connection...")
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  User: {user}")
    print(f"  Database: {database}")

    conn_str = f"postgresql://{user}:{password}@{host}:{port}/{database}"

    try:
        conn = psycopg2.connect(conn_str)
        cur = conn.cursor()

        # Check pgvector extension
        cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
        result = cur.fetchone()

        if result:
            print("\n[OK] pgvector extension is installed")
        else:
            print("\n[!] pgvector extension not found, trying to install...")
            try:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                conn.commit()
                print("[OK] pgvector extension installed successfully")
            except Exception as e:
                print(f"[ERROR] Could not install pgvector: {e}")
                print("Please run: CREATE EXTENSION vector; in PostgreSQL")

        cur.close()
        conn.close()

        print("\n[OK] PostgreSQL connection successful!")
        return True

    except Exception as e:
        print(f"\n[ERROR] Connection failed: {e}")
        return False


if __name__ == "__main__":
    print("Testing PGVectorStore...")
    print("=" * 60)

    # Test connection first
    if not test_connection():
        print("\nPlease check your PostgreSQL configuration.")
        exit(1)

    print("\n" + "=" * 60)
    print("PGVectorStore module ready!")
