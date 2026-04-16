#!/usr/bin/env python3
"""
Vector store and embedding module for RAG system.
Uses ChromaDB for vector storage and sentence-transformers for embeddings.
Supports hybrid retrieval (BM25 + Vector search).
"""

import os
import shutil
import warnings
from typing import List, Optional, Any

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_core.documents import Document

# Set Hugging Face mirror for users in China (can be overridden by environment variable)
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


class VectorStore:
    """Vector store manager for RAG system."""

    def __init__(self,
                 persist_directory: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize vector store with embedding model.

        Args:
            persist_directory: Directory to persist vector database
            embedding_model: HuggingFace model name for embeddings
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.embeddings = None
        self.vector_store = None
        self._using_fallback = False
        self._documents: List[Document] = []  # Store documents for hybrid retrieval

        # Initialize embedding model
        self._init_embeddings()

    def _init_embeddings(self):
        """Initialize embeddings with error handling for offline scenarios."""
        print(f"Loading embedding model: {self.embedding_model}")

        # Try DashScope (Alibaba) embeddings first - best for Chinese
        if self._try_dashscope_embeddings():
            return

        # Try HuggingFace embeddings
        if self._try_huggingface_embeddings():
            return

        # Fallback to simple embeddings
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

            # Test if embeddings work
            test_embedding = self.embeddings.embed_query("test")
            if test_embedding and len(test_embedding) > 0:
                print("Embedding model loaded successfully!")
                return True

        except Exception as e:
            print(f"\nCould not load HuggingFace embeddings: {type(e).__name__}")
            self._print_network_solutions()

        return False

    def _try_dashscope_embeddings(self) -> bool:
        """Try to initialize DashScope (Alibaba) embeddings."""
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            return False

        try:
            from dashscope_llm import DashScopeEmbeddings

            print("Using Alibaba DashScope embeddings (optimized for Chinese)")
            self.embeddings = DashScopeEmbeddings(api_key=api_key)

            # Test if embeddings work
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
        print("   Linux/Mac: export HF_ENDPOINT=https://hf-mirror.com")
        print("3. Use a VPN or proxy if in restricted network")

    def _init_fallback_embeddings(self):
        """Initialize fallback embeddings when HuggingFace is not available."""
        print("\nUsing fallback embeddings (limited semantic search capability).")
        self._using_fallback = True

        try:
            # Try FakeEmbeddings
            from langchain_community.embeddings import FakeEmbeddings
            self.embeddings = FakeEmbeddings(size=384)
            print("Using FakeEmbeddings for testing.")
        except Exception:
            # Create custom minimal embeddings
            self.embeddings = self._create_minimal_embeddings()

    def _create_minimal_embeddings(self):
        """Create minimal embedding function as last resort."""
        from langchain_core.embeddings import Embeddings

        class MinimalEmbeddings(Embeddings):
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                # Simple hash-based embedding for basic functionality
                embeddings = []
                for text in texts:
                    # Create deterministic embedding from text hash
                    hash_val = hash(text) % (10 ** 8)
                    embedding = [float((hash_val >> (i % 32)) % 100) / 100 for i in range(384)]
                    embeddings.append(embedding)
                return embeddings

            def embed_query(self, text: str) -> List[float]:
                return self.embed_documents([text])[0]

        print("Using minimal embeddings (hash-based).")
        return MinimalEmbeddings()

    def _get_chroma_class(self):
        """Get the appropriate Chroma class."""
        try:
            # Try langchain-chroma first (recommended)
            from langchain_chroma import Chroma
            return Chroma, "langchain_chroma"
        except ImportError:
            pass

        try:
            # Fallback to langchain_community
            from langchain_community.vectorstores import Chroma
            return Chroma, "langchain_community"
        except ImportError:
            raise ImportError(
                "ChromaDB not found. Install with:\n"
                "  pip install langchain-chroma chromadb\n"
                "or:\n"
                "  pip install langchain-community chromadb"
            )

    def create_from_documents(self,
                              documents: List[Document],
                              collection_name: str = "rag_collection") -> None:
        """
        Create vector store from documents.

        Args:
            documents: List of Document objects
            collection_name: Name of the collection in vector database
        """
        if not documents:
            raise ValueError("No documents provided")

        print(f"\nCreating vector store with {len(documents)} documents...")

        # Store documents for hybrid retrieval
        self._documents = documents

        # Clean up existing directory if exists
        if os.path.exists(self.persist_directory):
            print(f"Cleaning up existing vector store at {self.persist_directory}")
            shutil.rmtree(self.persist_directory)

        # Get Chroma class
        Chroma, source = self._get_chroma_class()
        print(f"Using Chroma from: {source}")

        try:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                collection_name=collection_name
            )

            print(f"Vector store created at {self.persist_directory}")
            print(f"  Collection: {collection_name}")
            print(f"  Documents indexed: {len(documents)}")

            if self._using_fallback:
                print("  Warning: Using fallback embeddings - semantic search may be limited")

        except Exception as e:
            raise RuntimeError(f"Failed to create vector store: {e}")

    def load_existing(self, collection_name: str = "rag_collection") -> bool:
        """
        Load existing vector store from disk.

        Args:
            collection_name: Name of the collection to load

        Returns:
            True if loaded successfully, False otherwise
        """
        if not os.path.exists(self.persist_directory):
            print(f"Vector store not found at {self.persist_directory}")
            return False

        try:
            Chroma, _ = self._get_chroma_class()

            self.vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name
            )

            # Get collection info
            collection = self.vector_store._collection
            if collection:
                count = collection.count()
                print(f"Loaded existing vector store")
                print(f"  Location: {self.persist_directory}")
                print(f"  Collection: {collection_name}")
                print(f"  Document count: {count}")

                # Load documents from vector store for hybrid retrieval
                self._load_documents_from_collection()

                return True
            else:
                print("Failed to load collection")
                return False

        except Exception as e:
            print(f"Error loading vector store: {e}")
            return False

    def _load_documents_from_collection(self) -> None:
        """Load all documents from the collection for hybrid retrieval."""
        try:
            # Get all documents from the collection
            collection = self.vector_store._collection
            if collection:
                # Get all documents with their embeddings and metadata
                result = collection.get(include=['documents', 'metadatas'])

                self._documents = []
                for i, doc_content in enumerate(result['documents']):
                    metadata = result['metadatas'][i] if result['metadatas'] else {}
                    self._documents.append(Document(
                        page_content=doc_content,
                        metadata=metadata
                    ))

                print(f"  Loaded {len(self._documents)} documents for hybrid retrieval")

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
            k: Number of documents to retrieve (final output)
            alpha: Weight for vector search (0-1), only used when Rerank is disabled
            candidate_k: Number of candidates from each retriever (default 20)
            use_rerank: Whether to use Rerank for final ordering (default True)

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
        """
        Get all stored documents.

        Returns:
            List of Document objects
        """
        return self._documents

    def get_collection_info(self) -> dict:
        """
        Get information about the vector store collection.

        Returns:
            Dictionary with collection information
        """
        if not self.vector_store:
            raise RuntimeError("Vector store not initialized.")

        try:
            collection = self.vector_store._collection
            if collection:
                count = collection.count()
                return {
                    "document_count": count,
                    "persist_directory": self.persist_directory,
                    "embedding_model": self.embedding_model,
                    "using_fallback": self._using_fallback
                }
        except Exception:
            pass

        return {"error": "Unable to get collection information"}

    def delete_collection(self):
        """Delete the vector store collection."""
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
                print(f"Collection deleted from {self.persist_directory}")
            except Exception as e:
                print(f"Error deleting collection: {e}")

        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print(f"Removed vector store directory: {self.persist_directory}")


# Example usage and testing
if __name__ == "__main__":
    import tempfile

    print("Testing VectorStore...")
    print("=" * 60)

    # Create sample document
    test_content = """Machine learning is a subset of artificial intelligence.
Deep learning uses neural networks with multiple layers.
Natural language processing helps computers understand human language.
Retrieval-Augmented Generation combines retrieval with generation models.
Vector databases store embeddings for semantic search."""

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        sample_file = f.name

    try:
        # Import document loader
        from document_loader import DocumentLoader

        # Load documents
        loader = DocumentLoader(chunk_size=300, chunk_overlap=50)
        chunks = loader.load_text(sample_file)

        # Create vector store
        print("\nCreating vector store...")
        vector_store = VectorStore(persist_directory="./test_chroma_db")
        vector_store.create_from_documents(chunks, collection_name="test_collection")

        # Test search
        query = "What is deep learning?"
        print(f"\nSearching for: '{query}'")
        results = vector_store.similarity_search(query, k=2)

        print(f"\nFound {len(results)} results:")
        for i, doc in enumerate(results):
            print(f"\n  Result {i + 1}:")
            print(f"    Content: {doc.page_content[:100]}...")
            print(f"    Metadata: {doc.metadata}")

        # Get collection info
        info = vector_store.get_collection_info()
        print(f"\nCollection info: {info}")

        # Test loading existing
        print("\nTesting load existing...")
        new_vector_store = VectorStore(persist_directory="./test_chroma_db")
        if new_vector_store.load_existing("test_collection"):
            print("Successfully loaded existing vector store")

            # Test retriever
            retriever = new_vector_store.get_retriever(search_kwargs={"k": 2})
            print(f"Retriever created: {type(retriever)}")

        print("\nVectorStore test completed successfully!")

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        if os.path.exists(sample_file):
            os.unlink(sample_file)
        if os.path.exists("./test_chroma_db"):
            shutil.rmtree("./test_chroma_db")
            print("\nCleaned up test vector store")
