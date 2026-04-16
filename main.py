#!/usr/bin/env python3
"""
Main script for the RAG system.
Complete RAG pipeline with document loading, vector storage, and QA.
"""

import os
import sys
import argparse
import warnings
from typing import Optional, List
from pathlib import Path

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Add current directory to path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from document_loader import DocumentLoader
from vector_store import VectorStore
from rag_qa import RAGQA, LocalRAGQA
from config import get_vector_store_backend

# Import new production modules
try:
    from exceptions import (
        RAGException, ConfigurationError, DocumentError,
        VectorStoreError, LLMError, ValidationError,
        handle_exception
    )
    EXCEPTIONS_AVAILABLE = True
except ImportError:
    EXCEPTIONS_AVAILABLE = False

try:
    from cache import get_query_cache, get_embedding_cache, get_all_cache_stats
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

try:
    from metrics import metrics, health_checker, HealthStatus, Timer
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

try:
    from logger import info, warning, error as log_error
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False
    # Fallback to print
    def info(msg): print(f"[INFO] {msg}")
    def warning(msg): print(f"[WARNING] {msg}")
    def log_error(msg): print(f"[ERROR] {msg}")


class RAGSystem:
    """Complete RAG system orchestrator with production features."""

    def __init__(self,
                 vector_db_path: str = "./chroma_db",
                 embedding_model: str = "all-MiniLM-L6-v2",
                 vector_store_backend: str = None):
        """
        Initialize RAG system.

        Args:
            vector_db_path: Path to vector database (for ChromaDB)
            embedding_model: Embedding model name
            vector_store_backend: "chroma" or "postgres" (default: from config)
        """
        self.vector_db_path = vector_db_path
        self.embedding_model = embedding_model
        self.vector_store_backend = vector_store_backend or get_vector_store_backend()
        self.document_loader: Optional[DocumentLoader] = None
        self.vector_store = None
        self.qa_system: Optional[RAGQA] = None

        # Register health checks
        if METRICS_AVAILABLE:
            self._register_health_checks()

    def _register_health_checks(self):
        """Register health check functions."""
        from metrics import HealthCheckResult, check_vector_store_health

        def system_health():
            return HealthCheckResult(
                name="rag_system",
                status=HealthStatus.HEALTHY,
                message="RAG system initialized"
            )

        health_checker.register("system", system_health)

    def setup(self,
              chunk_size: int = 1000,
              chunk_overlap: int = 200) -> None:
        """
        Setup document loader and vector store.

        Args:
            chunk_size: Document chunk size
            chunk_overlap: Chunk overlap
        """
        timer = None
        if METRICS_AVAILABLE:
            timer = Timer(metrics, "setup_duration")
            timer.__enter__()

        try:
            self.document_loader = DocumentLoader(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            # Choose vector store backend
            if self.vector_store_backend == "postgres":
                from pgvector_store import PGVectorStore
                self.vector_store = PGVectorStore(
                    embedding_model=self.embedding_model
                )
                info("RAG System initialized with PostgreSQL backend.")
                print(f"  Vector Store: PostgreSQL (pgvector)")
            else:
                self.vector_store = VectorStore(
                    persist_directory=self.vector_db_path,
                    embedding_model=self.embedding_model
                )
                info("RAG System initialized with ChromaDB backend.")
                print(f"  Vector DB: {self.vector_db_path}")

            print(f"  Embedding model: {self.embedding_model}")
            print(f"  Chunk size: {chunk_size}, overlap: {chunk_overlap}")

            if METRICS_AVAILABLE:
                metrics.gauge("chunk_size", chunk_size)
                metrics.gauge("chunk_overlap", chunk_overlap)

        except Exception as e:
            log_error(f"Setup failed: {e}")
            if EXCEPTIONS_AVAILABLE:
                raise ConfigurationError(f"Failed to setup RAG system: {e}")
            raise
        finally:
            if timer:
                timer.__exit__(None, None, None)

    def index_documents(self,
                        source_path: str,
                        collection_name: str = "rag_collection") -> bool:
        """
        Index documents from file or directory.

        Args:
            source_path: Path to file or directory
            collection_name: Vector store collection name

        Returns:
            True if successful, False otherwise
        """
        if not self.document_loader or not self.vector_store:
            warning("System not setup. Call setup() first.")
            return False

        timer = None
        if METRICS_AVAILABLE:
            timer = Timer(metrics, "index_documents_duration")
            timer.__enter__()

        try:
            source_path = Path(source_path)
            if not source_path.exists():
                warning(f"Source path not found: {source_path}")
                return False

            # Load documents
            if source_path.is_file():
                if source_path.suffix.lower() == '.pdf':
                    documents = self.document_loader.load_pdf(str(source_path))
                elif source_path.suffix.lower() in ['.txt', '.md']:
                    documents = self.document_loader.load_text(str(source_path))
                else:
                    warning(f"Unsupported file type: {source_path.suffix}")
                    print("Supported types: .pdf, .txt, .md")
                    return False
            else:
                # Directory
                documents = self.document_loader.load_directory(str(source_path))

            if not documents:
                warning("No documents loaded.")
                return False

            # Create vector store
            self.vector_store.create_from_documents(documents, collection_name)

            if METRICS_AVAILABLE:
                metrics.increment("documents_indexed", len(documents))
                metrics.gauge("last_index_count", len(documents))

            info(f"Successfully indexed {len(documents)} documents")
            return True

        except Exception as e:
            log_error(f"Error indexing documents: {e}")
            import traceback
            traceback.print_exc()
            if METRICS_AVAILABLE:
                metrics.increment("index_errors")
            return False
        finally:
            if timer:
                timer.__exit__(None, None, None)

    def load_existing_index(self,
                            collection_name: str = "rag_collection") -> bool:
        """
        Load existing vector store index.

        Args:
            collection_name: Collection name to load

        Returns:
            True if successful, False otherwise
        """
        if not self.vector_store:
            self.vector_store = VectorStore(
                persist_directory=self.vector_db_path,
                embedding_model=self.embedding_model
            )

        success = self.vector_store.load_existing(collection_name)
        return success

    def create_qa_system(self,
                         model_name: str = "gpt-3.5-turbo",
                         temperature: float = 0.7,
                         use_conversation: bool = False,
                         use_local_model: bool = False,
                         search_k: int = 10,
                         optimize_query: bool = False,
                         use_hybrid: bool = True,
                         hybrid_alpha: float = 0.5,
                         candidate_k: int = 20,
                         use_rerank: bool = True,
                         llm_provider: str = "auto") -> bool:
        """
        Create QA system with loaded vector store.

        Args:
            model_name: LLM model name
            temperature: Model temperature
            use_conversation: Whether to use conversation memory
            use_local_model: Whether to use local model
            search_k: Number of documents to retrieve
            optimize_query: Whether to optimize user queries before retrieval
            use_hybrid: Whether to use hybrid retrieval (BM25 + Vector)
            hybrid_alpha: Weight for vector search in hybrid mode (0-1), only for RRF fallback
            candidate_k: Number of candidates from each retriever in hybrid mode
            use_rerank: Whether to use Rerank (default True)
            llm_provider: LLM provider ("auto", "openai", "dashscope", "local", "demo")

        Returns:
            True if successful, False otherwise
        """
        if not self.vector_store:
            print("Error: Vector store not loaded. Call load_existing_index() or index_documents() first.")
            return False

        try:
            # Get retriever (hybrid or vector only)
            if use_hybrid:
                print("Using hybrid retrieval (BM25 + Vector + Rerank)...")
                try:
                    retriever = self.vector_store.get_hybrid_retriever(
                        k=search_k,
                        alpha=hybrid_alpha,
                        candidate_k=candidate_k,
                        use_rerank=use_rerank
                    )
                except Exception as e:
                    print(f"Warning: Hybrid retrieval failed, falling back to vector search: {e}")
                    retriever = self.vector_store.get_retriever(
                        search_kwargs={"k": search_k}
                    )
            else:
                retriever = self.vector_store.get_retriever(
                    search_kwargs={"k": search_k}
                )

            # Create QA system
            if use_local_model:
                self.qa_system = LocalRAGQA(
                    retriever=retriever,
                    model_name=model_name,
                    temperature=temperature,
                    use_conversation=use_conversation,
                    optimize_query=optimize_query
                )
            else:
                self.qa_system = RAGQA(
                    retriever=retriever,
                    model_name=model_name,
                    temperature=temperature,
                    use_conversation=use_conversation,
                    optimize_query=optimize_query,
                    llm_provider=llm_provider
                )

            print("QA System created.")
            print(f"  Model: {model_name}")
            print(f"  LLM Provider: {llm_provider}")
            print(f"  Temperature: {temperature}")
            print(f"  Conversation mode: {use_conversation}")
            print(f"  Query optimization: {optimize_query}")
            print(f"  Hybrid retrieval: {use_hybrid}")
            if use_hybrid:
                print(f"  Candidate k (each retriever): {candidate_k}")
                print(f"  Use Rerank: {use_rerank}")
                if not use_rerank:
                    print(f"  Hybrid alpha (RRF): {hybrid_alpha}")
            print(f"  Search k (final): {search_k}")

            return True

        except Exception as e:
            print(f"Error creating QA system: {e}")
            import traceback
            traceback.print_exc()
            return False

    def ask(self,
            question: str,
            show_sources: bool = True,
            max_sources: int = 3) -> Optional[dict]:
        """
        Ask a question to the RAG system.

        Args:
            question: Question to ask
            show_sources: Whether to show source documents
            max_sources: Maximum number of sources to show

        Returns:
            Answer dictionary or None if error
        """
        if not self.qa_system:
            warning("QA system not created. Call create_qa_system() first.")
            return None

        timer = None
        if METRICS_AVAILABLE:
            timer = Timer(metrics, "query_duration")
            timer.__enter__()
            metrics.increment("queries_total")

        try:
            if show_sources:
                result = self.qa_system.ask_with_sources(question, k=max_sources)
            else:
                result = self.qa_system.ask(question)

            if METRICS_AVAILABLE and result:
                if result.get('error'):
                    metrics.increment("query_errors")
                else:
                    metrics.increment("query_success")

            return result

        except Exception as e:
            log_error(f"Error asking question: {e}")
            if METRICS_AVAILABLE:
                metrics.increment("query_errors")
            return None
        finally:
            if timer:
                timer.__exit__(None, None, None)

    def interactive_mode(self):
        """Start interactive question answering mode."""
        if not self.qa_system:
            print("Error: QA system not created.")
            return

        print("\n" + "=" * 60)
        print("Interactive RAG QA System")
        print("=" * 60)
        self._print_interactive_help()

        show_sources = True

        while True:
            try:
                question = input("\nQuestion: ").strip()

                # Handle commands
                cmd = question.lower()

                if cmd in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break

                elif cmd == 'clear':
                    self.qa_system.clear_memory()
                    continue

                elif cmd == 'sources on':
                    show_sources = True
                    print("Source display enabled.")
                    continue

                elif cmd == 'sources off':
                    show_sources = False
                    print("Source display disabled.")
                    continue

                elif cmd in ['help', 'h', '?']:
                    self._print_interactive_help()
                    continue

                elif cmd == 'info':
                    self._print_system_info()
                    continue

                elif cmd == 'history':
                    self._print_conversation_history()
                    continue

                elif cmd == 'stats':
                    self._print_stats()
                    continue

                elif cmd == 'health':
                    self._print_health()
                    continue

                elif cmd == 'cache':
                    self._print_cache_stats()
                    continue

                elif not question:
                    continue

                # Process question
                print("\n" + "-" * 60)
                print(f"Processing: {question}")
                print("-" * 60)

                result = self.ask(question, show_sources=show_sources)

                if result:
                    # Show if query was optimized
                    if result.get('query_was_optimized'):
                        print(f"\n[Query optimized: {result.get('search_question', question)}]")

                    print(f"\nAnswer: {result.get('answer', 'No answer')}")

                    if show_sources and result.get('formatted_sources'):
                        print("\nSources:")
                        for source in result['formatted_sources']:
                            print(f"\n  Source {source['source_id']}:")
                            print(f"    Content: {source['content']}")

                    # Show LLM type
                    llm_type = result.get('llm_type', 'unknown')
                    if llm_type == 'demo':
                        print("\n[Demo Mode - Add OPENAI_API_KEY for real AI responses]")

                print("-" * 60)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

    def _print_interactive_help(self):
        """Print interactive mode help."""
        print("\nCommands:")
        print("  quit/exit/q   - Exit the program")
        print("  help/h/?      - Show this help")
        print("  clear         - Clear conversation memory")
        print("  sources on/off- Toggle source display")
        print("  info          - Show system information")
        print("  history       - Show conversation history")
        print("  stats         - Show vector store statistics")
        print("  health        - Show system health status")
        print("  cache         - Show cache statistics")
        print("=" * 60)

    def _print_system_info(self):
        """Print system information."""
        print("\n" + "-" * 60)
        print("System Information:")
        print("-" * 60)

        if self.qa_system:
            info = self.qa_system.get_llm_info()
            print(f"  Model: {info.get('model_name', 'N/A')}")
            print(f"  LLM Type: {info.get('llm_type', 'N/A')}")
            print(f"  Temperature: {info.get('temperature', 'N/A')}")
            print(f"  Conversation: {'Enabled' if info.get('conversation_enabled') else 'Disabled'}")
            print(f"  Query Optimization: {'Enabled' if info.get('query_optimization_enabled') else 'Disabled'}")

        if self.vector_store:
            try:
                info = self.vector_store.get_collection_info()
                print(f"  Document Count: {info.get('document_count', 'N/A')}")
                print(f"  Embedding Model: {info.get('embedding_model', 'N/A')}")
            except Exception:
                pass

        print("-" * 60)

    def _print_conversation_history(self):
        """Print conversation history."""
        print("\n" + "-" * 60)
        print("Conversation History:")
        print("-" * 60)

        if not self.qa_system:
            print("  No QA system available.")
            return

        memory = self.qa_system.get_memory_contents()
        history = memory.get('chat_history', [])

        if not history:
            print("  No conversation history.")
        else:
            for i, turn in enumerate(history, 1):
                print(f"\n  Turn {i}:")
                print(f"    Q: {turn.get('question', 'N/A')}")
                answer = turn.get('answer', 'N/A')
                if len(answer) > 100:
                    answer = answer[:100] + "..."
                print(f"    A: {answer}")

        print("-" * 60)

    def _print_stats(self):
        """Print vector store statistics."""
        print("\n" + "-" * 60)
        print("Vector Store Statistics:")
        print("-" * 60)

        if not self.vector_store:
            print("  No vector store available.")
            return

        try:
            info = self.vector_store.get_collection_info()
            print(f"  Documents: {info.get('document_count', 'N/A')}")
            print(f"  Location: {info.get('persist_directory', 'N/A')}")
            print(f"  Embedding Model: {info.get('embedding_model', 'N/A')}")
            print(f"  Using Fallback: {info.get('using_fallback', False)}")
        except Exception as e:
            print(f"  Error getting stats: {e}")

        print("-" * 60)

    def _print_health(self):
        """Print system health status."""
        print("\n" + "-" * 60)
        print("System Health:")
        print("-" * 60)

        if METRICS_AVAILABLE:
            health_result = health_checker.check()
            print(f"  Overall Status: {health_result['status'].upper()}")

            for check in health_result.get('checks', []):
                status = check['status'].upper()
                name = check['name']
                latency = check.get('latency_ms', 'N/A')
                print(f"  - {name}: {status} ({latency:.1f}ms)" if isinstance(latency, float) else f"  - {name}: {status}")

            # Show metrics summary
            all_metrics = metrics.get_all_metrics()
            print(f"\n  Uptime: {all_metrics['uptime_seconds']:.1f}s")
            print(f"  CPU: {all_metrics['system']['cpu_percent']:.1f}%")
            print(f"  Memory: {all_metrics['system']['memory_percent']:.1f}%")
        else:
            print("  Health monitoring not available")

        print("-" * 60)

    def _print_cache_stats(self):
        """Print cache statistics."""
        print("\n" + "-" * 60)
        print("Cache Statistics:")
        print("-" * 60)

        if CACHE_AVAILABLE:
            stats = get_all_cache_stats()
            for cache_name, cache_stats in stats.items():
                print(f"\n  {cache_name}:")
                print(f"    Size: {cache_stats['size']}/{cache_stats['max_size']}")
                print(f"    Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")
                print(f"    Hit Rate: {cache_stats['hit_rate']}")
        else:
            print("  Cache not available")

        print("-" * 60)


def create_sample_documents():
    """Create sample documents for demonstration."""
    sample_dir = Path("./sample_documents")
    sample_dir.mkdir(exist_ok=True)

    # Create sample text file about AI
    sample_text = sample_dir / "ai_basics.txt"
    with open(sample_text, 'w', encoding='utf-8') as f:
        f.write("""Artificial Intelligence (AI) Basics

Artificial Intelligence refers to the simulation of human intelligence in machines.
These machines are programmed to think like humans and mimic their actions.

Machine Learning (ML) is a subset of AI that enables systems to learn and improve from experience.
Instead of being explicitly programmed, ML algorithms use statistical techniques to learn patterns.

Deep Learning is a subset of machine learning that uses neural networks with many layers.
These deep neural networks can learn complex patterns from large amounts of data.

Natural Language Processing (NLP) enables computers to understand, interpret, and generate human language.
Applications include chatbots, translation, and sentiment analysis.

Computer Vision enables machines to interpret and understand visual information from the world.
It's used in facial recognition, object detection, and autonomous vehicles.

Robotics combines AI with mechanical engineering to create intelligent machines.
Robots can perform tasks autonomously or with human guidance.

These technologies are transforming industries and creating new possibilities.
""")

    # Create sample text file about RAG
    sample_rag = sample_dir / "rag_overview.txt"
    with open(sample_rag, 'w', encoding='utf-8') as f:
        f.write("""Retrieval-Augmented Generation (RAG) Overview

RAG combines retrieval-based methods with generative language models.
The system first retrieves relevant documents from a knowledge base.
Then it uses a language model to generate answers based on the retrieved content.

Key Components:
1. Document Loader: Loads and parses documents from various formats
2. Text Splitter: Splits documents into manageable chunks
3. Vector Store: Stores document embeddings for semantic search
4. Retriever: Finds relevant documents for a given query
5. Generator: Language model that generates answers

Benefits:
- Reduces hallucination in language models
- Improves factual accuracy
- Allows incorporation of up-to-date information
- Enables domain-specific knowledge integration

Applications:
- Question answering systems
- Research assistants
- Customer support chatbots
- Document analysis tools
""")

    print(f"Created sample documents in {sample_dir}")
    return str(sample_dir)


def print_banner():
    """Print application banner."""
    print("\n" + "=" * 60)
    print("  RAG System - Retrieval-Augmented Generation")
    print("  Version 1.0.0")
    print("=" * 60)


def main():
    """Main entry point for the RAG system."""
    print_banner()

    parser = argparse.ArgumentParser(
        description="RAG System - Retrieval-Augmented Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode demo                          # Run demo
  python main.py --mode index --source ./docs         # Index documents
  python main.py --mode query --question "What is?"   # Ask a question
  python main.py --mode interactive                   # Interactive mode
        """
    )
    parser.add_argument("--mode", choices=["index", "query", "interactive", "demo"],
                        default="demo", help="Operation mode (default: demo)")
    parser.add_argument("--source", type=str, help="Source file or directory for indexing")
    parser.add_argument("--db", type=str, default="./chroma_db", help="Vector database path")
    parser.add_argument("--collection", type=str, default="rag_collection", help="Collection name")
    parser.add_argument("--question", type=str, help="Question to ask (for query mode)")
    parser.add_argument("--model", type=str, default="qwen-plus", help="LLM model name")
    parser.add_argument("--provider", type=str, default="auto",
                        choices=["auto", "openai", "dashscope", "local", "demo"],
                        help="LLM provider (default: auto)")
    parser.add_argument("--local", action="store_true", help="Use local model (if available)")
    parser.add_argument("--temp", type=float, default=0.7, help="Model temperature")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Document chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap")
    parser.add_argument("--optimize-query", action="store_true",
                        help="Enable query optimization before retrieval")
    parser.add_argument("--no-hybrid", action="store_true",
                        help="Disable hybrid retrieval (use vector only)")
    parser.add_argument("--hybrid-alpha", type=float, default=0.5,
                        help="Weight for vector search in RRF mode (0-1, default 0.5)")
    parser.add_argument("--candidate-k", type=int, default=20,
                        help="Number of candidates from each retriever (default 20)")
    parser.add_argument("--no-rerank", action="store_true",
                        help="Disable Rerank, use RRF fusion instead")
    parser.add_argument("--search-k", type=int, default=10,
                        help="Number of final documents to retrieve (default 10)")

    args = parser.parse_args()

    # Create RAG system
    rag = RAGSystem(vector_db_path=args.db)

    if args.mode == "demo":
        print("\nRunning demo mode...")
        print("-" * 60)

        # Create sample documents
        sample_dir = create_sample_documents()

        # Setup system
        rag.setup(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

        # Index sample documents
        print("\nIndexing sample documents...")
        if rag.index_documents(sample_dir, args.collection):
            # Create QA system
            print("\nCreating QA system...")
            if rag.create_qa_system(
                model_name=args.model,
                temperature=args.temp,
                use_local_model=args.local,
                search_k=args.search_k,
                optimize_query=args.optimize_query,
                use_hybrid=not args.no_hybrid,
                hybrid_alpha=args.hybrid_alpha,
                candidate_k=args.candidate_k,
                use_rerank=not args.no_rerank,
                llm_provider=args.provider
            ):
                # Ask demo questions
                demo_questions = [
                    "What is artificial intelligence?",
                    "What is Retrieval-Augmented Generation?",
                    "What are the benefits of RAG?"
                ]

                for question in demo_questions:
                    print(f"\n{'=' * 60}")
                    print(f"Question: {question}")
                    result = rag.ask(question, show_sources=True)
                    if result:
                        print(f"\nAnswer: {result.get('answer', 'No answer')}")
                        if result.get('formatted_sources'):
                            print("\nTop source:")
                            source = result['formatted_sources'][0]
                            print(f"  Content: {source['content'][:150]}...")
                    print('=' * 60)

                print("\nDemo completed successfully!")
                print(f"\nTo continue with interactive mode, run:")
                print(f"  python main.py --mode interactive")
        else:
            print("Demo failed to index documents.")

    elif args.mode == "index":
        if not args.source:
            print("Error: --source required for index mode")
            print("Example: python main.py --mode index --source ./my_documents")
            return

        print(f"Indexing documents from: {args.source}")
        rag.setup(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)

        if rag.index_documents(args.source, args.collection):
            print(f"\nIndexing completed successfully!")
            print(f"Vector store saved to: {args.db}")
            print(f"\nTo query, run:")
            print(f"  python main.py --mode query --question \"Your question\"")
        else:
            print("Indexing failed.")

    elif args.mode == "query":
        if not args.question:
            print("Error: --question required for query mode")
            print("Example: python main.py --mode query --question \"What is AI?\"")
            return

        print(f"Loading existing vector store from: {args.db}")
        if rag.load_existing_index(args.collection):
            if rag.create_qa_system(
                model_name=args.model,
                temperature=args.temp,
                use_local_model=args.local,
                search_k=args.search_k,
                optimize_query=args.optimize_query,
                use_hybrid=not args.no_hybrid,
                hybrid_alpha=args.hybrid_alpha,
                candidate_k=args.candidate_k,
                use_rerank=not args.no_rerank,
                llm_provider=args.provider
            ):
                print(f"\nAsking: {args.question}")
                print("-" * 60)
                result = rag.ask(args.question, show_sources=True)

                if result:
                    print(f"\nAnswer: {result.get('answer', 'No answer')}")

                    if result.get('formatted_sources'):
                        print("\nSources:")
                        for source in result['formatted_sources']:
                            print(f"\n  Source {source['source_id']}:")
                            print(f"    Content: {source['content']}")
                else:
                    print("No answer received.")
            else:
                print("Failed to create QA system.")
        else:
            print(f"Failed to load vector store from {args.db}")
            print("You may need to index documents first with:")
            print("  python main.py --mode index --source <path>")

    elif args.mode == "interactive":
        print(f"Loading existing vector store from: {args.db}")
        if rag.load_existing_index(args.collection):
            if rag.create_qa_system(
                model_name=args.model,
                temperature=args.temp,
                use_local_model=args.local,
                use_conversation=True,
                optimize_query=args.optimize_query,
                use_hybrid=args.hybrid,
                hybrid_alpha=args.hybrid_alpha,
                candidate_k=args.candidate_k,
                use_rerank=not args.no_rerank,
                llm_provider=args.provider
            ):
                rag.interactive_mode()
            else:
                print("Failed to create QA system.")
        else:
            print(f"Failed to load vector store from {args.db}")
            print("You may need to index documents first with:")
            print("  python main.py --mode index --source <path>")


if __name__ == "__main__":
    main()
