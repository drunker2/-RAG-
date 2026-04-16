#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrieval-Augmented Generation (RAG) QA module.
Uses language models to generate answers based on retrieved documents.
Supports OpenAI, Alibaba DashScope (Qwen), local models, and demonstration mode.
Includes query optimization for better retrieval results.
Supports hybrid retrieval (BM25 + Vector search).
"""

import os
import warnings
from typing import Dict, Any, Optional, List

from dotenv import load_dotenv

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Import query optimizer
try:
    from query_optimizer import QueryOptimizer
    QUERY_OPTIMIZER_AVAILABLE = True
except ImportError:
    QUERY_OPTIMIZER_AVAILABLE = False
    print("Warning: query_optimizer module not found. Query optimization disabled.")

# Import hybrid retriever
try:
    from hybrid_retriever import HybridRetriever, HybridRetrieverManager
    HYBRID_RETRIEVER_AVAILABLE = True
except ImportError:
    HYBRID_RETRIEVER_AVAILABLE = False
    print("Warning: hybrid_retriever module not found. Hybrid retrieval disabled.")

# Import Alibaba DashScope
try:
    from dashscope_llm import DashScopeLLM, check_dashscope_credentials
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False


class RAGQA:
    """RAG Question Answering system with multiple LLM provider support."""

    def __init__(self,
                 retriever,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0.7,
                 use_conversation: bool = False,
                 system_prompt: Optional[str] = None,
                 optimize_query: bool = False,
                 query_optimizer_model: Optional[str] = None,
                 llm_provider: str = "auto"):
        """
        Initialize RAG QA system.

        Args:
            retriever: Vector store retriever
            model_name: LLM model name
            temperature: Model temperature (0-1)
            use_conversation: Enable conversation memory
            system_prompt: Custom system prompt for the LLM
            optimize_query: Enable query optimization before retrieval
            query_optimizer_model: Model for query optimization (defaults to model_name)
            llm_provider: LLM provider ("auto", "openai", "dashscope", "local", "demo")
                          "auto" will try providers in order: OpenAI -> DashScope -> Demo
        """
        self.retriever = retriever
        self.model_name = model_name
        self.temperature = temperature
        self.use_conversation = use_conversation
        self.conversation_history = []
        self.system_prompt = system_prompt
        self._llm_type = "unknown"
        self.llm_provider = llm_provider

        # Query optimization settings
        self.optimize_query = optimize_query
        self.query_optimizer = None

        # Initialize query optimizer if enabled
        if self.optimize_query and QUERY_OPTIMIZER_AVAILABLE:
            optimizer_model = query_optimizer_model or model_name
            self.query_optimizer = QueryOptimizer(
                model_name=optimizer_model,
                temperature=0.3,  # Lower temperature for more consistent optimization
                enabled=True
            )
            if not self.query_optimizer.is_enabled():
                print("Warning: Query optimizer initialized but not available (no LLM)")
                self.query_optimizer = None

        # Initialize LLM
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        """Initialize the language model based on provider setting."""
        # If provider is specified explicitly
        if self.llm_provider == "dashscope":
            return self._try_dashscope()
        elif self.llm_provider == "openai":
            return self._try_openai(os.getenv("OPENAI_API_KEY")) or self._create_demo_llm()
        elif self.llm_provider == "local":
            return self._try_local_model() or self._create_demo_llm()
        elif self.llm_provider == "demo":
            return self._create_demo_llm()

        # Auto mode: try providers in order
        # 1. Try OpenAI first
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            llm = self._try_openai(openai_api_key)
            if llm:
                return llm

        # 2. Try Alibaba DashScope (Qwen)
        if DASHSCOPE_AVAILABLE and check_dashscope_credentials():
            llm = self._try_dashscope()
            if llm:
                return llm

        # 3. Try local models
        llm = self._try_local_model()
        if llm:
            return llm

        # 4. Fall back to demonstration mode
        return self._create_demo_llm()

    def _try_dashscope(self):
        """Try to initialize Alibaba DashScope (Qwen) model."""
        try:
            # 确定使用哪个模型
            model = self.model_name
            # 如果传入的是 OpenAI 模型名，自动切换到通义千问对应模型
            if model.startswith("gpt"):
                model = "qwen-plus"  # 默认使用 qwen-plus
                print(f"检测到 OpenAI 模型名，自动切换到通义千问模型: {model}")

            self._llm_type = "dashscope"
            print(f"使用阿里云通义千问模型: {model}")

            return DashScopeLLM(
                model=model,
                temperature=self.temperature
            )

        except ImportError:
            print("DashScope SDK 未安装。请运行: pip install dashscope")
        except Exception as e:
            print(f"阿里云通义千问初始化错误: {e}")

        return None

    def _try_openai(self, api_key: str):
        """Try to initialize OpenAI model."""
        try:
            from langchain_openai import ChatOpenAI
            self._llm_type = "openai"
            print(f"Using OpenAI model: {self.model_name}")
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                openai_api_key=api_key
            )
        except ImportError:
            pass
        except Exception as e:
            print(f"OpenAI initialization error: {e}")

        return None

    def _try_local_model(self):
        """Try to initialize a local model."""
        # Check if user wants local model
        use_local = os.getenv("USE_LOCAL_MODEL", "").lower() == "true"
        if not use_local:
            return None

        try:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import pipeline

            print(f"Loading local model: {self.model_name}")

            pipe = pipeline(
                "text-generation",
                model=self.model_name,
                max_new_tokens=512,
                temperature=self.temperature
            )

            self._llm_type = "local"
            return HuggingFacePipeline(pipeline=pipe)

        except ImportError:
            print("Local model requires: pip install transformers torch")
        except Exception as e:
            print(f"Local model error: {e}")

        return None

    def _create_demo_llm(self):
        """Create demonstration LLM for testing without API key."""
        self._llm_type = "demo"
        self._print_demo_warning()

        class DemoLLM:
            """Demo LLM that generates placeholder responses."""

            def invoke(self, prompt, **kwargs):
                from langchain_core.messages import AIMessage
                # Generate a demo response
                return AIMessage(content=self._generate_response(prompt))

            def _generate_response(self, prompt):
                """Generate a demo response based on the prompt."""
                if "context" in str(prompt).lower() and "question" in str(prompt).lower():
                    return (
                        "[Demo Mode] This is a simulated response.\n\n"
                        "To get real AI responses:\n"
                        "1. Get an OpenAI API key from https://platform.openai.com/\n"
                        "2. Add it to your .env file: OPENAI_API_KEY=your_key_here\n"
                        "3. Run the program again\n\n"
                        "The retrieval system is working correctly - "
                        "you can see the relevant document sources below."
                    )
                return "[Demo Mode] Simulated response. Add OPENAI_API_KEY for real AI responses."

            def __call__(self, prompt, **kwargs):
                return self.invoke(prompt, **kwargs)

        return DemoLLM()

    def _print_demo_warning(self):
        """Print warning about demo mode."""
        print("=" * 60)
        print("  DEMONSTRATION MODE")
        print("=" * 60)
        print("No API key found. Using demo mode.")
        print()
        print("To enable real AI responses:")
        print("Option 1 - OpenAI:")
        print("  1. Get an API key from https://platform.openai.com/")
        print("  2. Add to .env file: OPENAI_API_KEY=sk-your-key-here")
        print()
        print("Option 2 - Alibaba DashScope (Qwen):")
        print("  1. Get an API key from https://dashscope.console.aliyun.com/")
        print("  2. Add to .env file: DASHSCOPE_API_KEY=your-key-here")
        print()
        print("The RAG retrieval will still work and show sources.")
        print("=" * 60)

    def _build_prompt(self, context: str, question: str) -> str:
        """Build the prompt with context and question."""
        if self.system_prompt:
            base_prompt = self.system_prompt
        else:
            base_prompt = """Based on the following context, please answer the question.
If the context doesn't contain relevant information, say "I don't have enough information to answer this question based on the provided context."

Context:
{context}

Question: {question}

Please provide a clear and concise answer:"""

        return base_prompt.format(context=context, question=question)

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with sources.

        Args:
            question: The question to ask

        Returns:
            Dictionary with answer and source documents
        """
        if not question or not question.strip():
            return {"answer": "Please provide a question.", "sources": [], "question": ""}

        # Track original and optimized questions
        original_question = question
        search_question = question

        # Optimize query if enabled
        if self.query_optimizer:
            # Build context from conversation history
            context = None
            if self.use_conversation and self.conversation_history:
                context = "\n".join([
                    f"Q: {h['question']}\nA: {h['answer'][:100]}..."
                    for h in self.conversation_history[-3:]
                ])

            # Optimize the question
            opt_result = self.query_optimizer.optimize(question, context)
            search_question = opt_result["optimized_question"]

        try:
            # Retrieve relevant documents using (possibly optimized) question
            docs = self._retrieve_documents(search_question)

            # Combine context
            context = "\n\n".join([doc.page_content for doc in docs])

            # Build prompt
            prompt = self._build_prompt(context, question)

            # Add conversation history if enabled
            if self.use_conversation and self.conversation_history:
                history_text = "\n".join([
                    f"Previous Q: {h['question']}\nPrevious A: {h['answer'][:100]}..."
                    for h in self.conversation_history[-3:]
                ])
                prompt = f"Conversation history:\n{history_text}\n\n{prompt}"

            # Get response from LLM
            answer = self._get_llm_response(prompt)

            # Store in conversation history
            if self.use_conversation:
                self.conversation_history.append({
                    "question": question,
                    "answer": answer
                })

            return {
                "answer": answer,
                "sources": docs,
                "question": question,
                "original_question": original_question,
                "search_question": search_question,
                "query_was_optimized": search_question != original_question,
                "llm_type": self._llm_type
            }

        except Exception as e:
            return {
                "answer": f"Error: {str(e)}",
                "sources": [],
                "question": question,
                "original_question": original_question,
                "search_question": search_question,
                "query_was_optimized": False,
                "error": str(e)
            }

    def _retrieve_documents(self, question: str):
        """Retrieve relevant documents from the retriever."""
        try:
            if hasattr(self.retriever, 'invoke'):
                return self.retriever.invoke(question)
            else:
                return self.retriever.get_relevant_documents(question)
        except Exception as e:
            print(f"Retrieval error: {e}")
            return []

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
            return f"Error generating response: {e}"

    def ask_with_sources(self, question: str, k: int = 3) -> Dict[str, Any]:
        """
        Ask a question and include detailed source information.

        Args:
            question: The question to ask
            k: Maximum number of sources to include

        Returns:
            Dictionary with answer and detailed sources
        """
        result = self.ask(question)

        # Format sources
        if result.get("sources"):
            formatted_sources = []
            for i, doc in enumerate(result["sources"][:k]):
                content = doc.page_content
                if len(content) > 200:
                    content = content[:200] + "..."

                formatted_sources.append({
                    "source_id": i + 1,
                    "content": content,
                    "full_content": doc.page_content,
                    "metadata": doc.metadata
                })

            result["formatted_sources"] = formatted_sources

        return result

    def clear_memory(self):
        """Clear conversation memory."""
        self.conversation_history = []
        print("Conversation memory cleared.")

    def get_memory_contents(self) -> Dict[str, Any]:
        """Get current conversation memory contents."""
        return {"chat_history": self.conversation_history}

    def get_llm_info(self) -> Dict[str, str]:
        """Get information about the LLM being used."""
        return {
            "model_name": self.model_name,
            "llm_type": self._llm_type,
            "llm_provider": self.llm_provider,
            "temperature": self.temperature,
            "conversation_enabled": self.use_conversation,
            "query_optimization_enabled": self.query_optimizer is not None
        }

    def get_query_optimizer_info(self) -> Dict[str, Any]:
        """Get information about the query optimizer."""
        if self.query_optimizer:
            return self.query_optimizer.get_status()
        return {
            "enabled": False,
            "llm_type": "none",
            "model_name": "N/A",
            "temperature": "N/A"
        }


class LocalRAGQA(RAGQA):
    """RAG QA system using local models."""

    def __init__(self, *args, optimize_query: bool = False, **kwargs):
        # Force local model usage
        os.environ["USE_LOCAL_MODEL"] = "true"
        super().__init__(*args, optimize_query=optimize_query, **kwargs)


# Example usage
if __name__ == "__main__":
    import tempfile

    print("Testing RAGQA Module...")
    print("=" * 60)

    # Create sample document
    test_content = """Artificial Intelligence (AI) is the simulation of human intelligence in machines.
Machine Learning (ML) is a subset of AI that enables systems to learn from data.
Deep Learning uses neural networks with many layers to learn complex patterns.
Natural Language Processing (NLP) enables computers to understand human language.
Computer Vision allows machines to interpret and understand visual information.
Robotics combines AI with mechanical engineering to create intelligent machines."""

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        sample_file = f.name

    try:
        from document_loader import DocumentLoader
        from vector_store import VectorStore

        # Load and process documents
        print("\n1. Loading documents...")
        loader = DocumentLoader(chunk_size=400, chunk_overlap=100)
        chunks = loader.load_text(sample_file)

        # Create vector store
        print("\n2. Creating vector store...")
        vector_store = VectorStore(persist_directory="./test_rag_db")
        vector_store.create_from_documents(chunks)

        # Create retriever
        retriever = vector_store.get_retriever(search_kwargs={"k": 3})

        # Create RAG QA system
        print("\n3. Creating RAG QA system...")
        rag_system = RAGQA(
            retriever=retriever,
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            use_conversation=False
        )

        # Print LLM info
        info = rag_system.get_llm_info()
        print(f"\nLLM Info: {info}")

        # Test questions
        test_questions = [
            "What is artificial intelligence?",
            "What is deep learning?",
            "What is NLP?"
        ]

        print("\n4. Testing questions...")
        for question in test_questions:
            print(f"\n{'=' * 60}")
            print(f"Question: {question}")

            result = rag_system.ask_with_sources(question)
            print(f"\nAnswer: {result['answer']}")

            if result.get('formatted_sources'):
                print("\nSources:")
                for source in result['formatted_sources']:
                    print(f"  [{source['source_id']}] {source['content'][:80]}...")

        # Test conversation mode
        print("\n" + "=" * 60)
        print("Testing conversation mode...")

        conv_rag = RAGQA(
            retriever=retriever,
            use_conversation=True
        )

        print("\nQ1: What is AI?")
        r1 = conv_rag.ask("What is AI?")
        print(f"A1: {r1['answer'][:100]}...")

        print("\nQ2: How is it related to machine learning?")
        r2 = conv_rag.ask("How is it related to machine learning?")
        print(f"A2: {r2['answer'][:100]}...")

        # Show memory
        memory = conv_rag.get_memory_contents()
        print(f"\nConversation turns: {len(memory['chat_history'])}")

        print("\n" + "=" * 60)
        print("RAGQA test completed successfully!")

    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        if os.path.exists(sample_file):
            os.unlink(sample_file)
        if os.path.exists("./test_rag_db"):
            import shutil
            shutil.rmtree("./test_rag_db")
            print("\nCleaned up test database")
