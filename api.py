#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI wrapper for RAG System.
Provides REST API endpoints for production deployment.
"""

import os
import sys
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# Import RAG system
from main import RAGSystem
from exceptions import RAGException, handle_exception

# Optional imports
try:
    from cache import get_all_cache_stats, clear_all_caches
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

try:
    from metrics import metrics, health_checker, HealthStatus
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

try:
    from retry import llm_circuit_breaker, api_rate_limiter
    RETRY_AVAILABLE = True
except ImportError:
    RETRY_AVAILABLE = False


# ============================================
# Request/Response Models
# ============================================

class QueryRequest(BaseModel):
    """Query request model."""
    question: str = Field(..., min_length=1, max_length=2000, description="Question to ask")
    show_sources: bool = Field(True, description="Include source documents in response")
    max_sources: int = Field(3, ge=1, le=10, description="Maximum sources to return")


class QueryResponse(BaseModel):
    """Query response model."""
    answer: str
    sources: List[Dict[str, Any]] = []
    original_question: str
    search_question: Optional[str] = None
    query_was_optimized: bool = False
    llm_type: str = "unknown"


class IndexRequest(BaseModel):
    """Index request model."""
    source_path: str = Field(..., description="Path to file or directory to index")
    collection_name: str = Field("rag_collection", description="Collection name")


class IndexResponse(BaseModel):
    """Index response model."""
    success: bool
    message: str
    documents_indexed: int = 0


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    checks: List[Dict[str, Any]] = []


class CacheStatsResponse(BaseModel):
    """Cache statistics response model."""
    caches: Dict[str, Dict[str, Any]]


class ErrorResponse(BaseModel):
    """Error response model."""
    error: bool
    error_code: str
    message: str
    context: Dict[str, Any] = {}


# ============================================
# Global RAG Instance
# ============================================

rag_instance: Optional[RAGSystem] = None


def get_rag() -> RAGSystem:
    """Get or create RAG instance."""
    global rag_instance
    if rag_instance is None:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_instance


# ============================================
# Lifespan Context Manager
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup RAG system."""
    global rag_instance

    # Startup
    print("Initializing RAG System...")

    db_path = os.getenv("RAG_DB_PATH", "./chroma_db")
    collection = os.getenv("RAG_COLLECTION", "rag_collection")

    rag_instance = RAGSystem(vector_db_path=db_path)

    # Try to load existing index
    if os.path.exists(db_path):
        rag_instance.load_existing_index(collection)
        print(f"Loaded existing vector store from {db_path}")

    yield

    # Shutdown
    print("Shutting down RAG System...")
    rag_instance = None


# ============================================
# FastAPI Application
# ============================================

app = FastAPI(
    title="RAG System API",
    description="Retrieval-Augmented Generation API for question answering",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Exception Handlers
# ============================================

@app.exception_handler(RAGException)
async def rag_exception_handler(request, exc: RAGException):
    """Handle RAG exceptions."""
    return JSONResponse(
        status_code=exc.http_status,
        content=exc.to_dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    error_response = handle_exception(exc)
    return JSONResponse(
        status_code=500,
        content=error_response
    )


# ============================================
# API Endpoints
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "RAG System API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Get system health status."""
    if METRICS_AVAILABLE:
        result = health_checker.check()
        return HealthResponse(**result)
    else:
        return HealthResponse(
            status="healthy",
            timestamp="",
            checks=[]
        )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(
    request: QueryRequest,
    rag: RAGSystem = Depends(get_rag)
):
    """
    Ask a question and get an answer.

    Requires the RAG system to be initialized with indexed documents.
    """
    # Check rate limit
    if RETRY_AVAILABLE and not api_rate_limiter.acquire():
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please try again later."
        )

    # Check circuit breaker
    if RETRY_AVAILABLE and not llm_circuit_breaker.can_execute():
        raise HTTPException(
            status_code=503,
            detail="Service temporarily unavailable due to repeated failures."
        )

    try:
        result = rag.ask(
            question=request.question,
            show_sources=request.show_sources,
            max_sources=request.max_sources
        )

        if result is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to process question"
            )

        # Record success
        if RETRY_AVAILABLE:
            llm_circuit_breaker.record_success()

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=result.get("formatted_sources", []),
            original_question=result.get("original_question", request.question),
            search_question=result.get("search_question"),
            query_was_optimized=result.get("query_was_optimized", False),
            llm_type=result.get("llm_type", "unknown")
        )

    except Exception as e:
        # Record failure
        if RETRY_AVAILABLE:
            llm_circuit_breaker.record_failure()
        raise


@app.post("/index", response_model=IndexResponse, tags=["Index"])
async def index_documents(
    request: IndexRequest,
    background_tasks: BackgroundTasks
):
    """
    Index documents from a file or directory.

    This is an asynchronous operation that runs in the background.
    """
    global rag_instance

    if rag_instance is None:
        rag_instance = RAGSystem(vector_db_path=os.getenv("RAG_DB_PATH", "./chroma_db"))
        rag_instance.setup()

    # Validate path exists
    if not os.path.exists(request.source_path):
        raise HTTPException(
            status_code=400,
            detail=f"Source path not found: {request.source_path}"
        )

    # Run indexing in background
    indexing_result = {"status": "pending", "documents": 0}

    def run_indexing():
        try:
            success = rag_instance.index_documents(
                request.source_path,
                request.collection_name
            )
            if success and rag_instance.vector_store:
                info = rag_instance.vector_store.get_collection_info()
                indexing_result["documents"] = info.get("document_count", 0)
            indexing_result["status"] = "completed" if success else "failed"
        except Exception as e:
            indexing_result["status"] = f"error: {str(e)}"

    background_tasks.add_task(run_indexing)

    return IndexResponse(
        success=True,
        message="Indexing started in background",
        documents_indexed=0
    )


@app.post("/setup", tags=["Setup"])
async def setup_qa_system(
    model: str = Query("ERNIE-3.5-8K", description="LLM model name"),
    temperature: float = Query(0.7, ge=0, le=2, description="Model temperature"),
    use_conversation: bool = Query(False, description="Enable conversation memory"),
    optimize_query: bool = Query(False, description="Enable query optimization"),
    use_hybrid: bool = Query(False, description="Enable hybrid retrieval"),
    hybrid_alpha: float = Query(0.5, ge=0, le=1, description="Hybrid retrieval weight"),
    provider: str = Query("auto", description="LLM provider (auto/openai/qianfan/local/demo)"),
    rag: RAGSystem = Depends(get_rag)
):
    """Setup or reconfigure the QA system."""
    success = rag.create_qa_system(
        model_name=model,
        temperature=temperature,
        use_conversation=use_conversation,
        optimize_query=optimize_query,
        use_hybrid=use_hybrid,
        hybrid_alpha=hybrid_alpha,
        llm_provider=provider
    )

    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to setup QA system"
        )

    return {
        "success": True,
        "message": "QA system configured",
        "config": {
            "model": model,
            "provider": provider,
            "temperature": temperature,
            "conversation": use_conversation,
            "optimize_query": optimize_query,
            "hybrid": use_hybrid
        }
    }


@app.get("/stats", tags=["Stats"])
async def get_statistics(rag: RAGSystem = Depends(get_rag)):
    """Get system statistics."""
    stats = {
        "vector_store": {},
        "cache": {},
        "metrics": {}
    }

    # Vector store stats
    if rag.vector_store:
        try:
            stats["vector_store"] = rag.vector_store.get_collection_info()
        except Exception:
            pass

    # Cache stats
    if CACHE_AVAILABLE:
        stats["cache"] = get_all_cache_stats()

    # Metrics
    if METRICS_AVAILABLE:
        stats["metrics"] = metrics.get_all_metrics()

    return stats


@app.get("/cache", response_model=CacheStatsResponse, tags=["Cache"])
async def get_cache_stats():
    """Get cache statistics."""
    if not CACHE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Cache module not available"
        )

    return CacheStatsResponse(caches=get_all_cache_stats())


@app.delete("/cache", tags=["Cache"])
async def clear_cache():
    """Clear all caches."""
    if not CACHE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Cache module not available"
        )

    clear_all_caches()
    return {"success": True, "message": "All caches cleared"}


@app.get("/circuit-breaker", tags=["Circuit Breaker"])
async def get_circuit_breaker_status():
    """Get circuit breaker status."""
    if not RETRY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Retry module not available"
        )

    return llm_circuit_breaker.get_stats()


@app.post("/circuit-breaker/reset", tags=["Circuit Breaker"])
async def reset_circuit_breaker():
    """Reset the circuit breaker."""
    if not RETRY_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Retry module not available"
        )

    llm_circuit_breaker.reset()
    return {"success": True, "message": "Circuit breaker reset"}


@app.get("/info", tags=["Info"])
async def get_system_info(rag: RAGSystem = Depends(get_rag)):
    """Get detailed system information."""
    info = {
        "qa_system": {},
        "vector_store": {},
        "features": {
            "cache": CACHE_AVAILABLE,
            "metrics": METRICS_AVAILABLE,
            "retry": RETRY_AVAILABLE
        }
    }

    if rag.qa_system:
        info["qa_system"] = rag.qa_system.get_llm_info()
        info["query_optimizer"] = rag.qa_system.get_query_optimizer_info()

    if rag.vector_store:
        info["vector_store"] = rag.vector_store.get_collection_info()

    return info


# ============================================
# Batch Operations
# ============================================

class BatchQueryRequest(BaseModel):
    """Batch query request model."""
    questions: List[str] = Field(..., min_items=1, max_items=20)


@app.post("/batch-query", tags=["Query"])
async def batch_query(
    request: BatchQueryRequest,
    rag: RAGSystem = Depends(get_rag)
):
    """
    Process multiple questions in batch.

    Maximum 20 questions per batch.
    """
    results = []

    for question in request.questions:
        result = rag.ask(question, show_sources=True)
        if result:
            results.append({
                "question": question,
                "answer": result.get("answer", ""),
                "success": True
            })
        else:
            results.append({
                "question": question,
                "answer": None,
                "success": False
            })

    return {
        "total": len(results),
        "successful": sum(1 for r in results if r["success"]),
        "results": results
    }


# ============================================
# Main Entry Point
# ============================================

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    print("Starting RAG System API Server...")
    print("=" * 60)
    print("Endpoints:")
    print("  - GET  /              : API info")
    print("  - GET  /health        : Health check")
    print("  - POST /query         : Ask a question")
    print("  - POST /index         : Index documents")
    print("  - POST /setup         : Setup QA system")
    print("  - GET  /stats         : Get statistics")
    print("  - GET  /cache         : Cache stats")
    print("  - DELETE /cache       : Clear cache")
    print("  - GET  /circuit-breaker : Circuit breaker status")
    print("  - POST /batch-query   : Batch queries")
    print("=" * 60)
    print(f"\nAPI Documentation: http://localhost:8000/docs")
    print(f"ReDoc Documentation: http://localhost:8000/redoc")
    print()

    run_server()
