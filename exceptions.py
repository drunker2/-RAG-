#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Custom exceptions for RAG System.
Provides structured error handling with error codes and context.
"""

from typing import Optional, Dict, Any


class RAGException(Exception):
    """Base exception for RAG System."""

    error_code: str = "RAG_ERROR"
    http_status: int = 500

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code or self.error_code
        self.context = context or {}
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": True,
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context
        }


# ============================================
# Configuration Errors
# ============================================

class ConfigurationError(RAGException):
    """Configuration related errors."""
    error_code = "CONFIG_ERROR"


class MissingAPIKeyError(ConfigurationError):
    """API key not configured."""
    error_code = "MISSING_API_KEY"

    def __init__(self, key_name: str = "API_KEY"):
        super().__init__(
            message=f"Required API key '{key_name}' not found in environment variables",
            context={"key_name": key_name}
        )


# ============================================
# Document Errors
# ============================================

class DocumentError(RAGException):
    """Document processing errors."""
    error_code = "DOCUMENT_ERROR"


class FileNotFoundError(DocumentError):
    """File not found."""
    error_code = "FILE_NOT_FOUND"

    def __init__(self, file_path: str):
        super().__init__(
            message=f"File not found: {file_path}",
            context={"file_path": file_path}
        )


class UnsupportedFileTypeError(DocumentError):
    """Unsupported file type."""
    error_code = "UNSUPPORTED_FILE_TYPE"

    def __init__(self, file_type: str, supported_types: list):
        super().__init__(
            message=f"Unsupported file type: {file_type}. Supported: {supported_types}",
            context={"file_type": file_type, "supported_types": supported_types}
        )


class DocumentLoadingError(DocumentError):
    """Error loading document."""
    error_code = "DOCUMENT_LOADING_ERROR"

    def __init__(self, file_path: str, reason: str):
        super().__init__(
            message=f"Failed to load document: {reason}",
            context={"file_path": file_path, "reason": reason}
        )


# ============================================
# Vector Store Errors
# ============================================

class VectorStoreError(RAGException):
    """Vector store related errors."""
    error_code = "VECTOR_STORE_ERROR"


class EmbeddingError(VectorStoreError):
    """Error creating embeddings."""
    error_code = "EMBEDDING_ERROR"

    def __init__(self, reason: str, model: Optional[str] = None):
        super().__init__(
            message=f"Failed to create embeddings: {reason}",
            context={"reason": reason, "model": model}
        )


class CollectionNotFoundError(VectorStoreError):
    """Collection not found in vector store."""
    error_code = "COLLECTION_NOT_FOUND"

    def __init__(self, collection_name: str):
        super().__init__(
            message=f"Collection '{collection_name}' not found",
            context={"collection_name": collection_name}
        )


class VectorStoreInitError(VectorStoreError):
    """Error initializing vector store."""
    error_code = "VECTOR_STORE_INIT_ERROR"

    def __init__(self, reason: str):
        super().__init__(
            message=f"Failed to initialize vector store: {reason}",
            context={"reason": reason}
        )


# ============================================
# LLM Errors
# ============================================

class LLMError(RAGException):
    """LLM related errors."""
    error_code = "LLM_ERROR"


class LLMNotAvailableError(LLMError):
    """No LLM available."""
    error_code = "LLM_NOT_AVAILABLE"

    def __init__(self):
        super().__init__(
            message="No LLM provider available. Configure OPENAI_API_KEY or enable local model."
        )


class LLMResponseError(LLMError):
    """Error from LLM response."""
    error_code = "LLM_RESPONSE_ERROR"

    def __init__(self, reason: str):
        super().__init__(
            message=f"LLM response error: {reason}",
            context={"reason": reason}
        )


class RateLimitError(LLMError):
    """Rate limit exceeded."""
    error_code = "RATE_LIMIT_EXCEEDED"
    http_status = 429

    def __init__(self, retry_after: Optional[int] = None):
        super().__init__(
            message="Rate limit exceeded. Please wait before retrying.",
            context={"retry_after": retry_after}
        )


# ============================================
# Retrieval Errors
# ============================================

class RetrievalError(RAGException):
    """Retrieval related errors."""
    error_code = "RETRIEVAL_ERROR"


class NoDocumentsRetrievedError(RetrievalError):
    """No documents retrieved."""
    error_code = "NO_DOCUMENTS_RETRIEVED"

    def __init__(self, query: str):
        super().__init__(
            message="No relevant documents found for the query",
            context={"query": query[:100]}  # Truncate for logging
        )


# ============================================
# Validation Errors
# ============================================

class ValidationError(RAGException):
    """Validation errors."""
    error_code = "VALIDATION_ERROR"
    http_status = 400


class EmptyQueryError(ValidationError):
    """Empty query provided."""
    error_code = "EMPTY_QUERY"

    def __init__(self):
        super().__init__(message="Query cannot be empty")


class InvalidParameterError(ValidationError):
    """Invalid parameter value."""
    error_code = "INVALID_PARAMETER"

    def __init__(self, param_name: str, value: Any, expected: str):
        super().__init__(
            message=f"Invalid parameter '{param_name}': expected {expected}, got {type(value).__name__}",
            context={"param_name": param_name, "value": str(value)[:100], "expected": expected}
        )


# ============================================
# Helper function
# ============================================

def handle_exception(e: Exception) -> Dict[str, Any]:
    """
    Convert any exception to a standardized error response.

    Args:
        e: Exception to handle

    Returns:
        Dictionary with error information
    """
    if isinstance(e, RAGException):
        return e.to_dict()

    # Wrap unknown exceptions
    return {
        "error": True,
        "error_code": "INTERNAL_ERROR",
        "message": str(e),
        "context": {"exception_type": type(e).__name__}
    }
