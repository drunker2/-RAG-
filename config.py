#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management module for RAG System.
Provides centralized configuration with environment variables and .env file support.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """
    Centralized configuration management for RAG System.
    Supports environment variables and .env file.
    """

    # Default configuration values
    DEFAULTS = {
        # Vector Store
        "VECTOR_DB_PATH": "./chroma_db",
        "EMBEDDING_MODEL": "all-MiniLM-L6-v2",

        # Vector Store Backend: "chroma" or "postgres"
        "VECTOR_STORE_BACKEND": "postgres",

        # PostgreSQL Settings
        "PG_HOST": "localhost",
        "PG_PORT": "5432",
        "PG_USER": "postgres",
        "PG_PASSWORD": "123456",
        "PG_DATABASE": "postgres",

        # LLM
        "LLM_MODEL": "gpt-3.5-turbo",
        "LLM_TEMPERATURE": "0.7",

        # Retrieval
        "SEARCH_K": "4",
        "CHUNK_SIZE": "1000",
        "CHUNK_OVERLAP": "200",

        # Hybrid Retrieval
        "HYBRID_ALPHA": "0.5",
        "BM25_K1": "1.5",
        "BM25_B": "0.75",

        # Query Optimization
        "QUERY_OPTIMIZER_TEMPERATURE": "0.3",

        # Hugging Face
        "HF_ENDPOINT": "https://hf-mirror.com",

        # System
        "LOG_LEVEL": "INFO",
        "MAX_HISTORY_TURNS": "3",
    }

    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            env_file: Optional path to .env file
        """
        if env_file:
            load_dotenv(env_file)

        # Set HF_ENDPOINT if not already set
        if "HF_ENDPOINT" not in os.environ:
            os.environ["HF_ENDPOINT"] = self.DEFAULTS["HF_ENDPOINT"]

    @classmethod
    def get(cls, key: str, default: Optional[str] = None) -> str:
        """
        Get configuration value.

        Args:
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value
        """
        return os.getenv(key, default or cls.DEFAULTS.get(key, ""))

    @classmethod
    def get_int(cls, key: str, default: Optional[int] = None) -> int:
        """Get configuration value as integer."""
        value = cls.get(key)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default if default is not None else int(cls.DEFAULTS.get(key, "0"))

    @classmethod
    def get_float(cls, key: str, default: Optional[float] = None) -> float:
        """Get configuration value as float."""
        value = cls.get(key)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default if default is not None else float(cls.DEFAULTS.get(key, "0.0"))

    @classmethod
    def get_bool(cls, key: str, default: Optional[bool] = None) -> bool:
        """Get configuration value as boolean."""
        value = cls.get(key).lower()
        if value in ('true', '1', 'yes', 'on'):
            return True
        elif value in ('false', '0', 'no', 'off', ''):
            return False
        return default if default is not None else False

    @classmethod
    def set(cls, key: str, value: str) -> None:
        """
        Set configuration value.

        Args:
            key: Configuration key
            value: Configuration value
        """
        os.environ[key] = str(value)

    @classmethod
    def get_all(cls) -> Dict[str, str]:
        """Get all configuration values (non-sensitive)."""
        # Keys that should be hidden
        sensitive_keys = {'OPENAI_API_KEY', 'API_KEY', 'SECRET', 'PASSWORD'}

        config = {}
        for key in cls.DEFAULTS:
            if key in sensitive_keys:
                config[key] = "***"
            else:
                config[key] = cls.get(key)

        return config

    @classmethod
    def print_config(cls) -> None:
        """Print current configuration."""
        print("\n" + "=" * 60)
        print("  RAG System Configuration")
        print("=" * 60)

        for key, default in cls.DEFAULTS.items():
            value = cls.get(key)
            # Mask sensitive values
            if 'KEY' in key or 'SECRET' in key or 'PASSWORD' in key:
                value = "***" if value else "(not set)"
            print(f"  {key}: {value}")

        print("=" * 60)


# Vector Store configuration helpers
def get_vector_db_path() -> str:
    """Get vector database path."""
    return Config.get("VECTOR_DB_PATH", "./chroma_db")


def get_embedding_model() -> str:
    """Get embedding model name."""
    return Config.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")


# LLM configuration helpers
def get_llm_model() -> str:
    """Get LLM model name."""
    return Config.get("LLM_MODEL", "gpt-3.5-turbo")


def get_llm_temperature() -> float:
    """Get LLM temperature."""
    return Config.get_float("LLM_TEMPERATURE", 0.7)


def has_openai_key() -> bool:
    """Check if OpenAI API key is configured."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    return bool(api_key and api_key.startswith("sk-"))


# Retrieval configuration helpers
def get_search_k() -> int:
    """Get default number of documents to retrieve."""
    return Config.get_int("SEARCH_K", 4)


def get_chunk_size() -> int:
    """Get document chunk size."""
    return Config.get_int("CHUNK_SIZE", 1000)


def get_chunk_overlap() -> int:
    """Get chunk overlap."""
    return Config.get_int("CHUNK_OVERLAP", 200)


def get_hybrid_alpha() -> float:
    """Get hybrid retrieval alpha parameter."""
    return Config.get_float("HYBRID_ALPHA", 0.5)


def get_bm25_params() -> Dict[str, float]:
    """Get BM25 parameters."""
    return {
        "k1": Config.get_float("BM25_K1", 1.5),
        "b": Config.get_float("BM25_B", 0.75)
    }


def get_max_history_turns() -> int:
    """Get maximum conversation history turns."""
    return Config.get_int("MAX_HISTORY_TURNS", 3)


# PostgreSQL configuration helpers
def get_pg_connection_string() -> str:
    """Get PostgreSQL connection string."""
    host = Config.get("PG_HOST", "localhost")
    port = Config.get("PG_PORT", "5432")
    user = Config.get("PG_USER", "postgres")
    password = Config.get("PG_PASSWORD", "123456")
    database = Config.get("PG_DATABASE", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


def get_vector_store_backend() -> str:
    """Get vector store backend: 'chroma' or 'postgres'."""
    return Config.get("VECTOR_STORE_BACKEND", "postgres")


# Example usage
if __name__ == "__main__":
    print("Testing Configuration Module...")
    print("=" * 60)

    # Test getters
    print(f"Vector DB Path: {get_vector_db_path()}")
    print(f"Embedding Model: {get_embedding_model()}")
    print(f"LLM Model: {get_llm_model()}")
    print(f"LLM Temperature: {get_llm_temperature()}")
    print(f"Search K: {get_search_k()}")
    print(f"Chunk Size: {get_chunk_size()}")
    print(f"Chunk Overlap: {get_chunk_overlap()}")
    print(f"Hybrid Alpha: {get_hybrid_alpha()}")
    print(f"BM25 Params: {get_bm25_params()}")
    print(f"Has OpenAI Key: {has_openai_key()}")

    # Print all config
    Config.print_config()
