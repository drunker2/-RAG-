#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging module for RAG System.
Provides centralized logging with configurable levels and formats.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional
from pathlib import Path


# Color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels."""

    LEVEL_COLORS = {
        logging.DEBUG: Colors.CYAN,
        logging.INFO: Colors.GREEN,
        logging.WARNING: Colors.YELLOW,
        logging.ERROR: Colors.RED,
        logging.CRITICAL: Colors.MAGENTA,
    }

    def format(self, record):
        # Get color for log level
        color = self.LEVEL_COLORS.get(record.levelno, Colors.WHITE)

        # Format timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Format the message
        level_name = record.levelname
        message = record.getMessage()

        # Build formatted string
        formatted = f"{color}[{timestamp}] [{level_name:7}] {message}{Colors.RESET}"

        return formatted


class RAGLogger:
    """
    Centralized logger for RAG System.
    Provides consistent logging across all modules.
    """

    _instance: Optional['RAGLogger'] = None
    _logger: Optional[logging.Logger] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the logger."""
        self._logger = logging.getLogger("RAGSystem")
        self._logger.setLevel(logging.DEBUG)

        # Clear existing handlers
        self._logger.handlers.clear()

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ColoredFormatter())
        self._logger.addHandler(console_handler)

        # File handler (optional)
        log_dir = Path("./logs")
        log_file = log_dir / "rag_system.log"

        try:
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self._logger.addHandler(file_handler)
        except Exception:
            pass  # Skip file logging if not possible

    @classmethod
    def get_logger(cls) -> logging.Logger:
        """Get the logger instance."""
        instance = cls()
        return instance._logger

    @classmethod
    def set_level(cls, level: str) -> None:
        """
        Set logging level.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        instance = cls()
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL,
        }
        log_level = level_map.get(level.upper(), logging.INFO)
        instance._logger.setLevel(log_level)

        for handler in instance._logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(log_level)


# Convenience functions
def get_logger(name: str = "RAGSystem") -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (optional)

    Returns:
        Logger instance
    """
    return RAGLogger.get_logger()


def debug(msg: str) -> None:
    """Log debug message."""
    get_logger().debug(msg)


def info(msg: str) -> None:
    """Log info message."""
    get_logger().info(msg)


def warning(msg: str) -> None:
    """Log warning message."""
    get_logger().warning(msg)


def error(msg: str) -> None:
    """Log error message."""
    get_logger().error(msg)


def critical(msg: str) -> None:
    """Log critical message."""
    get_logger().critical(msg)


# Example usage
if __name__ == "__main__":
    print("Testing Logger Module...")
    print("=" * 60)

    logger = get_logger()

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    print("\nTesting convenience functions...")
    debug("Debug via convenience function")
    info("Info via convenience function")
    warning("Warning via convenience function")
    error("Error via convenience function")

    print("\nLogger test completed!")
