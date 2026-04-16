#!/usr/bin/env python3
"""
Document loader and text splitter module for RAG system.
Supports PDF, TXT, Markdown, and other document formats.
"""

import os
import warnings
from typing import List, Optional
from pathlib import Path

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Try to import PDF loader, provide helpful error if not available
try:
    from langchain_community.document_loaders import PyPDFLoader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: PyPDFLoader not available. PDF support disabled.")
    print("Install with: pip install pypdf")


class DocumentLoader:
    """Load and split documents for RAG system."""

    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 separators: Optional[List[str]] = None):
        """
        Initialize document loader with text splitter.

        Args:
            chunk_size: Size of each text chunk (default: 1000)
            chunk_overlap: Overlap between chunks (default: 200)
            separators: Custom separators for text splitting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Default separators optimized for multiple languages
        if separators is None:
            separators = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators
        )

    def _detect_encoding(self, file_path: str) -> str:
        """
        Try to detect file encoding.

        Args:
            file_path: Path to the file

        Returns:
            Detected encoding or 'utf-8' as default
        """
        encodings = ['utf-8', 'utf-8-sig', 'gbk', 'gb2312', 'gb18030', 'latin-1']

        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1024)  # Try reading first 1KB
                return encoding
            except (UnicodeDecodeError, UnicodeError):
                continue

        return 'utf-8'  # Default fallback

    def _load_text_file(self, file_path: str) -> str:
        """
        Load text from a file with encoding detection.

        Args:
            file_path: Path to the file

        Returns:
            Text content of the file
        """
        encoding = self._detect_encoding(file_path)

        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            # Last resort: try with errors='ignore'
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load PDF document and split into chunks.

        Args:
            file_path: Path to PDF file

        Returns:
            List of Document chunks
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        if not PDF_SUPPORT:
            raise ImportError("PDF support not available. Install with: pip install pypdf")

        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()

            # Add source metadata
            for doc in documents:
                if 'source' not in doc.metadata:
                    doc.metadata['source'] = file_path
                doc.metadata['file_type'] = 'pdf'

            chunks = self.text_splitter.split_documents(documents)

            print(f"Loaded PDF: {file_path}")
            print(f"  Total pages: {len(documents)}")
            print(f"  Total chunks: {len(chunks)}")

            return chunks

        except Exception as e:
            raise RuntimeError(f"Error loading PDF {file_path}: {e}")

    def load_text(self, file_path: str) -> List[Document]:
        """
        Load text document and split into chunks.

        Args:
            file_path: Path to text file

        Returns:
            List of Document chunks
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Text file not found: {file_path}")

        try:
            # Load text with encoding detection
            text = self._load_text_file(file_path)

            if not text.strip():
                print(f"Warning: File {file_path} is empty or contains no readable text")
                return []

            # Create document with metadata
            doc = Document(
                page_content=text,
                metadata={
                    'source': file_path,
                    'file_type': Path(file_path).suffix.lower(),
                    'file_name': Path(file_path).name
                }
            )

            chunks = self.text_splitter.split_documents([doc])

            print(f"Loaded text file: {file_path}")
            print(f"  Total chunks: {len(chunks)}")

            return chunks

        except Exception as e:
            raise RuntimeError(f"Error loading text file {file_path}: {e}")

    def load_markdown(self, file_path: str) -> List[Document]:
        """
        Load Markdown document and split into chunks.

        Args:
            file_path: Path to markdown file

        Returns:
            List of Document chunks
        """
        # Markdown files are text files
        chunks = self.load_text(file_path)

        # Update file type in metadata
        for chunk in chunks:
            chunk.metadata['file_type'] = 'markdown'

        return chunks

    def load_directory(self,
                       directory_path: str,
                       recursive: bool = False,
                       extensions: Optional[List[str]] = None) -> List[Document]:
        """
        Load all supported documents from a directory.

        Args:
            directory_path: Path to directory containing documents
            recursive: Whether to search subdirectories
            extensions: List of file extensions to include

        Returns:
            List of Document chunks from all files
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        if extensions is None:
            extensions = ['.pdf', '.txt', '.md']

        all_chunks = []
        processed_files = 0
        failed_files = 0

        directory = Path(directory_path)

        # Get all files
        if recursive:
            files = list(directory.rglob('*'))
        else:
            files = list(directory.glob('*'))

        for file_path in files:
            if not file_path.is_file():
                continue

            ext = file_path.suffix.lower()
            if ext not in extensions:
                continue

            try:
                if ext == '.pdf':
                    if PDF_SUPPORT:
                        chunks = self.load_pdf(str(file_path))
                        all_chunks.extend(chunks)
                        processed_files += 1
                    else:
                        print(f"Skipping {file_path.name}: PDF support not available")
                        failed_files += 1
                elif ext in ['.txt', '.md']:
                    chunks = self.load_text(str(file_path))
                    all_chunks.extend(chunks)
                    processed_files += 1
            except Exception as e:
                print(f"Error loading {file_path.name}: {e}")
                failed_files += 1
                continue

        print(f"\nDirectory loading summary:")
        print(f"  Directory: {directory_path}")
        print(f"  Files processed: {processed_files}")
        print(f"  Files failed: {failed_files}")
        print(f"  Total chunks: {len(all_chunks)}")

        return all_chunks

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a single file based on its extension.

        Args:
            file_path: Path to the file

        Returns:
            List of Document chunks
        """
        ext = Path(file_path).suffix.lower()

        if ext == '.pdf':
            return self.load_pdf(file_path)
        elif ext in ['.txt', '.md']:
            return self.load_text(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def load_string(self,
                    text: str,
                    metadata: Optional[dict] = None) -> List[Document]:
        """
        Load text from a string and split into chunks.

        Args:
            text: Text content to split
            metadata: Optional metadata to attach to documents

        Returns:
            List of Document chunks
        """
        if not text.strip():
            return []

        if metadata is None:
            metadata = {}

        doc = Document(page_content=text, metadata=metadata)
        chunks = self.text_splitter.split_documents([doc])

        print(f"Split text into {len(chunks)} chunks")
        return chunks


# Example usage
if __name__ == "__main__":
    import tempfile

    print("Testing DocumentLoader...")
    print("-" * 60)

    # Create a test file with mixed content
    test_content = """# RAG System Overview

Retrieval-Augmented Generation (RAG) is a powerful technique that combines
information retrieval with language generation.

## Key Benefits

1. Reduces hallucination in AI responses
2. Provides factual, verifiable answers
3. Can be updated with new information without retraining

## How It Works

The RAG pipeline consists of several steps:
- Document loading and chunking
- Vector embedding creation
- Semantic search for relevant chunks
- Answer generation using retrieved context

This approach is particularly useful for domain-specific applications
where accuracy and source attribution are important.
"""

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(test_content)
        temp_file = f.name

    try:
        # Test loading
        loader = DocumentLoader(chunk_size=300, chunk_overlap=50)
        chunks = loader.load_text(temp_file)

        print(f"\nTest Results:")
        print(f"  Created {len(chunks)} chunks")

        print(f"\nChunk details:")
        for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
            print(f"\n  Chunk {i + 1}:")
            print(f"    Length: {len(chunk.page_content)} chars")
            print(f"    Preview: {chunk.page_content[:80]}...")
            print(f"    Metadata: {chunk.metadata}")

        # Test string loading
        print(f"\nTesting string loading...")
        str_chunks = loader.load_string(test_content, metadata={'source': 'test'})
        print(f"  Created {len(str_chunks)} chunks from string")

        print(f"\nDocumentLoader test completed successfully!")

    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.unlink(temp_file)
