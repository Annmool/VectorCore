"""
Universal document loader for PDFs, Markdown, HTML, Python/Code, and Plaintext files.
Extracts rich metadata (page numbers, section headers, authors, timestamps).
"""

import os
import re
import html
from typing import List, Dict, Any, Optional
from pypdf import PdfReader


def clean_text(text: str) -> str:
    """Normalize text whitespace, strip control characters, and unescape HTML."""
    if not text:
        return ""
    text = html.unescape(text)
    # Replace non-breaking spaces and tabs
    text = text.replace("\u00a0", " ").replace("\t", " ")
    # Normalize multiple newlines and carriage returns
    text = re.sub(r"\r\n|\r", "\n", text)
    # Collapse multiple spaces while preserving paragraphs
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class Document:
    """Represents an ingested document with content and metadata."""

    def __init__(self, content: str, metadata: Optional[Dict[str, Any]] = None):
        self.content = clean_text(content)
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        source = self.metadata.get("source", "unknown")
        return f"<Document source='{source}' chars={len(self.content)}>"


class DocumentLoader:
    """
    Loads documents across multiple formats into normalized Document instances.
    """

    @classmethod
    def load_pdf(cls, file_path: str) -> List[Document]:
        """Extract text page by page from PDF file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        reader = PdfReader(file_path)
        docs = []
        base_name = os.path.basename(file_path)
        total_pages = len(reader.pages)

        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(
                    Document(
                        content=text,
                        metadata={
                            "source": base_name,
                            "file_path": file_path,
                            "file_type": "pdf",
                            "page": page_idx + 1,
                            "total_pages": total_pages,
                        },
                    )
                )
        return docs

    @classmethod
    def load_markdown(cls, file_path: str) -> List[Document]:
        """Load markdown file with header structure extraction."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Markdown file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        base_name = os.path.basename(file_path)
        # Extract title from first # header if present
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else base_name

        return [
            Document(
                content=content,
                metadata={
                    "source": base_name,
                    "title": title,
                    "file_path": file_path,
                    "file_type": "markdown",
                },
            )
        ]

    @classmethod
    def load_code(cls, file_path: str) -> List[Document]:
        """Load code or plaintext files."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        ext = os.path.splitext(file_path)[1].lstrip(".")
        base_name = os.path.basename(file_path)

        return [
            Document(
                content=content,
                metadata={
                    "source": base_name,
                    "file_path": file_path,
                    "file_type": ext or "text",
                },
            )
        ]

    @classmethod
    def load_file(cls, file_path: str) -> List[Document]:
        """Auto-detect file extension and load documents."""
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            return cls.load_pdf(file_path)
        elif ext in [".md", ".markdown"]:
            return cls.load_markdown(file_path)
        else:
            return cls.load_code(file_path)

    @classmethod
    def load_text_string(
        cls, text: str, source_name: str = "user_input", metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Create a Document directly from raw text string."""
        meta = {"source": source_name, "file_type": "text"}
        if metadata:
            meta.update(metadata)
        return Document(content=text, metadata=meta)
