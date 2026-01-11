"""
PDF Document Loader

Handles loading PDF documents and providing page-level access.
Uses PyMuPDF (fitz) as the primary PDF library.

This is the entry point for the ingestion pipeline.
"""

import fitz  # PyMuPDF
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


class DocumentNotFoundError(Exception):
    """Raised when document file cannot be found."""
    pass


class DocumentParseError(Exception):
    """Raised when document cannot be parsed."""
    pass


class DocumentAccessError(Exception):
    """Raised when document is encrypted or otherwise inaccessible."""
    pass


@dataclass
class PageInfo:
    """
    Information about a single PDF page.
    
    Provides:
    - Page number (1-indexed for user-facing, 0-indexed for internal)
    - Page dimensions
    - Access to PyMuPDF page object for extraction
    """
    page_number: int        # 1-indexed page number
    width: float            # Page width in points
    height: float           # Page height in points
    _fitz_page: fitz.Page   # Internal PyMuPDF page object
    
    @property
    def fitz_page(self) -> fitz.Page:
        """Access to underlying PyMuPDF page for extraction."""
        return self._fitz_page


@dataclass
class LoadedDocument:
    """
    A loaded PDF document ready for extraction.
    
    Provides:
    - Document metadata
    - Page iteration
    - Resource cleanup
    
    Usage:
        with PDFLoader.load("document.pdf") as doc:
            for page in doc.pages():
                # Process page
    """
    document_id: str
    source_path: str
    total_pages: int
    _fitz_doc: fitz.Document
    
    def pages(self) -> Iterator[PageInfo]:
        """
        Iterate over all pages in the document.
        
        Yields PageInfo objects for each page in order.
        Page numbers are 1-indexed.
        """
        for page_idx in range(self.total_pages):
            fitz_page = self._fitz_doc[page_idx]
            rect = fitz_page.rect
            
            yield PageInfo(
                page_number=page_idx + 1,  # 1-indexed
                width=rect.width,
                height=rect.height,
                _fitz_page=fitz_page
            )
    
    def get_page(self, page_number: int) -> Optional[PageInfo]:
        """
        Get a specific page by 1-indexed page number.
        
        Args:
            page_number: 1-indexed page number
            
        Returns:
            PageInfo or None if page doesn't exist
        """
        if page_number < 1 or page_number > self.total_pages:
            return None
        
        page_idx = page_number - 1
        fitz_page = self._fitz_doc[page_idx]
        rect = fitz_page.rect
        
        return PageInfo(
            page_number=page_number,
            width=rect.width,
            height=rect.height,
            _fitz_page=fitz_page
        )
    
    def close(self):
        """Release document resources."""
        if self._fitz_doc:
            self._fitz_doc.close()


class PDFLoader:
    """
    PDF document loader using PyMuPDF.
    
    Handles:
    - File existence validation
    - Document parsing
    - Encryption detection
    - Resource management
    
    Usage:
        loader = PDFLoader()
        doc = loader.load("document.pdf")
        # or as context manager:
        with loader.load("document.pdf") as doc:
            ...
    """
    
    @staticmethod
    def load(file_path: str, document_id: Optional[str] = None) -> LoadedDocument:
        """
        Load a PDF document.
        
        Args:
            file_path: Path to PDF file
            document_id: Optional document ID (auto-generated if not provided)
            
        Returns:
            LoadedDocument ready for extraction
            
        Raises:
            DocumentNotFoundError: If file doesn't exist
            DocumentParseError: If file is not a valid PDF
            DocumentAccessError: If file is encrypted
        """
        path = Path(file_path)
        
        # Validate file exists
        if not path.exists():
            raise DocumentNotFoundError(f"File not found: {file_path}")
        
        if not path.is_file():
            raise DocumentNotFoundError(f"Not a file: {file_path}")
        
        # Generate document ID if not provided
        if document_id is None:
            # Import here to avoid circular imports
            from src.ingestion.utils.id_generator import generate_document_id
            document_id = generate_document_id(file_path)
        
        # Attempt to open document
        try:
            fitz_doc = fitz.open(file_path)
        except Exception as e:
            raise DocumentParseError(f"Failed to parse PDF: {e}")
        
        # Check for encryption
        if fitz_doc.is_encrypted:
            fitz_doc.close()
            raise DocumentAccessError(f"Document is encrypted: {file_path}")
        
        # Validate it's actually a PDF with pages
        if fitz_doc.page_count == 0:
            fitz_doc.close()
            raise DocumentParseError(f"PDF has no pages: {file_path}")
        
        return LoadedDocument(
            document_id=document_id,
            source_path=str(path.absolute()),
            total_pages=fitz_doc.page_count,
            _fitz_doc=fitz_doc
        )
