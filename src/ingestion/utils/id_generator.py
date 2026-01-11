"""
Block ID Generation

Generates deterministic, stable identifiers for content blocks.
The same document content in the same position will always produce the same ID.
This enables:
- Re-ingestion without breaking references
- Deduplication detection
- Stable citations
"""

import hashlib
from typing import Optional


def generate_document_id(file_path: str) -> str:
    """
    Generate a stable document identifier from file path.
    
    For MVP, we use the file name. In production, you might use
    a content hash of the entire document.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Stable document identifier
    """
    # Extract filename without extension
    from pathlib import Path
    filename = Path(file_path).stem
    
    # Sanitize for use as ID (alphanumeric + underscore)
    sanitized = "".join(c if c.isalnum() else "_" for c in filename)
    
    return sanitized.lower()


def generate_block_id(
    document_id: str,
    page_number: int,
    modality: str,
    content_hash: str,
    position_in_page: int = 0
) -> str:
    """
    Generate a deterministic block identifier.
    
    The ID is based on:
    - Document identity
    - Page location
    - Content type
    - Content hash (first 8 chars)
    - Position within page (for ordering disambiguation)
    
    This ensures:
    - Same content → same ID (for dedup)
    - Different positions → different IDs (for uniqueness)
    - Readable structure (for debugging)
    
    Args:
        document_id: Parent document identifier
        page_number: 1-indexed page number
        modality: Content modality (TEXT, TABLE, etc.)
        content_hash: Hash of content (from _hash_content)
        position_in_page: Ordinal position on this page
        
    Returns:
        Stable block identifier like "qatar_p12_text_a1b2c3d4_001"
    """
    # Take first 8 characters of content hash
    short_hash = content_hash[:8]
    
    # Format: {doc}_{page}_{type}_{hash}_{position}
    block_id = f"{document_id}_p{page_number:03d}_{modality.lower()}_{short_hash}_{position_in_page:03d}"
    
    return block_id


def hash_content(content: str) -> str:
    """
    Generate a stable hash of text content.
    
    Used for block ID generation and deduplication.
    Normalizes whitespace before hashing to handle
    extraction variations.
    
    Args:
        content: Text content to hash
        
    Returns:
        Hex digest of content hash
    """
    # Normalize whitespace for consistent hashing
    normalized = " ".join(content.split())
    
    # Use SHA-256 for cryptographic stability
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
