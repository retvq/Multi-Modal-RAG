"""
ContentBlock Schema Definition

This module defines the core data structures for representing extracted document content.
All modalities share the common ContentBlock structure with modality-specific metadata.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Modality(Enum):
    """Document content modality classification."""
    TEXT = "TEXT"
    TABLE = "TABLE"
    FIGURE = "FIGURE"
    BOX = "BOX"
    ANNEX = "ANNEX"
    FOOTNOTE = "FOOTNOTE"


class TextType(Enum):
    """Classification of text block types."""
    BODY = "BODY"           # Regular paragraph text
    HEADING = "HEADING"     # Section heading
    BULLET = "BULLET"       # Bullet list item
    NUMBERED_LIST = "NUMBERED_LIST"  # Numbered list item
    CAPTION = "CAPTION"     # Figure/table caption


@dataclass
class BoundingBox:
    """
    Normalized bounding box coordinates.
    
    Coordinates are normalized to 0-1 range relative to page dimensions.
    This allows consistent positioning regardless of page size or resolution.
    """
    x0: float  # Left edge (0 = left margin)
    y0: float  # Top edge (0 = top margin)
    x1: float  # Right edge (1 = right margin)
    y1: float  # Bottom edge (1 = bottom margin)
    
    def __post_init__(self):
        """Validate bounding box coordinates."""
        if not (0 <= self.x0 <= self.x1 <= 1):
            raise ValueError(f"Invalid x coordinates: x0={self.x0}, x1={self.x1}")
        if not (0 <= self.y0 <= self.y1 <= 1):
            raise ValueError(f"Invalid y coordinates: y0={self.y0}, y1={self.y1}")
    
    @classmethod
    def from_pdf_coords(cls, x0: float, y0: float, x1: float, y1: float, 
                        page_width: float, page_height: float) -> "BoundingBox":
        """
        Create BoundingBox from PDF coordinate system.
        
        PDF coordinates have origin at bottom-left, we normalize to top-left origin.
        """
        return cls(
            x0=x0 / page_width,
            y0=1.0 - (y1 / page_height),  # Flip y-axis
            x1=x1 / page_width,
            y1=1.0 - (y0 / page_height),
        )


@dataclass
class TextMetadata:
    """Modality-specific metadata for TEXT blocks."""
    paragraph_number: Optional[int] = None  # Official para number (e.g., "1.", "2.")
    text_type: TextType = TextType.BODY
    heading_level: Optional[int] = None     # 1-6 for headings
    has_cross_references: bool = False
    font_size: Optional[float] = None       # For section detection heuristics
    is_bold: bool = False                   # For section detection heuristics


@dataclass
class ContentBlock:
    """
    Universal content block representing any extracted document element.
    
    This is the core output unit of the ingestion pipeline. Every piece of
    document content (text paragraph, table, figure, etc.) becomes one ContentBlock.
    """
    
    # === Required Core Fields ===
    block_id: str                           # Unique identifier (deterministic)
    document_id: str                        # Parent document identifier
    modality: Modality                      # Content type classification
    content: str                            # Primary content (text or description)
    page_number: int                        # 1-indexed page number
    
    # === Provenance Fields ===
    section_hierarchy: list[str] = field(default_factory=list)  # Path from root
    section_id: Optional[str] = None        # Section identifier
    position_in_section: int = 0            # Ordinal within section
    bounding_box: Optional[BoundingBox] = None
    
    # === Multi-Page Support ===
    page_range: Optional[tuple[int, int]] = None  # (start, end) if spans pages
    
    # === Modality-Specific Metadata ===
    text_metadata: Optional[TextMetadata] = None
    # table_metadata, figure_metadata, etc. to be added
    
    # === Citation Fields ===
    citation_short: str = ""                # "[Page 12]"
    citation_full: str = ""                 # Full attribution
    citation_section: str = ""              # "[Section, Para N]"
    
    # === Chunking Support ===
    content_length: int = 0
    is_complete: bool = True
    continuation_of: Optional[str] = None   # Block ID if continues another
    continued_by: Optional[str] = None      # Block ID if continued elsewhere
    
    # === Confidence ===
    extraction_confidence: float = 1.0
    extraction_warnings: list[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Compute derived fields after initialization."""
        self.content_length = len(self.content)
        self._generate_citations()
    
    def _generate_citations(self):
        """Generate citation strings from provenance data."""
        # Short citation: just page
        self.citation_short = f"[Page {self.page_number}]"
        
        # Section-based citation
        if self.section_hierarchy:
            section_path = " > ".join(self.section_hierarchy[-2:])  # Last 2 levels
            if self.text_metadata and self.text_metadata.paragraph_number:
                self.citation_section = f"[{section_path}, Para {self.text_metadata.paragraph_number}]"
            else:
                self.citation_section = f"[{section_path}, Page {self.page_number}]"
        else:
            self.citation_section = self.citation_short
        
        # Full citation: modality-aware
        if self.modality == Modality.TEXT:
            self.citation_full = f"Page {self.page_number}"
            if self.section_hierarchy:
                self.citation_full = f"{self.section_hierarchy[-1]}, {self.citation_full}"
        else:
            self.citation_full = f"{self.modality.value}, Page {self.page_number}"
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate that this block meets schema requirements.
        
        Returns:
            (is_valid, list of error messages)
        """
        errors = []
        
        if not self.block_id:
            errors.append("block_id is required")
        if not self.document_id:
            errors.append("document_id is required")
        if self.page_number < 1:
            errors.append(f"Invalid page_number: {self.page_number}")
        if not self.content and self.modality == Modality.TEXT:
            errors.append("TEXT blocks must have content")
        if not self.section_hierarchy:
            errors.append("section_hierarchy must not be empty")
        
        return (len(errors) == 0, errors)
