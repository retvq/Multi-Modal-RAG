"""
Footnote Extractor

Extracts footnotes from PDF pages with marker-to-content linking.

Footnotes in IMF documents typically:
- Use superscript markers in body text (1, 2, 3, etc.)
- Have footnote content at page bottom
- May also use N/ format in tables (1/, 2/, 3/)

This extractor identifies both markers and content, enabling
downstream linking to the referencing blocks.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_path))

from src.ingestion.loaders.pdf_loader import PageInfo
from src.schemas.content_block import BoundingBox


@dataclass
class FootnoteMarker:
    """
    A footnote marker found in body text.
    
    Tracks where in the source text a footnote reference appears.
    """
    marker: str                         # The marker text (e.g., "1", "2/", "*")
    page_number: int
    position_in_text: int               # Character offset in source text
    source_block_position: int          # Position of containing text block
    bbox: Optional[BoundingBox] = None


@dataclass
class RawFootnote:
    """
    Raw extracted footnote before normalization.
    
    Contains the footnote content and marker information.
    """
    # Marker identity
    marker: str                         # e.g., "1", "2", "1/", "*"
    normalized_marker: str = ""         # Normalized form (e.g., "1/" -> "1")
    
    # Content
    content: str = ""                   # Footnote text
    
    # Location
    page_number: int = 0
    position_in_page: int = 0
    bbox: Optional[BoundingBox] = None
    
    # Linking (to be populated by linker)
    referenced_by: List[str] = field(default_factory=list)  # Block IDs
    
    # Quality
    extraction_confidence: float = 1.0
    extraction_warnings: List[str] = field(default_factory=list)


class FootnoteExtractor:
    """
    Extracts footnotes from PDF pages.
    
    Strategy:
    1. Identify footnote region (bottom of page)
    2. Extract footnote content with markers
    3. Identify markers in body text
    4. Match markers to footnote content
    
    IMF-specific patterns:
    - Superscript numbers (1, 2, 3) for text footnotes
    - N/ format (1/, 2/, 3/) for table footnotes
    - Source attributions starting with "Sources:" or "Source:"
    """
    
    def __init__(
        self,
        footer_region_ratio: float = 0.15,  # Bottom 15% is footnote region
        detect_markers: bool = True
    ):
        """
        Initialize footnote extractor.
        
        Args:
            footer_region_ratio: Fraction of page height considered footnote region
            detect_markers: Whether to detect markers in body text
        """
        self.footer_region_ratio = footer_region_ratio
        self.detect_markers = detect_markers
        
        # Regex patterns for footnote detection
        self.footnote_patterns = [
            # Standard: "1 Text of footnote" or "1. Text"
            r'^(\d+)[.\s]+(.+)$',
            # IMF table style: "1/ Text of footnote"
            r'^(\d+/)\s*(.+)$',
            # Star notation: "* Text of footnote"
            r'^(\*+)\s*(.+)$',
        ]
        
        self.marker_patterns = [
            # Superscript numbers in text (captured as standalone)
            r'(\d+)(?=[.\s,;)\]])',
            # Table-style markers
            r'(\d+/)',
        ]
    
    def extract(self, page: PageInfo) -> Tuple[List[RawFootnote], List[FootnoteMarker]]:
        """
        Extract footnotes and markers from a single page.
        
        Args:
            page: PageInfo from document loader
            
        Returns:
            (list of RawFootnotes, list of FootnoteMarkers)
        """
        fitz_page = page.fitz_page
        page_width = page.width
        page_height = page.height
        
        # Step 1: Extract footnote content from bottom region
        footnotes = self._extract_footnote_content(
            page=fitz_page,
            page_number=page.page_number,
            page_width=page_width,
            page_height=page_height
        )
        
        # Step 2: Extract markers from body text (optional)
        markers = []
        if self.detect_markers:
            markers = self._extract_markers(
                page=fitz_page,
                page_number=page.page_number,
                page_width=page_width,
                page_height=page_height
            )
        
        return (footnotes, markers)
    
    def _extract_footnote_content(
        self,
        page,
        page_number: int,
        page_width: float,
        page_height: float
    ) -> List[RawFootnote]:
        """
        Extract footnote content from bottom region of page.
        """
        footnotes = []
        
        # Define footnote region (bottom portion of page)
        footnote_y_start = page_height * (1 - self.footer_region_ratio)
        
        # Get all text blocks
        blocks = page.get_text("dict")["blocks"]
        
        for block in blocks:
            if block.get("type") != 0:  # Text block only
                continue
            
            bbox = block.get("bbox", (0, 0, 0, 0))
            
            # Check if block is in footnote region
            if bbox[1] >= footnote_y_start:
                # Extract text from block
                text_parts = []
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text_parts.append(span.get("text", ""))
                
                text = " ".join(text_parts).strip()
                
                if not text:
                    continue
                
                # Try to parse as footnote
                parsed = self._parse_footnote_text(text)
                
                for marker, content in parsed:
                    norm_bbox = BoundingBox(
                        x0=bbox[0] / page_width,
                        y0=bbox[1] / page_height,
                        x1=bbox[2] / page_width,
                        y1=bbox[3] / page_height
                    )
                    
                    footnotes.append(RawFootnote(
                        marker=marker,
                        normalized_marker=self._normalize_marker(marker),
                        content=content,
                        page_number=page_number,
                        position_in_page=len(footnotes),
                        bbox=norm_bbox,
                        extraction_confidence=1.0
                    ))
        
        return footnotes
    
    def _parse_footnote_text(self, text: str) -> List[Tuple[str, str]]:
        """
        Parse footnote text to extract marker and content.
        
        Returns list of (marker, content) tuples.
        """
        results = []
        
        # Try each pattern
        for pattern in self.footnote_patterns:
            match = re.match(pattern, text.strip())
            if match:
                marker = match.group(1)
                content = match.group(2).strip()
                results.append((marker, content))
                return results
        
        # If no pattern matches, check for multiple footnotes in one block
        # Pattern: "1 first footnote. 2 second footnote."
        multi_pattern = r'(\d+)[.\s]+([^0-9]+?)(?=\d+[.\s]|$)'
        matches = re.findall(multi_pattern, text)
        
        for marker, content in matches:
            results.append((marker.strip(), content.strip()))
        
        if not results and text:
            # Fallback: treat as source attribution or unmarked footnote
            if text.lower().startswith(("source", "note:")):
                results.append(("source", text))
        
        return results
    
    def _normalize_marker(self, marker: str) -> str:
        """
        Normalize marker to standard form for matching.
        
        "1/" -> "1"
        "1" -> "1"
        "*" -> "*"
        """
        # Remove trailing slash
        normalized = marker.rstrip("/")
        return normalized
    
    def _extract_markers(
        self,
        page,
        page_number: int,
        page_width: float,
        page_height: float
    ) -> List[FootnoteMarker]:
        """
        Extract footnote markers from body text.
        
        Looks for superscript or inline markers that reference footnotes.
        """
        markers = []
        
        # Get text with position information
        blocks = page.get_text("dict")["blocks"]
        
        footnote_y_start = page_height * (1 - self.footer_region_ratio)
        
        for block_idx, block in enumerate(blocks):
            if block.get("type") != 0:
                continue
            
            bbox = block.get("bbox", (0, 0, 0, 0))
            
            # Skip footnote region
            if bbox[1] >= footnote_y_start:
                continue
            
            # Check spans for superscript (smaller font size)
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "")
                    size = span.get("size", 12)
                    
                    # Superscript indicators: smaller font, or explicit superscript flag
                    is_superscript = size < 10 or "super" in span.get("font", "").lower()
                    
                    if is_superscript:
                        # Check if it's a number (likely footnote marker)
                        if re.match(r'^\d+$', text.strip()):
                            span_bbox = span.get("bbox", bbox)
                            markers.append(FootnoteMarker(
                                marker=text.strip(),
                                page_number=page_number,
                                position_in_text=0,  # Would need full text to compute
                                source_block_position=block_idx,
                                bbox=BoundingBox(
                                    x0=span_bbox[0] / page_width,
                                    y0=span_bbox[1] / page_height,
                                    x1=span_bbox[2] / page_width,
                                    y1=span_bbox[3] / page_height
                                )
                            ))
        
        return markers


# Footnote normalizer (inline since simple)
def normalize_footnote_to_block(
    raw: RawFootnote,
    document_id: str,
    section_hierarchy: List[str]
) -> 'ContentBlock':
    """
    Convert RawFootnote to ContentBlock.
    """
    from src.ingestion.utils.id_generator import generate_block_id, hash_content
    from src.schemas.content_block import ContentBlock, Modality
    
    content_hash = hash_content(raw.content or raw.marker)
    block_id = generate_block_id(
        document_id=document_id,
        page_number=raw.page_number,
        modality="FOOTNOTE",
        content_hash=content_hash,
        position_in_page=raw.position_in_page
    )
    
    # Content includes marker for clarity
    content = f"[{raw.marker}] {raw.content}"
    
    if not section_hierarchy:
        section_hierarchy = ["Document", "Footnotes"]
    
    block = ContentBlock(
        block_id=block_id,
        document_id=document_id,
        modality=Modality.FOOTNOTE,
        content=content,
        page_number=raw.page_number,
        section_hierarchy=section_hierarchy,
        bounding_box=raw.bbox,
        extraction_confidence=raw.extraction_confidence,
        extraction_warnings=raw.extraction_warnings.copy()
    )
    
    # Override citations
    block.citation_short = f"[Footnote {raw.marker}, Page {raw.page_number}]"
    block.citation_full = f"Footnote {raw.marker}, Page {raw.page_number}: {raw.content[:50]}..."
    
    # Store marker for linking
    block._footnote_marker = raw.marker
    block._normalized_marker = raw.normalized_marker
    
    return block


EXAMPLE_OUTPUT = """
Example RawFootnote:

RawFootnote(
    marker='1/',
    normalized_marker='1',
    content='Crude oil, natural gas, propane, butane, and condensates.',
    page_number=39,
    position_in_page=0,
    bbox=BoundingBox(x0=0.10, y0=0.92, x1=0.90, y1=0.94),
    referenced_by=['qatar_p039_table_xxx_000'],
    extraction_confidence=1.0
)

Example FootnoteMarker:

FootnoteMarker(
    marker='1',
    page_number=5,
    position_in_text=245,
    source_block_position=3,
    bbox=BoundingBox(x0=0.45, y0=0.32, x1=0.46, y1=0.33)
)
"""

if __name__ == "__main__":
    print(EXAMPLE_OUTPUT)
