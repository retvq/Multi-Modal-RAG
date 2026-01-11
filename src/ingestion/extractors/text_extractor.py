"""
Text Extractor

Extracts text content from PDF pages with layout awareness.
Produces RawTextBlocks that preserve:
- Reading order
- Paragraph boundaries
- Font information (for heading detection)
- Bounding boxes

IMPORTANT: This module only extracts text. It does NOT:
- Chunk text
- Summarize text
- Embed text
- Detect tables or figures (those have separate extractors)
"""

import fitz  # PyMuPDF
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_path))

from src.ingestion.loaders.pdf_loader import PageInfo
from src.ingestion.utils.text_utils import (
    normalize_whitespace,
    clean_extracted_text,
    is_header_or_footer,
    detect_heading_level,
    detect_paragraph_number,
)
from src.ingestion.utils.bbox import normalize_bbox
from src.schemas.content_block import BoundingBox, TextType


@dataclass
class RawTextBlock:
    """
    Raw extracted text block before normalization.
    
    This is the intermediate representation between extraction
    and ContentBlock creation. It preserves extraction-level
    details that help with normalization.
    """
    text: str                           # Raw extracted text
    page_number: int                    # 1-indexed page number
    bbox: BoundingBox                   # Normalized bounding box
    
    # Font information for heading detection
    font_size: float = 12.0
    is_bold: bool = False
    font_name: str = ""
    
    # Classification hints
    text_type: TextType = TextType.BODY
    heading_level: Optional[int] = None
    paragraph_number: Optional[int] = None
    
    # Position for ordering
    position_in_page: int = 0
    
    # Quality flags
    is_header_footer: bool = False
    extraction_confidence: float = 1.0


class TextExtractor:
    """
    Extracts text from PDF pages with layout awareness.
    
    Strategy:
    1. Extract text blocks using PyMuPDF's "dict" extraction
       (preserves layout and font information)
    2. Group related spans into paragraphs
    3. Detect headers/footers and mark for exclusion
    4. Compute bounding boxes
    5. Detect heading levels and paragraph numbers
    
    This extractor is designed for IMF-style documents:
    - Multi-column layouts
    - Page headers/footers
    - Numbered paragraphs
    - Hierarchical sections
    """
    
    def __init__(self, exclude_headers_footers: bool = True):
        """
        Initialize text extractor.
        
        Args:
            exclude_headers_footers: If True, mark headers/footers but
                don't exclude them (preserves info for page number detection)
        """
        self.exclude_headers_footers = exclude_headers_footers
    
    def extract(self, page: PageInfo) -> List[RawTextBlock]:
        """
        Extract text blocks from a single page.
        
        Args:
            page: PageInfo from document loader
            
        Returns:
            List of RawTextBlocks in reading order
        """
        fitz_page = page.fitz_page
        page_width = page.width
        page_height = page.height
        
        # Extract text with layout using "dict" format
        # This gives us blocks → lines → spans with font info
        page_dict = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        
        raw_blocks = []
        position = 0
        
        for block in page_dict.get("blocks", []):
            # Skip image blocks (handled by figure extractor)
            if block.get("type") == 1:  # Image block
                continue
            
            # Process text block
            if block.get("type") == 0:  # Text block
                raw_block = self._process_text_block(
                    block=block,
                    page_number=page.page_number,
                    page_width=page_width,
                    page_height=page_height,
                    position=position
                )
                
                if raw_block and raw_block.text.strip():
                    raw_blocks.append(raw_block)
                    position += 1
        
        # Post-processing: merge split paragraphs
        merged_blocks = self._merge_split_paragraphs(raw_blocks)
        
        # Post-processing: detect headers/footers
        for block in merged_blocks:
            if is_header_or_footer(block.text, 
                                   block.bbox.y0 * page_height, 
                                   page_height):
                block.is_header_footer = True
        
        # Re-number positions after merging
        for i, block in enumerate(merged_blocks):
            block.position_in_page = i
        
        return merged_blocks
    
    def _process_text_block(
        self,
        block: dict,
        page_number: int,
        page_width: float,
        page_height: float,
        position: int
    ) -> Optional[RawTextBlock]:
        """
        Process a single PyMuPDF text block.
        
        Extracts:
        - Concatenated text from all lines/spans
        - Dominant font information
        - Bounding box
        """
        # Collect all text and font info from lines/spans
        text_parts = []
        font_sizes = []
        font_names = []
        bold_flags = []
        
        for line in block.get("lines", []):
            line_text = ""
            for span in line.get("spans", []):
                span_text = span.get("text", "")
                line_text += span_text
                
                # Collect font info for each span
                font_sizes.append(span.get("size", 12.0))
                font_names.append(span.get("font", ""))
                
                # Detect bold from font name or flags
                font = span.get("font", "").lower()
                flags = span.get("flags", 0)
                is_bold = "bold" in font or (flags & 2 ** 4)  # Bit 4 = bold
                bold_flags.append(is_bold)
            
            text_parts.append(line_text)
        
        # Join lines with space (newlines within a block are usually soft wraps)
        raw_text = " ".join(text_parts)
        
        # Clean the extracted text
        raw_text = clean_extracted_text(raw_text)
        raw_text = normalize_whitespace(raw_text)
        
        if not raw_text.strip():
            return None
        
        # Compute dominant font info (most common size)
        dom_font_size = max(set(font_sizes), key=font_sizes.count) if font_sizes else 12.0
        dom_is_bold = any(bold_flags) and sum(bold_flags) > len(bold_flags) / 2
        dom_font_name = max(set(font_names), key=font_names.count) if font_names else ""
        
        # Extract bounding box
        bbox_raw = block.get("bbox", (0, 0, page_width, page_height))
        bbox = normalize_bbox(
            x0=bbox_raw[0],
            y0=bbox_raw[1],
            x1=bbox_raw[2],
            y1=bbox_raw[3],
            page_width=page_width,
            page_height=page_height,
            origin_bottom_left=False  # PyMuPDF uses top-left origin
        )
        
        # Detect text type and heading level
        heading_level = detect_heading_level(
            text=raw_text,
            font_size=dom_font_size,
            is_bold=dom_is_bold,
            is_uppercase=raw_text.isupper()
        )
        
        if heading_level:
            text_type = TextType.HEADING
        elif raw_text.strip().startswith(("•", "●", "○", "-", "–", "—")):
            text_type = TextType.BULLET
        else:
            text_type = TextType.BODY
        
        # Detect paragraph number
        para_num, cleaned_text = detect_paragraph_number(raw_text)
        if para_num:
            # Keep original text but note the paragraph number
            pass
        
        return RawTextBlock(
            text=raw_text,
            page_number=page_number,
            bbox=bbox,
            font_size=dom_font_size,
            is_bold=dom_is_bold,
            font_name=dom_font_name,
            text_type=text_type,
            heading_level=heading_level,
            paragraph_number=para_num,
            position_in_page=position,
            is_header_footer=False,
            extraction_confidence=1.0
        )
    
    def _merge_split_paragraphs(self, blocks: List[RawTextBlock]) -> List[RawTextBlock]:
        """
        Merge paragraphs that were incorrectly split by extraction.
        
        Heuristics for merging:
        - Same font size and style
        - Vertically adjacent (small y gap)
        - First block doesn't end with sentence-ending punctuation
        - Second block starts with lowercase
        """
        if len(blocks) <= 1:
            return blocks
        
        merged = []
        i = 0
        
        while i < len(blocks):
            current = blocks[i]
            
            # Check if we should merge with next block
            if i + 1 < len(blocks):
                next_block = blocks[i + 1]
                
                if self._should_merge(current, next_block):
                    # Create merged block
                    merged_text = current.text.rstrip() + " " + next_block.text.lstrip()
                    merged_bbox = BoundingBox(
                        x0=min(current.bbox.x0, next_block.bbox.x0),
                        y0=min(current.bbox.y0, next_block.bbox.y0),
                        x1=max(current.bbox.x1, next_block.bbox.x1),
                        y1=max(current.bbox.y1, next_block.bbox.y1)
                    )
                    
                    merged_block = RawTextBlock(
                        text=merged_text,
                        page_number=current.page_number,
                        bbox=merged_bbox,
                        font_size=current.font_size,
                        is_bold=current.is_bold,
                        font_name=current.font_name,
                        text_type=current.text_type,
                        heading_level=current.heading_level,
                        paragraph_number=current.paragraph_number,
                        position_in_page=current.position_in_page,
                        is_header_footer=current.is_header_footer,
                        extraction_confidence=min(current.extraction_confidence, 
                                                   next_block.extraction_confidence)
                    )
                    
                    merged.append(merged_block)
                    i += 2  # Skip both blocks
                    continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def _should_merge(self, block1: RawTextBlock, block2: RawTextBlock) -> bool:
        """
        Determine if two blocks should be merged.
        
        Conservative merging to avoid incorrect joins.
        """
        # Don't merge if on different pages
        if block1.page_number != block2.page_number:
            return False
        
        # Don't merge headings
        if block1.text_type == TextType.HEADING or block2.text_type == TextType.HEADING:
            return False
        
        # Don't merge if font size differs significantly
        if abs(block1.font_size - block2.font_size) > 0.5:
            return False
        
        # Don't merge if first block ends with sentence-ending punctuation
        text1 = block1.text.rstrip()
        if text1 and text1[-1] in ".!?:":
            return False
        
        # Don't merge if second block starts with uppercase (new sentence)
        text2 = block2.text.lstrip()
        if text2 and text2[0].isupper():
            # Exception: might be continuation with acronym
            if not text2[:3].isupper():  # Not an acronym
                return False
        
        # Don't merge if second block has paragraph number
        if block2.paragraph_number is not None:
            return False
        
        # Check vertical proximity (should be close)
        y_gap = block2.bbox.y0 - block1.bbox.y1
        if y_gap > 0.03:  # More than 3% of page height apart
            return False
        
        return True


# Example usage and output demonstration
if __name__ == "__main__":
    """
    Example output for one page:
    
    >>> from src.ingestion.loaders import PDFLoader
    >>> from src.ingestion.extractors import TextExtractor
    >>> 
    >>> doc = PDFLoader.load("qatar_test_doc.pdf")
    >>> extractor = TextExtractor()
    >>> 
    >>> page = doc.get_page(5)  # Page 5 has numbered paragraphs
    >>> blocks = extractor.extract(page)
    >>> 
    >>> for block in blocks[:3]:
    ...     print(f"---")
    ...     print(f"Position: {block.position_in_page}")
    ...     print(f"Type: {block.text_type}")
    ...     print(f"Heading Level: {block.heading_level}")
    ...     print(f"Para Number: {block.paragraph_number}")
    ...     print(f"BBox: ({block.bbox.x0:.2f}, {block.bbox.y0:.2f}) - ({block.bbox.x1:.2f}, {block.bbox.y1:.2f})")
    ...     print(f"Text: {block.text[:100]}...")
    ...
    ---
    Position: 0
    Type: TextType.HEADING
    Heading Level: 1
    Para Number: None
    BBox: (0.10, 0.08) - (0.90, 0.12)
    Text: KEY ISSUES...
    ---
    Position: 1
    Type: TextType.BODY
    Heading Level: None
    Para Number: None
    BBox: (0.10, 0.14) - (0.90, 0.20)
    Text: Context. Qatar has started the implementation of the much anticipated Third National Development S...
    ---
    Position: 2
    Type: TextType.BODY
    Heading Level: None
    Para Number: 1
    BBox: (0.10, 0.22) - (0.90, 0.28)
    Text: 1. Qatar has started implementing NDS3, which charts a new course for economic transformation...
    """
    print("Text Extractor module loaded. See docstring for example usage.")
