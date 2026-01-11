"""
Text Normalizer

Converts RawTextBlocks from the extractor into schema-valid ContentBlocks.

This is the bridge between extraction (raw data) and the normalized output
format that downstream systems consume.

Responsibilities:
- Generate stable block IDs
- Populate all required ContentBlock fields
- Attach section context
- Generate citations
- Validate output
"""

import sys
from pathlib import Path
from typing import List, Optional

# Add src to path for imports
src_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_path))

from src.ingestion.extractors.text_extractor import RawTextBlock
from src.ingestion.utils.id_generator import generate_block_id, hash_content
from src.schemas.content_block import (
    ContentBlock,
    Modality,
    TextMetadata,
    TextType,
)


class TextNormalizer:
    """
    Converts RawTextBlocks to ContentBlocks.
    
    Each RawTextBlock becomes exactly one ContentBlock with:
    - Unique, deterministic block_id
    - Modality = TEXT
    - Populated provenance fields
    - Generated citations
    
    Usage:
        normalizer = TextNormalizer(document_id="qatar_2024")
        blocks = [normalizer.normalize(raw, section_path) for raw in raw_blocks]
    """
    
    def __init__(self, document_id: str):
        """
        Initialize normalizer for a specific document.
        
        Args:
            document_id: Stable document identifier
        """
        self.document_id = document_id
    
    def normalize(
        self,
        raw: RawTextBlock,
        section_hierarchy: List[str],
        section_id: Optional[str] = None,
        position_in_section: int = 0
    ) -> ContentBlock:
        """
        Convert a raw text block to a ContentBlock.
        
        Args:
            raw: RawTextBlock from extractor
            section_hierarchy: Ordered path from document root
            section_id: Section identifier (optional)
            position_in_section: Ordinal position within section
            
        Returns:
            Schema-valid ContentBlock
        """
        # Generate deterministic block ID
        content_hash = hash_content(raw.text)
        block_id = generate_block_id(
            document_id=self.document_id,
            page_number=raw.page_number,
            modality="TEXT",
            content_hash=content_hash,
            position_in_page=raw.position_in_page
        )
        
        # Build text metadata
        text_metadata = TextMetadata(
            paragraph_number=raw.paragraph_number,
            text_type=raw.text_type,
            heading_level=raw.heading_level,
            has_cross_references=self._has_cross_references(raw.text),
            font_size=raw.font_size,
            is_bold=raw.is_bold
        )
        
        # Ensure section hierarchy is not empty
        if not section_hierarchy:
            section_hierarchy = ["Document"]
        
        # Create ContentBlock
        block = ContentBlock(
            block_id=block_id,
            document_id=self.document_id,
            modality=Modality.TEXT,
            content=raw.text,
            page_number=raw.page_number,
            section_hierarchy=section_hierarchy,
            section_id=section_id,
            position_in_section=position_in_section,
            bounding_box=raw.bbox,
            text_metadata=text_metadata,
            extraction_confidence=raw.extraction_confidence,
            extraction_warnings=[]
        )
        
        # Add warning if header/footer marked but included
        if raw.is_header_footer:
            block.extraction_warnings.append("Content identified as header/footer")
        
        return block
    
    def normalize_batch(
        self,
        raw_blocks: List[RawTextBlock],
        section_hierarchy: List[str],
        section_id: Optional[str] = None
    ) -> List[ContentBlock]:
        """
        Normalize a batch of raw blocks with sequential positioning.
        
        Args:
            raw_blocks: List of RawTextBlocks from extractor
            section_hierarchy: Section context for all blocks
            section_id: Section identifier
            
        Returns:
            List of ContentBlocks in order
        """
        return [
            self.normalize(
                raw=raw,
                section_hierarchy=section_hierarchy,
                section_id=section_id,
                position_in_section=i
            )
            for i, raw in enumerate(raw_blocks)
        ]
    
    def _has_cross_references(self, text: str) -> bool:
        """
        Detect if text contains cross-references to other elements.
        
        Patterns detected:
        - "Table N"
        - "Figure N" / "Text Figure N"
        - "Box N"
        - "Annex N"
        """
        import re
        
        patterns = [
            r"\bTable\s+\d+",
            r"\bFigure\s+\d+",
            r"\bText\s+Figure\s+\d+",
            r"\bBox\s+\d+",
            r"\bAnnex\s+[IVX]+",
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False


# Example output demonstration
EXAMPLE_OUTPUT = """
Example ContentBlock output from normalization:

ContentBlock(
    block_id='qatar_p005_text_a1b2c3d4_002',
    document_id='qatar_2024',
    modality=<Modality.TEXT: 'TEXT'>,
    content='1. Qatar has started implementing NDS3, which charts a new course for economic transformation. The strategy aims to leverage the country's substantial LNG expansion to diversify the economy and enhance private sector participation.',
    page_number=5,
    section_hierarchy=['Staff Report', 'Context'],
    section_id='context',
    position_in_section=2,
    bounding_box=BoundingBox(x0=0.10, y0=0.22, x1=0.90, y1=0.28),
    text_metadata=TextMetadata(
        paragraph_number=1,
        text_type=<TextType.BODY: 'BODY'>,
        heading_level=None,
        has_cross_references=False,
        font_size=10.5,
        is_bold=False
    ),
    citation_short='[Page 5]',
    citation_full='Context, Page 5',
    citation_section='[Staff Report > Context, Para 1]',
    content_length=235,
    is_complete=True,
    extraction_confidence=1.0,
    extraction_warnings=[]
)
"""

if __name__ == "__main__":
    print(EXAMPLE_OUTPUT)
