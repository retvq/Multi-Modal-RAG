"""
Figure Normalizer

Converts RawFigure from the extractor into schema-valid ContentBlocks.

Responsibilities:
- Generate stable block IDs for figures
- Populate FIGURE-specific metadata
- Map caption/title to figure
- Generate figure-aware citations
- Handle multi-panel metadata

IMPORTANT: This normalizer does NOT generate interpretations or descriptions
of figure content. The content field contains the title/caption, and the
image_path points to the raw figure for downstream vision processing.
"""

import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_path))

from src.ingestion.extractors.figure_extractor import RawFigure, PanelInfo
from src.ingestion.utils.id_generator import generate_block_id, hash_content
from src.schemas.content_block import (
    ContentBlock,
    Modality,
    BoundingBox,
)


@dataclass
class FigureMetadata:
    """
    Modality-specific metadata for FIGURE blocks.
    
    Preserves structural information about the figure
    for downstream processing.
    """
    figure_id: Optional[str] = None     # e.g., "Figure 1"
    figure_title: str = ""              # Full title
    figure_type: str = "UNKNOWN"        # LINE_CHART, BAR_CHART, etc.
    
    # Multi-panel support
    is_multi_panel: bool = False
    panel_count: int = 1
    panels: List[PanelInfo] = field(default_factory=list)
    
    # Image reference
    image_path: Optional[str] = None    # Path to saved image
    image_width: int = 0
    image_height: int = 0
    
    # Extracted text
    caption: str = ""
    extracted_labels: List[str] = field(default_factory=list)
    
    # OCR flag
    ocr_applied: bool = False


class FigureNormalizer:
    """
    Converts RawFigure to ContentBlock with modality=FIGURE.
    
    The content field contains the figure title and caption concatenated.
    The actual image is referenced via image_path in the metadata.
    
    This allows:
    - Text-based retrieval using title/caption
    - Image-based processing via stored image file
    - Multi-panel awareness for downstream systems
    
    Usage:
        normalizer = FigureNormalizer(document_id="qatar_2024")
        block = normalizer.normalize(raw_figure, section_hierarchy)
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
        raw: RawFigure,
        section_hierarchy: List[str],
        section_id: Optional[str] = None,
        position_in_section: int = 0
    ) -> ContentBlock:
        """
        Convert a raw figure to a ContentBlock.
        
        The content field contains title + caption for text-based retrieval.
        Full image data is preserved in metadata.
        
        Args:
            raw: RawFigure from extractor
            section_hierarchy: Ordered path from document root
            section_id: Section identifier (optional)
            position_in_section: Ordinal position within section
            
        Returns:
            Schema-valid ContentBlock with modality=FIGURE
        """
        # Generate deterministic block ID
        content_for_hash = raw.figure_title or raw.caption or f"figure_{raw.page_number}_{raw.position_in_page}"
        content_hash = hash_content(content_for_hash)
        
        block_id = generate_block_id(
            document_id=self.document_id,
            page_number=raw.page_number,
            modality="FIGURE",
            content_hash=content_hash,
            position_in_page=raw.position_in_page
        )
        
        # Build figure metadata
        figure_metadata = FigureMetadata(
            figure_id=raw.figure_id,
            figure_title=raw.figure_title,
            figure_type=raw.figure_type,
            is_multi_panel=raw.is_multi_panel,
            panel_count=raw.panel_count,
            panels=raw.panels,
            image_path=raw.image_path,
            image_width=raw.image_width,
            image_height=raw.image_height,
            caption=raw.caption,
            extracted_labels=raw.extracted_labels,
            ocr_applied=raw.ocr_applied
        )
        
        # Ensure section hierarchy is not empty
        if not section_hierarchy:
            section_hierarchy = ["Document", "Figures"]
        
        # Content combines title and caption for text-based retrieval
        # No interpretation or description is added
        content_parts = []
        if raw.figure_title:
            content_parts.append(raw.figure_title)
        if raw.caption:
            content_parts.append(raw.caption)
        if raw.extracted_labels:
            # Include OCR labels for searchability
            content_parts.append("Labels: " + ", ".join(raw.extracted_labels[:10]))
        
        content = "\n\n".join(content_parts) if content_parts else f"Figure on page {raw.page_number}"
        
        # Create ContentBlock
        block = ContentBlock(
            block_id=block_id,
            document_id=self.document_id,
            modality=Modality.FIGURE,
            content=content,
            page_number=raw.page_number,
            section_hierarchy=section_hierarchy,
            section_id=section_id,
            position_in_section=position_in_section,
            bounding_box=raw.bbox,
            extraction_confidence=raw.extraction_confidence,
            extraction_warnings=raw.extraction_warnings.copy() if raw.extraction_warnings else []
        )
        
        # Override default citations for figure-specific format
        fig_name = raw.figure_id or "Figure"
        block.citation_short = f"[{fig_name}, Page {raw.page_number}]"
        block.citation_full = f"{fig_name}: {raw.figure_title}, Page {raw.page_number}"
        block.citation_section = block.citation_short
        
        # Add multi-panel warning if applicable
        if raw.is_multi_panel:
            block.extraction_warnings.append(f"Multi-panel figure with {raw.panel_count} panels")
        
        # Store figure metadata (accessible via _figure_metadata)
        block._figure_metadata = figure_metadata
        
        return block
    
    def normalize_batch(
        self,
        raw_figures: List[RawFigure],
        section_hierarchy: List[str],
        section_id: Optional[str] = None
    ) -> List[ContentBlock]:
        """
        Normalize a batch of raw figures.
        """
        return [
            self.normalize(
                raw=raw,
                section_hierarchy=section_hierarchy,
                section_id=section_id,
                position_in_section=i
            )
            for i, raw in enumerate(raw_figures)
        ]


# Example ContentBlock output for multi-panel figure
EXAMPLE_OUTPUT = """
Example ContentBlock for multi-panel Figure 1:

ContentBlock(
    block_id='qatar_p034_figure_c3d4e5f6_000',
    document_id='qatar_2024',
    modality=<Modality.FIGURE: 'FIGURE'>,
    content='Figure 1. Qatar: Real Sector Developments

Sources: Haver Analytics, QCB, and IMF staff calculations.

Labels: 2019, 2020, 2021, 2022, 2023, 2024, Percent, YoY',
    page_number=34,
    section_hierarchy=['Staff Report', 'Real Sector Developments'],
    citation_short='[Figure 1, Page 34]',
    citation_full='Figure 1: Figure 1. Qatar: Real Sector Developments, Page 34',
    content_length=156,
    extraction_confidence=1.0,
    extraction_warnings=['Multi-panel figure with 6 panels'],
    _figure_metadata=FigureMetadata(
        figure_id='Figure 1',
        figure_title='Figure 1. Qatar: Real Sector Developments',
        figure_type='CHART',
        is_multi_panel=True,
        panel_count=6,
        panels=[
            PanelInfo(panel_index=0, panel_title='Real GDP Growth'),
            PanelInfo(panel_index=1, panel_title='Non-hydrocarbon Growth'),
            PanelInfo(panel_index=2, panel_title='PMI'),
            PanelInfo(panel_index=3, panel_title='Tourism'),
            PanelInfo(panel_index=4, panel_title='Construction'),
            PanelInfo(panel_index=5, panel_title='Employment'),
        ],
        image_path='outputs/figures/qatar_p034_fig00.png',
        image_width=1200,
        image_height=900,
        caption='Sources: Haver Analytics, QCB, and IMF staff calculations.',
        extracted_labels=['2019', '2020', '2021', '2022', '2023', '2024', 'Percent', 'YoY'],
        ocr_applied=True
    )
)

Key Design Decisions:
1. Content = title + caption + OCR labels (for text retrieval)
2. No visual interpretation/description generated
3. Image path preserved for downstream vision processing
4. Multi-panel awareness via panel_count and panels list
5. OCR labels included for keyword searchability
"""

if __name__ == "__main__":
    print(EXAMPLE_OUTPUT)
