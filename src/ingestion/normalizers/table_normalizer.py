"""
Table Normalizer

Converts RawTable from the extractor into schema-valid ContentBlocks.

Responsibilities:
- Generate stable block IDs for tables
- Populate TABLE-specific metadata
- Create both structured and text content representations
- Generate table-aware citations
- Validate output
"""

import sys
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_path))

from src.ingestion.extractors.table_extractor import RawTable, TableStructure
from src.ingestion.utils.id_generator import generate_block_id, hash_content
from src.schemas.content_block import (
    ContentBlock,
    Modality,
    BoundingBox,
)


@dataclass
class TableMetadata:
    """
    Modality-specific metadata for TABLE blocks.
    
    Contains structural information about the table
    beyond what's in the content field.
    """
    table_id: Optional[str] = None      # e.g., "Table 1"
    table_title: str = ""               # Full title
    column_headers: List[str] = field(default_factory=list)
    row_headers: List[str] = field(default_factory=list)
    column_count: int = 0
    row_count: int = 0
    has_nested_headers: bool = False
    has_merged_cells: bool = False
    units: Optional[str] = None         # e.g., "Percent of GDP"
    footnote_refs: List[str] = field(default_factory=list)
    
    # Structured data preserved
    structured_data: Optional[TableStructure] = None
    
    # Alternative representations
    markdown_repr: str = ""
    linear_text: str = ""


class TableNormalizer:
    """
    Converts RawTable to ContentBlock with modality=TABLE.
    
    Each RawTable becomes one ContentBlock containing:
    - Content: markdown representation (best for LLM consumption)
    - Metadata: full structural information
    
    Usage:
        normalizer = TableNormalizer(document_id="qatar_2024")
        block = normalizer.normalize(raw_table, section_hierarchy)
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
        raw: RawTable,
        section_hierarchy: List[str],
        section_id: Optional[str] = None,
        position_in_section: int = 0
    ) -> ContentBlock:
        """
        Convert a raw table to a ContentBlock.
        
        The content field contains the markdown representation,
        which is most useful for LLM consumption. The full
        structural data is preserved in table_metadata.
        
        Args:
            raw: RawTable from extractor
            section_hierarchy: Ordered path from document root
            section_id: Section identifier (optional)
            position_in_section: Ordinal position within section
            
        Returns:
            Schema-valid ContentBlock with modality=TABLE
        """
        # Generate deterministic block ID
        # Use table title + page for unique identification
        content_for_hash = raw.table_title or raw.raw_text or f"table_{raw.page_number}_{raw.position_in_page}"
        content_hash = hash_content(content_for_hash)
        
        block_id = generate_block_id(
            document_id=self.document_id,
            page_number=raw.page_number,
            modality="TABLE",
            content_hash=content_hash,
            position_in_page=raw.position_in_page
        )
        
        # Build table metadata
        has_nested_headers = False
        if raw.structure and raw.structure.headers:
            has_nested_headers = len(raw.structure.headers) > 1
        
        table_metadata = TableMetadata(
            table_id=raw.table_id,
            table_title=raw.table_title,
            column_headers=raw.column_headers,
            row_headers=raw.row_headers,
            column_count=raw.structure.column_count if raw.structure else 0,
            row_count=raw.structure.row_count if raw.structure else 0,
            has_nested_headers=has_nested_headers,
            has_merged_cells=raw.structure.has_merged_cells if raw.structure else False,
            units=raw.units,
            footnote_refs=raw.footnote_refs,
            structured_data=raw.structure,
            markdown_repr=raw.markdown,
            linear_text=raw.raw_text
        )
        
        # Ensure section hierarchy is not empty
        if not section_hierarchy:
            section_hierarchy = ["Document", "Tables"]
        
        # Content is markdown for best LLM compatibility
        # Prepend title for context
        content = ""
        if raw.table_title:
            content = f"{raw.table_title}\n\n"
        content += raw.markdown or raw.raw_text
        
        # Create ContentBlock
        block = ContentBlock(
            block_id=block_id,
            document_id=self.document_id,
            modality=Modality.TABLE,
            content=content,
            page_number=raw.page_number,
            section_hierarchy=section_hierarchy,
            section_id=section_id,
            position_in_section=position_in_section,
            bounding_box=raw.bbox,
            extraction_confidence=raw.extraction_confidence,
            extraction_warnings=raw.extraction_warnings.copy() if raw.extraction_warnings else []
        )
        
        # Override default citations for table-specific format
        block.citation_short = f"[{raw.table_id or 'Table'}, Page {raw.page_number}]"
        block.citation_full = f"{raw.table_id or 'Table'}: {raw.table_title}, Page {raw.page_number}"
        block.citation_section = block.citation_short
        
        # Store table metadata (not in base schema, but accessible)
        block._table_metadata = table_metadata
        
        return block
    
    def normalize_batch(
        self,
        raw_tables: List[RawTable],
        section_hierarchy: List[str],
        section_id: Optional[str] = None
    ) -> List[ContentBlock]:
        """
        Normalize a batch of raw tables.
        
        Args:
            raw_tables: List of RawTables from extractor
            section_hierarchy: Section context for all tables
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
            for i, raw in enumerate(raw_tables)
        ]


# Example output demonstration  
EXAMPLE_OUTPUT = """
Example ContentBlock output from table normalization (Table 1):

ContentBlock(
    block_id='qatar_p039_table_e5f6g7h8_000',
    document_id='qatar_2024',
    modality=<Modality.TABLE: 'TABLE'>,
    content='Table 1. Qatar: Selected Macroeconomic Indicators, 2020-29
    
| | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | 2027 | 2028 | 2029 |
|---|---|---|---|---|---|---|---|---|---|---|
| Real GDP growth | -3.6 | 1.6 | 4.9 | 1.2 | 1.7 | 2.4 | 5.2 | 7.9 | 3.5 | 1.6 |
| Hydrocarbon | -2.0 | 0.7 | 1.0 | -0.5 | -0.8 | -0.2 | 7.5 | 15.3 | 5.3 | 1.1 |
| Non-hydrocarbon | -4.8 | 2.4 | 7.5 | 2.5 | 3.5 | 4.1 | 3.6 | 2.7 | 2.2 | 2.0 |
...',
    page_number=39,
    section_hierarchy=['Staff Report', 'Statistical Appendix'],
    citation_short='[Table 1, Page 39]',
    citation_full='Table 1: Table 1. Qatar: Selected Macroeconomic Indicators, 2020-29, Page 39',
    content_length=1847,
    extraction_confidence=1.0,
    _table_metadata=TableMetadata(
        table_id='Table 1',
        table_title='Table 1. Qatar: Selected Macroeconomic Indicators, 2020-29',
        column_headers=['', '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029'],
        column_count=11,
        row_count=45,
        has_nested_headers=True,  # Has year row + Actual/Proj row
        has_merged_cells=False,
        units='Percent',
        footnote_refs=['1/', '2/', '3/']
    )
)
"""

if __name__ == "__main__":
    print(EXAMPLE_OUTPUT)
