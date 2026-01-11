"""
Footnote Linker

Links footnote blocks to the text/table blocks that reference them.

Creates bidirectional relationships:
- Text block → Footnote block (FOOTNOTE_REFERENCE)
- Updates footnote's referenced_by field

Error Handling:
- Orphan footnotes (no referencing block) are logged but kept
- Orphan markers (no footnote content) are logged
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_path))

from src.schemas.content_block import ContentBlock, Modality
from src.schemas.relationship import Relationship, RelationshipType
from src.ingestion.utils.id_generator import hash_content


@dataclass
class OrphanFootnote:
    """A footnote with no detected referencing block."""
    block_id: str
    marker: str
    page_number: int
    content_preview: str


@dataclass
class OrphanMarker:
    """A marker in text with no matching footnote content."""
    source_block_id: str
    marker: str
    page_number: int


class FootnoteLinker:
    """
    Links footnote references to footnote content blocks.
    
    Strategy:
    1. Index all footnote blocks by their normalized marker
    2. Scan text/table blocks for footnote markers
    3. Create FOOTNOTE_REFERENCE relationships
    4. Report orphans (unlinked footnotes or markers)
    
    IMF Pattern Recognition:
    - Text footnotes: superscript numbers (1, 2, 3)
    - Table footnotes: N/ format (1/, 2/, 3/)
    - Source attributions: "Sources:" text
    """
    
    def __init__(self):
        # Patterns for detecting footnote markers in content
        self.marker_patterns = [
            # Superscript-style (appears at word boundaries)
            r'(?<=[a-zA-Z.)])\s*(\d)\s*(?=[,.\s])',
            # Table-style N/ markers
            r'(\d+/)',
        ]
    
    def link(
        self,
        blocks: List[ContentBlock]
    ) -> Tuple[List[Relationship], List[OrphanFootnote], List[OrphanMarker]]:
        """
        Link footnote references to footnotes.
        
        Args:
            blocks: All ContentBlocks including footnotes
            
        Returns:
            (relationships, orphan_footnotes, orphan_markers)
        """
        relationships = []
        orphan_footnotes = []
        orphan_markers = []
        
        # Build footnote index
        footnote_index = self._build_footnote_index(blocks)
        
        # Track which footnotes get referenced
        referenced_footnotes = set()
        
        # Scan non-footnote blocks for markers
        for block in blocks:
            if block.modality == Modality.FOOTNOTE:
                continue
            
            # Find markers in this block's content
            markers = self._find_markers(block.content, block.modality)
            
            for marker in markers:
                # Normalize marker for lookup
                normalized = self._normalize_marker(marker)
                
                # Look up footnote on same page first, then nearby pages
                footnote_block = self._find_footnote(
                    normalized_marker=normalized,
                    source_page=block.page_number,
                    footnote_index=footnote_index
                )
                
                if footnote_block:
                    # Create relationship
                    rel_id = f"fnref_{hash_content(f'{block.block_id}_{footnote_block.block_id}')}[:16]"
                    
                    relationships.append(Relationship(
                        relationship_id=rel_id,
                        source_block_id=block.block_id,
                        target_block_id=footnote_block.block_id,
                        relationship_type=RelationshipType.FOOTNOTE_REFERENCE,
                        reference_text=marker
                    ))
                    
                    referenced_footnotes.add(footnote_block.block_id)
                else:
                    # Only report as orphan if it looks like a real footnote marker
                    # (single digits in context that suggests footnote)
                    if len(normalized) <= 2 and block.modality in [Modality.TEXT, Modality.TABLE]:
                        orphan_markers.append(OrphanMarker(
                            source_block_id=block.block_id,
                            marker=marker,
                            page_number=block.page_number
                        ))
        
        # Find orphan footnotes (not referenced by any block)
        for marker, page_dict in footnote_index.items():
            for page, fn_block in page_dict.items():
                if fn_block.block_id not in referenced_footnotes:
                    orphan_footnotes.append(OrphanFootnote(
                        block_id=fn_block.block_id,
                        marker=marker,
                        page_number=page,
                        content_preview=fn_block.content[:50] if fn_block.content else ""
                    ))
        
        return (relationships, orphan_footnotes, orphan_markers)
    
    def _build_footnote_index(
        self,
        blocks: List[ContentBlock]
    ) -> Dict[str, Dict[int, ContentBlock]]:
        """
        Build index of footnotes by normalized marker and page.
        
        Returns:
            {
                "1": {39: block, 40: block},
                "2": {39: block},
                ...
            }
        """
        index = {}
        
        for block in blocks:
            if block.modality != Modality.FOOTNOTE:
                continue
            
            # Get marker from stored metadata
            marker = getattr(block, '_normalized_marker', None)
            if not marker:
                # Try to extract from content
                match = re.match(r'\[(\d+/?)\]', block.content)
                if match:
                    marker = self._normalize_marker(match.group(1))
            
            if marker:
                if marker not in index:
                    index[marker] = {}
                index[marker][block.page_number] = block
        
        return index
    
    def _find_markers(self, content: str, modality: Modality) -> List[str]:
        """
        Find footnote markers in content.
        """
        markers = []
        
        if modality == Modality.TABLE:
            # Tables use N/ format
            matches = re.findall(r'(\d+/)', content)
            markers.extend(matches)
        else:
            # Text uses various formats
            # Look for patterns that suggest footnote markers
            # Be conservative to avoid false positives
            
            # N/ format in any content
            matches = re.findall(r'(\d+/)', content)
            markers.extend(matches)
        
        return markers
    
    def _normalize_marker(self, marker: str) -> str:
        """Normalize marker for matching."""
        return marker.rstrip("/").strip()
    
    def _find_footnote(
        self,
        normalized_marker: str,
        source_page: int,
        footnote_index: Dict[str, Dict[int, ContentBlock]]
    ) -> Optional[ContentBlock]:
        """
        Find footnote block matching marker and page.
        
        Prefers same-page footnotes, then checks nearby pages.
        """
        if normalized_marker not in footnote_index:
            return None
        
        page_dict = footnote_index[normalized_marker]
        
        # Same page first
        if source_page in page_dict:
            return page_dict[source_page]
        
        # Check nearby pages (footnotes might appear on next page)
        for delta in [1, -1, 2, -2]:
            nearby = source_page + delta
            if nearby in page_dict:
                return page_dict[nearby]
        
        # Any page as last resort
        if page_dict:
            return list(page_dict.values())[0]
        
        return None


EXAMPLE_OUTPUT = """
Example Footnote Relationship:

Relationship(
    relationship_id='fnref_a1b2c3d4e5f6_001',
    source_block_id='qatar_p039_table_xxx_000',  # Table with "1/" marker
    target_block_id='qatar_p039_footnote_yyy_000',  # Footnote content
    relationship_type=<RelationshipType.FOOTNOTE_REFERENCE: 'FOOTNOTE_REFERENCE'>,
    reference_text='1/'
)

Example OrphanFootnote (no referencing block found):

OrphanFootnote(
    block_id='qatar_p042_footnote_zzz_000',
    marker='5',
    page_number=42,
    content_preview='This footnote was extracted but no marker found...'
)

Example OrphanMarker (marker found but no footnote content):

OrphanMarker(
    source_block_id='qatar_p015_text_xxx_003',
    marker='3/',
    page_number=15
)
"""

if __name__ == "__main__":
    print(EXAMPLE_OUTPUT)
