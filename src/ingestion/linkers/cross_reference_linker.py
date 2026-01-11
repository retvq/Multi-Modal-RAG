"""
Cross-Reference Linker

Detects and resolves cross-references between document elements.

Cross-references in IMF documents include:
- "See Table 1" / "Table 3 shows..."
- "See Figure 2" / "as illustrated in Figure 4"
- "See Annex IV" / "Annex II provides..."
- "See Box 1" / "Box 2 discusses..."

This linker creates Relationship records connecting
referencing text blocks to referenced elements.
"""

import re
from dataclasses import dataclass, field
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
class UnresolvedReference:
    """
    A cross-reference that could not be resolved to a target block.
    
    Used for error reporting and quality assessment.
    """
    source_block_id: str
    reference_text: str
    reference_type: str                 # TABLE, FIGURE, ANNEX, BOX
    target_id: str                      # e.g., "Table 1", "Annex IV"
    page_number: int
    reason: str = "Target not found"


class CrossReferenceLinker:
    """
    Detects cross-references in text blocks and creates relationships.
    
    Strategy:
    1. Scan all TEXT blocks for reference patterns
    2. Extract target identifiers (e.g., "Table 1", "Annex IV")
    3. Match to TABLE/FIGURE/ANNEX/BOX blocks
    4. Create Relationship records
    5. Report unresolved references
    
    Error Handling:
    - Unresolved references are logged but don't fail ingestion
    - Ambiguous references (multiple matches) logged as warnings
    """
    
    def __init__(self):
        # Reference patterns with named groups
        self.patterns = [
            # Tables: "Table 1", "Table 3a", "Tables 1 and 2"
            (r'\b(Table\s+(\d+[a-z]?))\b', "TABLE"),
            # Figures: "Figure 1", "Text Figure 8", "Figures 1-3"
            (r'\b(Text\s+Figure\s+(\d+))\b', "FIGURE"),
            (r'\b(Figure\s+(\d+))\b', "FIGURE"),
            # Annexes: "Annex I", "Annex IV", "Annexes I and II"
            (r'\b(Annex\s+([IVX]+))\b', "ANNEX"),
            # Boxes: "Box 1", "Box 2"
            (r'\b(Box\s+(\d+))\b', "BOX"),
        ]
    
    def link(
        self,
        blocks: List[ContentBlock]
    ) -> Tuple[List[Relationship], List[UnresolvedReference]]:
        """
        Detect and create cross-reference relationships.
        
        Args:
            blocks: All ContentBlocks from the document
            
        Returns:
            (list of Relationships, list of UnresolvedReferences)
        """
        relationships = []
        unresolved = []
        
        # Build index of target blocks by ID
        target_index = self._build_target_index(blocks)
        
        # Scan text blocks for references
        for block in blocks:
            if block.modality != Modality.TEXT:
                continue
            
            # Find all references in this block
            refs = self._find_references(block.content)
            
            for ref_text, ref_type, ref_id in refs:
                # Try to resolve reference
                target_block = self._resolve_reference(
                    ref_type=ref_type,
                    ref_id=ref_id,
                    target_index=target_index
                )
                
                if target_block:
                    # Create relationship
                    rel_id = f"xref_{hash_content(f'{block.block_id}_{target_block.block_id}')}[:16]"
                    
                    relationships.append(Relationship(
                        relationship_id=rel_id,
                        source_block_id=block.block_id,
                        target_block_id=target_block.block_id,
                        relationship_type=RelationshipType.CROSS_REFERENCE,
                        reference_text=ref_text,
                        source_position=None  # Would need to track char offset
                    ))
                else:
                    # Log unresolved reference
                    unresolved.append(UnresolvedReference(
                        source_block_id=block.block_id,
                        reference_text=ref_text,
                        reference_type=ref_type,
                        target_id=ref_id,
                        page_number=block.page_number,
                        reason="Target not found in document"
                    ))
        
        return (relationships, unresolved)
    
    def _build_target_index(
        self,
        blocks: List[ContentBlock]
    ) -> Dict[str, Dict[str, ContentBlock]]:
        """
        Build index of potential reference targets.
        
        Returns:
            {
                "TABLE": {"1": block, "3a": block, ...},
                "FIGURE": {"1": block, ...},
                "ANNEX": {"I": block, "IV": block, ...},
                "BOX": {"1": block, ...}
            }
        """
        index = {
            "TABLE": {},
            "FIGURE": {},
            "ANNEX": {},
            "BOX": {}
        }
        
        for block in blocks:
            # Check for table ID
            if block.modality == Modality.TABLE:
                if hasattr(block, '_table_metadata') and block._table_metadata.table_id:
                    # Extract number from "Table 1" -> "1"
                    match = re.search(r'Table\s+(\d+[a-z]?)', block._table_metadata.table_id, re.IGNORECASE)
                    if match:
                        index["TABLE"][match.group(1)] = block
            
            # Check for figure ID
            elif block.modality == Modality.FIGURE:
                if hasattr(block, '_figure_metadata') and block._figure_metadata.figure_id:
                    # "Figure 1" -> "1", "Text Figure 8" -> "8"
                    match = re.search(r'(?:Text\s+)?Figure\s+(\d+)', block._figure_metadata.figure_id, re.IGNORECASE)
                    if match:
                        index["FIGURE"][match.group(1)] = block
            
            # Check for annex
            elif "annex" in block.content.lower()[:100] or \
                 any("annex" in s.lower() for s in block.section_hierarchy):
                # Try to extract annex ID from content or hierarchy
                for text in [block.content[:200]] + block.section_hierarchy:
                    match = re.search(r'Annex\s+([IVX]+)', text, re.IGNORECASE)
                    if match:
                        index["ANNEX"][match.group(1).upper()] = block
                        break
            
            # Check for box
            if "box" in block.content.lower()[:50]:
                match = re.search(r'Box\s+(\d+)', block.content[:100], re.IGNORECASE)
                if match:
                    index["BOX"][match.group(1)] = block
        
        return index
    
    def _find_references(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Find all cross-references in text.
        
        Returns:
            List of (full_match, type, id) tuples
            e.g., [("Table 1", "TABLE", "1"), ("Annex IV", "ANNEX", "IV")]
        """
        results = []
        
        for pattern, ref_type in self.patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                full_match = match.group(1)
                ref_id = match.group(2)
                results.append((full_match, ref_type, ref_id))
        
        return results
    
    def _resolve_reference(
        self,
        ref_type: str,
        ref_id: str,
        target_index: Dict[str, Dict[str, ContentBlock]]
    ) -> Optional[ContentBlock]:
        """
        Resolve a reference to its target block.
        """
        if ref_type not in target_index:
            return None
        
        # Normalize ID for matching
        normalized_id = ref_id.strip()
        
        # Direct lookup
        if normalized_id in target_index[ref_type]:
            return target_index[ref_type][normalized_id]
        
        # Case-insensitive lookup for annexes
        if ref_type == "ANNEX":
            normalized_upper = normalized_id.upper()
            if normalized_upper in target_index[ref_type]:
                return target_index[ref_type][normalized_upper]
        
        return None


EXAMPLE_OUTPUT = """
Example Relationship records:

# Cross-reference from text to table
Relationship(
    relationship_id='xref_a1b2c3d4e5f6_001',
    source_block_id='qatar_p012_text_xxx_005',
    target_block_id='qatar_p039_table_yyy_000',
    relationship_type=<RelationshipType.CROSS_REFERENCE: 'CROSS_REFERENCE'>,
    reference_text='Table 1',
    source_position=(245, 252)
)

# Cross-reference from text to annex
Relationship(
    relationship_id='xref_g7h8i9j0k1l2_002',
    source_block_id='qatar_p008_text_xxx_003',
    target_block_id='qatar_p055_text_zzz_000',
    relationship_type=<RelationshipType.CROSS_REFERENCE: 'CROSS_REFERENCE'>,
    reference_text='Annex IV',
    source_position=(89, 97)
)

Example UnresolvedReference:

UnresolvedReference(
    source_block_id='qatar_p015_text_xxx_007',
    reference_text='Table 9',
    reference_type='TABLE',
    target_id='9',
    page_number=15,
    reason='Target not found in document'
)
"""

if __name__ == "__main__":
    print(EXAMPLE_OUTPUT)
