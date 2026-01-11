"""
Relationship Schema Definition

Relationships connect ContentBlocks to each other, capturing:
- Cross-references (text → table, text → figure)
- Footnote links (marker → footnote content)
- Containment (box → child elements)
- Continuation (multi-page elements)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RelationshipType(Enum):
    """Types of relationships between content blocks."""
    CROSS_REFERENCE = "CROSS_REFERENCE"      # "See Table 1", "Figure 2 shows..."
    FOOTNOTE_REFERENCE = "FOOTNOTE_REFERENCE" # Superscript → footnote
    PARENT_CHILD = "PARENT_CHILD"             # Box contains paragraphs
    CONTINUATION = "CONTINUATION"              # Block continues on next page
    SUPPORTS = "SUPPORTS"                      # Evidence relationship


@dataclass
class Relationship:
    """
    A directed relationship between two content blocks.
    
    Relationships are stored separately from blocks to allow:
    - Many-to-many relationships
    - Relationship queries without loading full blocks
    - Validation that both endpoints exist
    """
    
    relationship_id: str
    source_block_id: str              # Block containing the reference
    target_block_id: str              # Block being referenced
    relationship_type: RelationshipType
    reference_text: str = ""          # The text of the reference (e.g., "See Table 1")
    source_position: Optional[tuple[int, int]] = None  # Character offsets in source
    
    def validate(self, known_block_ids: set[str]) -> tuple[bool, list[str]]:
        """
        Validate relationship integrity.
        
        Args:
            known_block_ids: Set of all valid block IDs in the document
            
        Returns:
            (is_valid, list of error messages)
        """
        errors = []
        
        if not self.relationship_id:
            errors.append("relationship_id is required")
        if self.source_block_id not in known_block_ids:
            errors.append(f"source_block_id '{self.source_block_id}' not found")
        if self.target_block_id not in known_block_ids:
            errors.append(f"target_block_id '{self.target_block_id}' not found")
        if self.source_block_id == self.target_block_id:
            errors.append("source and target cannot be the same block")
        
        return (len(errors) == 0, errors)
