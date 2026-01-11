"""
Ingestion Manifest Schema

The manifest summarizes the results of document ingestion, providing:
- Document metadata
- Block counts by modality
- Warnings and errors
- Processing statistics
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class IngestionStatus(Enum):
    """Overall status of ingestion operation."""
    SUCCESS = "SUCCESS"        # All content extracted without errors
    PARTIAL = "PARTIAL"        # Some content extracted, some failed
    FAILED = "FAILED"          # Document could not be processed


@dataclass
class IngestionManifest:
    """
    Summary of document ingestion results.
    
    This manifest allows downstream systems to:
    - Verify ingestion completed successfully
    - Understand document composition
    - Review warnings before proceeding
    """
    
    # === Document Identity ===
    document_id: str
    source_path: str
    
    # === Processing Metadata ===
    status: IngestionStatus
    ingestion_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    processing_duration_seconds: float = 0.0
    
    # === Document Statistics ===
    total_pages: int = 0
    
    # === Block Counts by Modality ===
    text_block_count: int = 0
    table_block_count: int = 0
    figure_block_count: int = 0
    footnote_block_count: int = 0
    box_block_count: int = 0
    annex_block_count: int = 0
    
    # === Relationship Counts ===
    relationship_count: int = 0
    
    # === Quality Metrics ===
    average_extraction_confidence: float = 1.0
    low_confidence_block_count: int = 0  # Blocks below 0.8 confidence
    
    # === Warnings and Errors ===
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    failed_pages: list[int] = field(default_factory=list)
    
    @property
    def total_block_count(self) -> int:
        """Total blocks across all modalities."""
        return (
            self.text_block_count +
            self.table_block_count +
            self.figure_block_count +
            self.footnote_block_count +
            self.box_block_count +
            self.annex_block_count
        )
