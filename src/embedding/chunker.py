"""
Chunk Schema and Chunker

Converts ContentBlocks into retrieval-optimized Chunks.

Implements Phase 4 chunking rules:
- Modality-aware boundaries
- Section-aware splitting
- Metadata preservation
- Context prefix enrichment
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from pathlib import Path
from enum import Enum
import hashlib
import sys

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from src.schemas.content_block import ContentBlock, Modality


@dataclass
class Chunk:
    """
    A retrieval-optimized unit derived from a ContentBlock.
    
    Each chunk is sized for embedding and contains full provenance.
    """
    # Identity
    chunk_id: str
    parent_block_id: str
    document_id: str
    
    # Position within parent
    chunk_index: int = 0
    total_chunks: int = 1
    is_complete: bool = True
    
    # Modality
    modality: Modality = Modality.TEXT
    sub_type: Optional[str] = None  # For figures: LINE_CHART, BAR_CHART, etc.
    
    # Content
    content: str = ""                   # Original content
    embedding_input: str = ""          # Enriched text for embedding
    content_length: int = 0
    
    # Provenance
    page_number: int = 0
    page_range: tuple = field(default_factory=lambda: (0, 0))
    section_hierarchy: List[str] = field(default_factory=list)
    section_path: str = ""              # Flattened: "Staff Report > Fiscal"
    
    # Modality-specific metadata
    table_id: Optional[str] = None
    figure_id: Optional[str] = None
    footnote_marker: Optional[str] = None
    panel_titles: List[str] = field(default_factory=list)
    
    # Quality
    extraction_confidence: float = 1.0
    
    def __post_init__(self):
        self.content_length = len(self.content)
        self.page_range = (self.page_number, self.page_number)
        if self.section_hierarchy:
            self.section_path = self._build_section_path()
    
    def _build_section_path(self) -> str:
        """Build flattened section path (max 3 levels after Document)."""
        parts = []
        for i, section in enumerate(self.section_hierarchy):
            if i >= 3:
                break
            # Remove A., B., 1., 2. prefixes
            clean = re.sub(r'^[A-Z]\.\s*|^\d+\.\s*', '', section)
            if clean and clean != "Document":
                parts.append(clean)
        return " > ".join(parts[:3])


class Chunker:
    """
    Converts ContentBlocks into Chunks.
    
    Applies modality-specific strategies:
    - TEXT: Split by paragraph/sentence boundaries
    - TABLE: Keep intact, split rows only if exceeds limit
    - FIGURE: Single chunk (title + caption)
    - FOOTNOTE: Single chunk (atomic)
    """
    
    def __init__(
        self,
        max_tokens: int = 512,
        min_tokens: int = 50,
        target_tokens: int = 400,
        chars_per_token: float = 4.0  # Rough estimate
    ):
        """
        Initialize chunker.
        
        Args:
            max_tokens: Maximum tokens per chunk (embedding limit)
            min_tokens: Minimum tokens (avoid fragments)
            target_tokens: Preferred chunk size
            chars_per_token: Approximate characters per token
        """
        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.target_tokens = target_tokens
        self.chars_per_token = chars_per_token
        
        self.max_chars = int(max_tokens * chars_per_token)
        self.min_chars = int(min_tokens * chars_per_token)
        self.target_chars = int(target_tokens * chars_per_token)
    
    def chunk(self, block: ContentBlock) -> List[Chunk]:
        """
        Convert a ContentBlock into one or more Chunks.
        """
        if block.modality == Modality.TEXT:
            return self._chunk_text(block)
        elif block.modality == Modality.TABLE:
            return self._chunk_table(block)
        elif block.modality == Modality.FIGURE:
            return self._chunk_figure(block)
        elif block.modality == Modality.FOOTNOTE:
            return self._chunk_footnote(block)
        else:
            # Default: treat as text
            return self._chunk_text(block)
    
    def chunk_batch(self, blocks: List[ContentBlock]) -> List[Chunk]:
        """Chunk multiple blocks."""
        chunks = []
        for block in blocks:
            chunks.extend(self.chunk(block))
        return chunks
    
    def _chunk_text(self, block: ContentBlock) -> List[Chunk]:
        """Chunk TEXT content by paragraph/sentence boundaries."""
        content = block.content
        
        if len(content) <= self.max_chars:
            # Fits in one chunk
            return [self._create_chunk(block, content, 0, 1)]
        
        # Split by paragraphs first
        paragraphs = content.split('\n\n')
        chunks = []
        current_text = ""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_text) + len(para) + 2 <= self.max_chars:
                if current_text:
                    current_text += "\n\n" + para
                else:
                    current_text = para
            else:
                # Save current chunk
                if current_text:
                    chunks.append(current_text)
                
                # Handle paragraph exceeding limit
                if len(para) > self.max_chars:
                    # Split by sentences
                    sentences = self._split_sentences(para)
                    para_chunks = self._merge_sentences(sentences)
                    chunks.extend(para_chunks)
                else:
                    current_text = para
        
        if current_text:
            chunks.append(current_text)
        
        # Filter small chunks
        chunks = [c for c in chunks if len(c) >= self.min_chars]
        
        # Create Chunk objects
        total = len(chunks)
        return [
            self._create_chunk(block, text, i, total)
            for i, text in enumerate(chunks)
        ]
    
    def _chunk_table(self, block: ContentBlock) -> List[Chunk]:
        """Chunk TABLE content, preserving structure."""
        content = block.content
        
        # Get table metadata
        table_id = None
        table_title = ""
        if hasattr(block, '_table_metadata') and block._table_metadata:
            table_id = block._table_metadata.table_id
            table_title = block._table_metadata.table_title
        
        if len(content) <= self.max_chars:
            # Fits in one chunk
            chunk = self._create_chunk(block, content, 0, 1)
            chunk.table_id = table_id
            return [chunk]
        
        # Need to split by rows, keeping header
        lines = content.split('\n')
        chunks = []
        
        # Find header rows (first rows, typically before data)
        header_lines = []
        data_start = 0
        
        for i, line in enumerate(lines):
            if line.startswith('|') and '---' not in line:
                # Check if this is header or data
                if i == 0 or (i == 1 and '---' in lines[1] if len(lines) > 1 else False):
                    header_lines.append(line)
                    data_start = i + 1
                else:
                    break
            elif '---' in line:
                header_lines.append(line)
                data_start = i + 1
        
        header_text = '\n'.join(header_lines)
        header_len = len(header_text) + len(table_title) + 10
        
        # Split data rows into chunks
        current_rows = []
        current_len = header_len
        
        for line in lines[data_start:]:
            line_len = len(line) + 1
            if current_len + line_len <= self.max_chars:
                current_rows.append(line)
                current_len += line_len
            else:
                if current_rows:
                    chunk_text = header_text + '\n' + '\n'.join(current_rows)
                    if table_title and len(chunks) > 0:
                        chunk_text = f"{table_title} (continued)\n\n{chunk_text}"
                    elif table_title:
                        chunk_text = f"{table_title}\n\n{chunk_text}"
                    chunks.append(chunk_text)
                current_rows = [line]
                current_len = header_len + line_len
        
        if current_rows:
            chunk_text = header_text + '\n' + '\n'.join(current_rows)
            if table_title:
                prefix = " (continued)" if len(chunks) > 0 else ""
                chunk_text = f"{table_title}{prefix}\n\n{chunk_text}"
            chunks.append(chunk_text)
        
        total = len(chunks)
        result = []
        for i, text in enumerate(chunks):
            chunk = self._create_chunk(block, text, i, total)
            chunk.table_id = table_id
            chunk.is_complete = (total == 1)
            result.append(chunk)
        
        return result
    
    def _chunk_figure(self, block: ContentBlock) -> List[Chunk]:
        """Chunk FIGURE content (single chunk)."""
        # Get figure metadata
        figure_id = None
        panel_titles = []
        if hasattr(block, '_figure_metadata') and block._figure_metadata:
            figure_id = block._figure_metadata.figure_id
            if block._figure_metadata.panels:
                panel_titles = [p.panel_title for p in block._figure_metadata.panels if p.panel_title]
        
        chunk = self._create_chunk(block, block.content, 0, 1)
        chunk.figure_id = figure_id
        chunk.panel_titles = panel_titles
        chunk.sub_type = block._figure_metadata.figure_type if hasattr(block, '_figure_metadata') and block._figure_metadata else None
        
        return [chunk]
    
    def _chunk_footnote(self, block: ContentBlock) -> List[Chunk]:
        """Chunk FOOTNOTE content (single chunk)."""
        marker = getattr(block, '_footnote_marker', None)
        
        chunk = self._create_chunk(block, block.content, 0, 1)
        chunk.footnote_marker = marker
        
        return [chunk]
    
    def _create_chunk(
        self, 
        block: ContentBlock, 
        content: str, 
        index: int, 
        total: int
    ) -> Chunk:
        """Create a Chunk from block and content."""
        # Generate chunk ID
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        chunk_id = f"{block.block_id}_chunk_{index}_{content_hash}"
        
        # Build embedding input
        embedding_input = self._build_embedding_input(block, content)
        
        return Chunk(
            chunk_id=chunk_id,
            parent_block_id=block.block_id,
            document_id=block.document_id,
            chunk_index=index,
            total_chunks=total,
            is_complete=(total == 1),
            modality=block.modality,
            content=content,
            embedding_input=embedding_input,
            page_number=block.page_number,
            section_hierarchy=block.section_hierarchy.copy() if block.section_hierarchy else [],
            extraction_confidence=block.extraction_confidence,
        )
    
    def _build_embedding_input(self, block: ContentBlock, content: str) -> str:
        """
        Construct embedding input with context enrichment.
        
        Format: [MODALITY] Section Path\n\nContent
        """
        parts = []
        
        # Modality tag
        modality_tag = f"[{block.modality.value}]"
        
        # Section path
        section_path = ""
        if block.section_hierarchy:
            path_parts = []
            for i, section in enumerate(block.section_hierarchy[:3]):
                clean = re.sub(r'^[A-Z]\.\s*|^\d+\.\s*', '', section)
                if clean and clean != "Document":
                    path_parts.append(clean)
            section_path = " > ".join(path_parts)
        
        # Build header
        header = modality_tag
        if section_path:
            header += f" {section_path}"
        
        parts.append(header)
        parts.append("")  # Blank line
        parts.append(content)
        
        return "\n".join(parts)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _merge_sentences(self, sentences: List[str]) -> List[str]:
        """Merge sentences into chunks respecting max size."""
        chunks = []
        current = ""
        
        for sentence in sentences:
            if len(current) + len(sentence) + 1 <= self.max_chars:
                if current:
                    current += " " + sentence
                else:
                    current = sentence
            else:
                if current:
                    chunks.append(current)
                # Handle sentence exceeding limit
                if len(sentence) > self.max_chars:
                    # Force split at max_chars
                    for i in range(0, len(sentence), self.max_chars):
                        chunks.append(sentence[i:i+self.max_chars])
                    current = ""
                else:
                    current = sentence
        
        if current:
            chunks.append(current)
        
        return chunks
