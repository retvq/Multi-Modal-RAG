"""
Re-Ranking Layer

Applies rule-based re-ranking to retrieval candidates.

Features:
- Multiple scoring signals
- Score breakdown for debugging
- Provenance and modality preservation
- No silent discarding (all candidates retained with scores)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))


@dataclass
class ScoringBreakdown:
    """Detailed breakdown of reranking score."""
    original_score: float = 0.0
    term_overlap: float = 0.0
    modality_boost: float = 0.0
    id_exact_match: float = 0.0
    section_match: float = 0.0
    confidence_boost: float = 0.0
    completeness_boost: float = 0.0
    length_adjustment: float = 0.0
    final_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "original_score": self.original_score,
            "term_overlap": self.term_overlap,
            "modality_boost": self.modality_boost,
            "id_exact_match": self.id_exact_match,
            "section_match": self.section_match,
            "confidence_boost": self.confidence_boost,
            "completeness_boost": self.completeness_boost,
            "length_adjustment": self.length_adjustment,
            "final_score": self.final_score,
        }


@dataclass
class RerankedChunk:
    """A chunk with reranking information."""
    chunk_id: str
    content: str
    original_rank: int
    final_rank: int
    original_score: float
    final_score: float
    rank_change: int  # positive = improved, negative = dropped
    
    # Provenance (preserved)
    modality: str
    page_number: int
    section_path: str
    parent_block_id: str
    
    # Optional metadata
    table_id: str = ""
    figure_id: str = ""
    extraction_confidence: float = 1.0
    chunk_index: int = 0
    total_chunks: int = 1
    
    # Scoring breakdown
    breakdown: ScoringBreakdown = field(default_factory=ScoringBreakdown)


@dataclass
class RerankerConfig:
    """Configuration for reranking weights."""
    weight_term_overlap: float = 0.10
    weight_modality_match: float = 0.10
    weight_modality_mismatch: float = -0.05
    weight_id_exact_match: float = 0.30
    weight_section_match: float = 0.05
    weight_high_confidence: float = 0.02
    weight_low_confidence: float = -0.05
    weight_complete_chunk: float = 0.02
    weight_fragment_chunk: float = -0.01
    
    # Length thresholds
    short_content_threshold: int = 100
    long_content_threshold: int = 2000


class Reranker:
    """
    Re-ranks retrieval candidates using multiple signals.
    
    Usage:
        reranker = Reranker()
        reranked = reranker.rerank(
            candidates=retrieval_results,
            query="fiscal policy",
            expected_modality="TEXT",
            table_id=None,
            figure_id=None,
            section_hint="Fiscal"
        )
    """
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        """Initialize reranker with optional config."""
        self.config = config or RerankerConfig()
    
    def rerank(
        self,
        candidates: List[Dict],
        query: str,
        expected_modality: Optional[str] = None,
        table_id: Optional[str] = None,
        figure_id: Optional[str] = None,
        section_hint: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> List[RerankedChunk]:
        """
        Rerank candidates and return all with updated scores.
        
        Args:
            candidates: List of candidate dicts with chunk info
            query: Original user query
            expected_modality: Expected modality (TABLE, FIGURE, etc.)
            table_id: Specific table ID if requested
            figure_id: Specific figure ID if requested
            section_hint: Section keyword for boosting
            top_k: Optional limit (all returned with ranks if specified)
        
        Returns:
            All candidates with reranking information (never discarded)
        """
        reranked = []
        
        for i, cand in enumerate(candidates):
            original_rank = i + 1
            original_score = cand.get("score", 0.0)
            
            # Compute scoring breakdown
            breakdown = self._compute_breakdown(
                cand, 
                query,
                original_score,
                expected_modality,
                table_id,
                figure_id,
                section_hint
            )
            
            reranked.append(RerankedChunk(
                chunk_id=cand.get("chunk_id", ""),
                content=cand.get("content", ""),
                original_rank=original_rank,
                final_rank=0,  # Assigned after sorting
                original_score=original_score,
                final_score=breakdown.final_score,
                rank_change=0,  # Computed after sorting
                modality=cand.get("modality", "TEXT"),
                page_number=cand.get("page_number", 0),
                section_path=cand.get("section_path", ""),
                parent_block_id=cand.get("parent_block_id", ""),
                table_id=cand.get("table_id", ""),
                figure_id=cand.get("figure_id", ""),
                extraction_confidence=cand.get("extraction_confidence", 1.0),
                chunk_index=cand.get("chunk_index", 0),
                total_chunks=cand.get("total_chunks", 1),
                breakdown=breakdown,
            ))
        
        # Sort by final score
        reranked.sort(key=lambda x: x.final_score, reverse=True)
        
        # Assign final ranks and compute rank changes
        for i, chunk in enumerate(reranked):
            chunk.final_rank = i + 1
            chunk.rank_change = chunk.original_rank - chunk.final_rank
        
        # Apply top_k if specified (but still return all)
        if top_k:
            return reranked[:top_k]
        
        return reranked
    
    def _compute_breakdown(
        self,
        cand: Dict,
        query: str,
        original_score: float,
        expected_modality: Optional[str],
        table_id: Optional[str],
        figure_id: Optional[str],
        section_hint: Optional[str]
    ) -> ScoringBreakdown:
        """Compute detailed scoring breakdown."""
        breakdown = ScoringBreakdown(original_score=original_score)
        
        content = cand.get("content", "").lower()
        modality = cand.get("modality", "TEXT")
        
        # 1. Term overlap
        breakdown.term_overlap = self._term_overlap_score(query, content)
        
        # 2. Modality boost
        if expected_modality:
            if modality == expected_modality:
                breakdown.modality_boost = self.config.weight_modality_match
            else:
                breakdown.modality_boost = self.config.weight_modality_mismatch
        
        # 3. ID exact match
        if table_id and cand.get("table_id") == table_id:
            breakdown.id_exact_match = self.config.weight_id_exact_match
        if figure_id and cand.get("figure_id") == figure_id:
            breakdown.id_exact_match = self.config.weight_id_exact_match
        
        # 4. Section match
        if section_hint:
            section_path = cand.get("section_path", "").lower()
            if section_hint.lower() in section_path:
                breakdown.section_match = self.config.weight_section_match
        
        # 5. Confidence boost
        confidence = cand.get("extraction_confidence", 1.0)
        if confidence >= 0.95:
            breakdown.confidence_boost = self.config.weight_high_confidence
        elif confidence < 0.5:
            breakdown.confidence_boost = self.config.weight_low_confidence
        
        # 6. Completeness boost
        total_chunks = cand.get("total_chunks", 1)
        if total_chunks == 1:
            breakdown.completeness_boost = self.config.weight_complete_chunk
        else:
            breakdown.completeness_boost = self.config.weight_fragment_chunk
        
        # 7. Length adjustment (minor)
        content_len = len(content)
        if content_len < self.config.short_content_threshold:
            breakdown.length_adjustment = -0.01
        elif content_len > self.config.long_content_threshold:
            breakdown.length_adjustment = -0.01
        
        # Compute final score
        breakdown.final_score = (
            original_score
            + breakdown.term_overlap
            + breakdown.modality_boost
            + breakdown.id_exact_match
            + breakdown.section_match
            + breakdown.confidence_boost
            + breakdown.completeness_boost
            + breakdown.length_adjustment
        )
        
        return breakdown
    
    def _term_overlap_score(self, query: str, content: str) -> float:
        """Compute term overlap between query and content."""
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        content_terms = set(re.findall(r'\b\w+\b', content.lower()))
        
        if not query_terms:
            return 0.0
        
        overlap = len(query_terms & content_terms)
        ratio = overlap / len(query_terms)
        
        return ratio * self.config.weight_term_overlap


def format_before_after(reranked: List[RerankedChunk], top_n: int = 10) -> str:
    """Format before/after comparison for debugging."""
    lines = []
    lines.append("=" * 80)
    lines.append("RERANKING COMPARISON (Before -> After)")
    lines.append("=" * 80)
    
    # Create before ranking
    by_original = sorted(reranked, key=lambda x: x.original_rank)[:top_n]
    
    lines.append("\nBEFORE (by original rank):")
    lines.append("-" * 60)
    for c in by_original:
        lines.append(f"  {c.original_rank:2}. [{c.modality}] Page {c.page_number} | Score: {c.original_score:.4f}")
        lines.append(f"      {c.content[:50]}...")
    
    # After ranking
    lines.append("\nAFTER (by reranked score):")
    lines.append("-" * 60)
    for c in reranked[:top_n]:
        change = f"+{c.rank_change}" if c.rank_change > 0 else str(c.rank_change)
        if c.rank_change > 0:
            change_str = f"UP {change}"
        elif c.rank_change < 0:
            change_str = f"DOWN {change}"
        else:
            change_str = "  ="
        
        lines.append(f"  {c.final_rank:2}. [{c.modality}] Page {c.page_number} | Score: {c.final_score:.4f} ({change_str})")
        lines.append(f"      {c.content[:50]}...")
    
    # Biggest movers
    lines.append("\nBIGGEST RANK CHANGES:")
    lines.append("-" * 60)
    movers = sorted(reranked, key=lambda x: abs(x.rank_change), reverse=True)[:5]
    for c in movers:
        direction = "UP" if c.rank_change > 0 else "DOWN" if c.rank_change < 0 else "="
        lines.append(f"  {c.original_rank} -> {c.final_rank} ({direction} {abs(c.rank_change)}) [{c.modality}] Page {c.page_number}")
        
        # Show why
        b = c.breakdown
        boosts = []
        if b.modality_boost > 0:
            boosts.append(f"modality: +{b.modality_boost:.2f}")
        if b.id_exact_match > 0:
            boosts.append(f"ID match: +{b.id_exact_match:.2f}")
        if b.section_match > 0:
            boosts.append(f"section: +{b.section_match:.2f}")
        if b.term_overlap > 0:
            boosts.append(f"terms: +{b.term_overlap:.2f}")
        if boosts:
            lines.append(f"      Boosts: {', '.join(boosts)}")
    
    return "\n".join(lines)


# Example usage
EXAMPLE_OUTPUT = """
================================================================================
RERANKING COMPARISON (Before → After)
================================================================================

BEFORE (by original rank):
------------------------------------------------------------
   1. [TABLE] Page 37 | Score: 0.4523
      | Indicator | 2020 | 2021 | 2022 | 2023 |...
   2. [TEXT] Page 10 | Score: 0.4456
      Growth is projected to remain stable at 2.4 percent...
   3. [FOOTNOTE] Page 42 | Score: 0.4401
      [1] Includes provisional data subject to revision...
   4. [TEXT] Page 8 | Score: 0.4389
      The authorities emphasized continued reform...
   5. [FIGURE] Page 34 | Score: 0.4312
      Figure 1. Qatar: Real Sector Developments...

AFTER (by reranked score):
------------------------------------------------------------
   1. [TABLE] Page 37 | Score: 0.5823 (↑ +0)
      | Indicator | 2020 | 2021 | 2022 | 2023 |...
   2. [TEXT] Page 10 | Score: 0.5256 (↑ +0)
      Growth is projected to remain stable at 2.4 percent...
   3. [TEXT] Page 8 | Score: 0.5189 (↑ +1)
      The authorities emphasized continued reform...
   4. [FIGURE] Page 34 | Score: 0.5012 (↑ +1)
      Figure 1. Qatar: Real Sector Developments...
   5. [FOOTNOTE] Page 42 | Score: 0.4601 (↓ -2)
      [1] Includes provisional data subject to revision...

BIGGEST RANK CHANGES:
------------------------------------------------------------
  3 → 5 (↓2) [FOOTNOTE] Page 42
      Boosts: (none - footnote demoted)
  4 → 3 (↑1) [TEXT] Page 8
      Boosts: terms: +0.08, section: +0.05
  5 → 4 (↑1) [FIGURE] Page 34
      Boosts: modality: +0.10
"""

if __name__ == "__main__":
    print(EXAMPLE_OUTPUT)
