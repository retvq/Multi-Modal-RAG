"""
Reranker Test

Tests reranking with before/after comparison.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.retrieval.pipeline import RetrievalPipeline
from src.retrieval.reranker import Reranker, format_before_after


def main():
    """Run reranker test."""
    
    print("=" * 70)
    print("RERANKER TEST")
    print("=" * 70)
    
    # Check for vector index
    vector_path = Path("outputs/vectordb")
    if not (vector_path / "chunks.json").exists():
        print(f"ERROR: Vector index not found at {vector_path}")
        return
    
    # Initialize pipeline
    print("\n[1] Initializing...")
    pipeline = RetrievalPipeline(
        vector_store_path=str(vector_path),
        use_mock=True
    )
    reranker = Reranker()
    
    # Test query
    query = "fiscal policy recommendations"
    print(f"\n[2] Query: \"{query}\"")
    
    # Get candidates from hybrid retrieval (before reranking)
    result = pipeline.retrieve(query, top_k=20, method="hybrid")
    
    # Convert to dict format for reranker
    candidates = []
    for r in result.results:
        candidates.append({
            "chunk_id": r.chunk_id,
            "content": r.content,
            "score": r.score,
            "modality": r.modality,
            "page_number": r.page_number,
            "section_path": r.section_path,
            "parent_block_id": r.parent_block_id,
            "table_id": r.table_id,
            "figure_id": r.figure_id,
            "extraction_confidence": r.extraction_confidence,
            "chunk_index": r.chunk_index,
            "total_chunks": r.total_chunks,
        })
    
    print(f"    Retrieved {len(candidates)} candidates")
    
    # Apply reranking
    print("\n[3] Applying reranking...")
    reranked = reranker.rerank(
        candidates=candidates,
        query=query,
        expected_modality=None,  # No specific modality
        section_hint="Fiscal",   # Boost fiscal section
        top_k=10
    )
    
    # Print before/after comparison
    print("\n" + format_before_after(reranked, top_n=10))
    
    # Detailed breakdown for top result
    print("\n\n" + "=" * 70)
    print("DETAILED BREAKDOWN: Top Result")
    print("=" * 70)
    
    top = reranked[0]
    print(f"\nChunk ID: {top.chunk_id[:50]}...")
    print(f"Modality: {top.modality}")
    print(f"Page: {top.page_number}")
    print(f"Section: {top.section_path}")
    print(f"\nOriginal rank: {top.original_rank} -> Final rank: {top.final_rank}")
    print(f"Original score: {top.original_score:.4f} → Final score: {top.final_score:.4f}")
    
    print("\nScoring Breakdown:")
    b = top.breakdown
    print(f"  + Original score:     {b.original_score:.4f}")
    print(f"  + Term overlap:       {b.term_overlap:.4f}")
    print(f"  + Modality boost:     {b.modality_boost:.4f}")
    print(f"  + ID exact match:     {b.id_exact_match:.4f}")
    print(f"  + Section match:      {b.section_match:.4f}")
    print(f"  + Confidence boost:   {b.confidence_boost:.4f}")
    print(f"  + Completeness boost: {b.completeness_boost:.4f}")
    print(f"  + Length adjustment:  {b.length_adjustment:.4f}")
    print(f"  ──────────────────────────────")
    print(f"  = Final score:        {b.final_score:.4f}")
    
    print("\nContent preview:")
    print(f"  {top.content[:200]}...")


if __name__ == "__main__":
    main()
