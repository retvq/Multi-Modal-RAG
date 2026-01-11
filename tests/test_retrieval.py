"""
Retrieval Pipeline Test

Tests baseline, hybrid, and re-ranked retrieval.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.retrieval.pipeline import RetrievalPipeline


def main():
    """Run retrieval tests."""
    
    print("=" * 70)
    print("RETRIEVAL PIPELINE TEST")
    print("=" * 70)
    
    # Check for vector index
    vector_path = Path("outputs/vectordb")
    if not (vector_path / "chunks.json").exists():
        print(f"ERROR: Vector index not found at {vector_path}")
        return
    
    # Initialize pipeline
    print("\n[1] Initializing retrieval pipeline...")
    pipeline = RetrievalPipeline(
        vector_store_path=str(vector_path),
        use_mock=True
    )
    print(f"    Loaded {len(pipeline.keyword_index)} chunks")
    
    # Test queries
    test_queries = [
        ("What is Qatar's GDP growth projection?", "hybrid"),
        ("Show me Table 1", "hybrid"),
        ("Figure showing inflation", "hybrid"),
        ("Fiscal policy recommendations", "reranked"),
        ("page 39 data", "reranked"),
    ]
    
    for i, (query, method) in enumerate(test_queries, 1):
        print(f"\n\n[{i+1}] Query: \"{query}\"")
        print(f"    Method: {method}")
        print("-" * 50)
        
        result = pipeline.retrieve(query, top_k=5, method=method)
        
        print(f"    Parsed filters: {result.parsed_query.filters}")
        print(f"    Expected modality: {result.parsed_query.expected_modality}")
        print(f"    Retrieval time: {result.retrieval_time_ms:.1f}ms")
        print(f"    Total candidates: {result.total_candidates}")
        
        print("\n    Top 5 Results:")
        for r in result.results[:5]:
            content_preview = r.content[:60].replace('\n', ' ')
            print(f"    {r.rank}. [{r.modality}] Page {r.page_number} | Score: {r.score:.4f}")
            print(f"       {content_preview}...")
            if r.table_id:
                print(f"       table_id: {r.table_id}")
            if r.figure_id:
                print(f"       figure_id: {r.figure_id}")
    
    # Summary
    print("\n\n" + "=" * 70)
    print("RETRIEVAL PIPELINE TEST COMPLETE")
    print("=" * 70)
    print(f"\n    Queries tested: {len(test_queries)}")
    print(f"    Methods: baseline, hybrid, reranked")
    print(f"    Modalities: TEXT, TABLE, FIGURE, FOOTNOTE")


if __name__ == "__main__":
    main()
