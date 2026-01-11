"""
Retrieval Quality Verification

Tests retrieval with real embeddings on predefined queries.
Reports top 5 chunks per query with modality, page, and score.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval.pipeline import RetrievalPipeline


def main():
    print("=" * 70)
    print("RETRIEVAL QUALITY VERIFICATION")
    print("=" * 70)
    
    # Initialize pipeline
    print("\nInitializing retrieval pipeline...")
    pipeline = RetrievalPipeline(
        vector_store_path="./outputs/vectordb",
        use_mock=False  # Use real embeddings for query
    )
    print(f"  Indexed chunks: {len(pipeline.keyword_index)}")
    
    # Test queries
    queries = [
        ("What is Qatar's GDP growth for 2024?", "Table 1 (Page 39)"),
        ("What is the inflation rate for 2023?", "Table 1 (Page 39)"),
        ("Show me Table 1", "Table 1 (Page 39)"),
    ]
    
    results_summary = []
    
    for query, expected in queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {query}")
        print("=" * 70)
        print(f"Expected: {expected}")
        
        # Run retrieval
        result = pipeline.retrieve(query, top_k=10, method="reranked")
        
        # Report top 5
        print("\nTop 5 Retrieved Chunks:")
        print("-" * 70)
        print(f"{'#':<3} {'Modality':<10} {'Page':<6} {'Score':<8} {'Table/Figure':<15} Preview")
        print("-" * 70)
        
        table1_found = False
        table1_rank = -1
        
        for i, r in enumerate(result.results[:5]):
            mod = f"[{r.modality}]"
            page = r.page_number
            score = f"{r.score:.4f}"
            
            # Check for Table 1
            table_id = r.table_id or ""
            figure_id = r.figure_id or ""
            id_str = table_id or figure_id or "-"
            
            preview = r.content[:40].replace('\n', ' ')
            
            print(f"{i+1:<3} {mod:<10} {page:<6} {score:<8} {id_str:<15} {preview}...")
            
            # Check if this is Table 1 on page 39
            if "Table 1" in table_id and page == 39:
                table1_found = True
                if table1_rank == -1:
                    table1_rank = i + 1
        
        # Check in remaining results for Table 1
        for i, r in enumerate(result.results[5:], start=6):
            table_id = r.table_id or ""
            if "Table 1" in table_id and r.page_number == 39:
                table1_found = True
                if table1_rank == -1:
                    table1_rank = i
        
        # Verdict
        print("\n" + "-" * 70)
        print("VERDICT")
        print("-" * 70)
        
        if "Table 1" in expected:
            if table1_found:
                if table1_rank <= 5:
                    print(f"  [PASS] Table 1 (Page 39) found at rank {table1_rank}")
                    results_summary.append((query, "PASS", f"Table 1 at rank {table1_rank}"))
                else:
                    print(f"  [PARTIAL] Table 1 (Page 39) found at rank {table1_rank} (outside top 5)")
                    results_summary.append((query, "PARTIAL", f"Table 1 at rank {table1_rank}"))
            else:
                print(f"  [FAIL] Table 1 (Page 39) NOT found in top 10")
                results_summary.append((query, "FAIL", "Table 1 not in top 10"))
        else:
            print(f"  [INFO] No specific table expected")
            results_summary.append((query, "INFO", "No table expected"))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for query, status, note in results_summary:
        print(f"\n  [{status}] {query[:40]}...")
        print(f"         {note}")
    
    passes = sum(1 for _, s, _ in results_summary if s == "PASS")
    total = len(results_summary)
    
    print(f"\n  Result: {passes}/{total} PASS")


if __name__ == "__main__":
    main()
