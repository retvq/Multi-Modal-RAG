"""
Answer Generation Test

Tests the complete answer generation pipeline.
"""

import sys
import json
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.retrieval.pipeline import RetrievalPipeline
from src.generation.generator import AnswerGenerator


def main():
    """Run answer generation test."""
    
    print("=" * 70)
    print("ANSWER GENERATION TEST")
    print("=" * 70)
    
    # Check for vector index
    vector_path = Path("outputs/vectordb")
    if not (vector_path / "chunks.json").exists():
        print(f"ERROR: Vector index not found at {vector_path}")
        return
    
    # Initialize pipelines
    print("\n[1] Initializing pipelines...")
    retrieval = RetrievalPipeline(
        vector_store_path=str(vector_path),
        use_mock=True
    )
    generator = AnswerGenerator(use_llm=False)
    
    print(f"    Retrieval: {len(retrieval.keyword_index)} chunks indexed")
    print(f"    Generator: rule-based mode")
    
    # Test queries
    test_queries = [
        "What is Qatar's GDP growth projection for 2024?",
        "Show me Table 1 data",
        "What are the fiscal policy recommendations?",
        "What is the population of Mars?",  # Should fail
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n\n{'='*70}")
        print(f"[{i+1}] Query: \"{query}\"")
        print("=" * 70)
        
        # Retrieve
        retrieval_result = retrieval.retrieve(query, top_k=10, method="reranked")
        
        # Convert to dict format for generator
        chunks = []
        for r in retrieval_result.results:
            chunks.append({
                "chunk_id": r.chunk_id,
                "content": r.content,
                "modality": r.modality,
                "page_number": r.page_number,
                "section_path": r.section_path,
                "table_id": r.table_id,
                "figure_id": r.figure_id,
            })
        
        # Generate
        answer = generator.generate(query, chunks)
        
        # Print result
        print(f"\nAnswer Type: {answer.answer_type.value}")
        print(f"Confidence: {answer.confidence.value}")
        print(f"Sources Used: {answer.sources_used}/{answer.sources_available}")
        print(f"Generation Time: {answer.generation_time_ms:.1f}ms")
        
        print("\nAnswer Text:")
        print("-" * 50)
        # Wrap long lines
        text = answer.answer_text
        if len(text) > 500:
            text = text[:500] + "..."
        print(text)
        
        print("\nCitations:")
        print("-" * 50)
        for c in answer.citations[:3]:
            print(f"  {c.citation_id} [{c.modality}] Page {c.page_number}")
            if c.table_id:
                print(f"      Table: {c.table_id}")
            print(f"      Section: {c.section_path}")
    
    # Summary
    print("\n\n" + "=" * 70)
    print("ANSWER GENERATION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
