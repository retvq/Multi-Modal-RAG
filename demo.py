"""
Multi-Modal RAG Demo CLI

A transparency-first evaluation interface for the RAG system.

Features:
- Query input
- Retrieved context display with modality tags and scores
- Answer display with citations
- Clear separation between retrieval and generation
- Timing breakdown
"""

import sys
import time
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.retrieval.pipeline import RetrievalPipeline
from src.generation.generator import AnswerGenerator, classify_intent


class RAGDemo:
    """Interactive RAG demonstration interface."""
    
    def __init__(self, vector_store_path: str = "./outputs/vectordb", use_llm: bool = True):
        """Initialize the demo."""
        print("=" * 70)
        print("MULTI-MODAL RAG DEMO")
        print("=" * 70)
        print("\nInitializing...")
        
        # Initialize retrieval pipeline with real embeddings
        self.retrieval = RetrievalPipeline(
            vector_store_path=vector_store_path,
            use_mock=False  # Use real embeddings
        )
        
        # Initialize generator with LLM support
        self.use_llm = use_llm
        self.generator = AnswerGenerator(use_llm=use_llm)
        
        # Check LLM availability
        if use_llm:
            if self.generator._llm_available:
                print(f"  LLM: Gemini (enabled)")
            else:
                print(f"  LLM: Not available (using rule-based fallback)")
        else:
            print(f"  LLM: Disabled (rule-based mode)")
        
        # Index stats
        self.chunk_count = len(self.retrieval.keyword_index)
        modality_counts = {}
        for chunk_id, data in self.retrieval.keyword_index.items():
            mod = data["metadata"].get("modality", "TEXT")
            modality_counts[mod] = modality_counts.get(mod, 0) + 1
        
        print(f"\nDocument: Qatar Article IV Consultation 2024")
        print(f"Status: Indexed")
        print(f"\nIndex Statistics:")
        print(f"  Total Chunks: {self.chunk_count}")
        for mod, count in sorted(modality_counts.items()):
            pct = count / self.chunk_count * 100
            print(f"  {mod}: {count} ({pct:.0f}%)")

    
    def run(self):
        """Run interactive demo loop."""
        print("\n" + "=" * 70)
        print("Enter queries below. Type 'quit' to exit.")
        print("=" * 70)
        
        while True:
            print("\n")
            query = input("Query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nExiting demo.")
                break
            
            if not query:
                continue
            
            self.process_query(query)
    
    def process_query(self, query: str):
        """Process a single query and display results."""
        
        # =====================================================
        # SECTION 0: INTENT CLASSIFICATION
        # =====================================================
        print("\n" + "-" * 70)
        print("INTENT CLASSIFICATION")
        print("-" * 70)
        
        intent = classify_intent(query)
        print(f"  Query: \"{query}\"")
        print(f"  Intent: {intent.value.upper()}")
        
        # =====================================================
        # SECTION 1: PATTERN EXTRACTION
        # =====================================================
        print("\n" + "-" * 70)
        print("PATTERN EXTRACTION (regex-based)")
        print("-" * 70)
        
        parsed = self.retrieval.query_parser.parse(query)
        print(f"  Query tokens: {query.split()}")
        
        # Show pattern matches
        if parsed.table_id:
            print(f"  'Table X' pattern: matched -> {parsed.table_id}")
        else:
            print(f"  'Table X' pattern: not found")
        
        if parsed.figure_id:
            print(f"  'Figure X' pattern: matched -> {parsed.figure_id}")
        else:
            print(f"  'Figure X' pattern: not found")
        
        if parsed.page_number:
            print(f"  'page N' pattern: matched -> page {parsed.page_number}")
        else:
            print(f"  'page N' pattern: not found")
        
        if parsed.section_hint:
            print(f"  Section keyword: '{parsed.section_hint}' matched")
        else:
            print(f"  Section keyword: none matched")
        
        print(f"  Filters applied: {parsed.filters if parsed.filters else 'None'}")
        
        # =====================================================
        # SECTION 2: RETRIEVAL
        # =====================================================
        print("\n" + "-" * 70)
        print("RETRIEVAL")
        print("-" * 70)
        
        start_retrieval = time.time()
        retrieval_result = self.retrieval.retrieve(query, top_k=10, method="reranked")
        retrieval_time = (time.time() - start_retrieval) * 1000
        
        print(f"  Method: hybrid + rerank")
        print(f"  Time: {retrieval_time:.0f}ms")
        print(f"  Results: {len(retrieval_result.results)} chunks")
        
        # Modality distribution
        mod_dist = {}
        for r in retrieval_result.results:
            mod_dist[r.modality] = mod_dist.get(r.modality, 0) + 1
        print(f"  Modalities: {mod_dist}")
        
        # Score range
        if retrieval_result.results:
            scores = [r.score for r in retrieval_result.results]
            print(f"  Score range: {max(scores):.3f} - {min(scores):.3f}")
        
        # Display retrieved chunks
        print("\n  Retrieved Chunks:")
        print("  " + "-" * 66)
        
        for r in retrieval_result.results[:10]:
            mod_tag = f"[{r.modality}]".ljust(10)
            page_tag = f"Page {r.page_number}".ljust(8)
            score_tag = f"{r.score:.3f}"
            
            # Truncate content preview
            preview = r.content[:50].replace('\n', ' ')
            
            print(f"  #{r.rank:2} {mod_tag} {page_tag} {score_tag}  {preview}...")
            
            if r.table_id:
                print(f"       Table: {r.table_id}")
            if r.figure_id:
                print(f"       Figure: {r.figure_id}")
        
        # =====================================================
        # SECTION 3: GENERATION
        # =====================================================
        print("\n" + "-" * 70)
        print("GENERATION")
        print("-" * 70)
        
        # Convert to dict format
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
        
        start_gen = time.time()
        answer = self.generator.generate(query, chunks)
        gen_time = (time.time() - start_gen) * 1000
        
        print(f"  Time: {gen_time:.0f}ms")
        print(f"  Type: {answer.answer_type.value}")
        print(f"  Confidence: {answer.confidence.value}")
        print(f"  Sources used: {answer.sources_used}/{answer.sources_available}")
        
        # =====================================================
        # SECTION 4: ANSWER
        # =====================================================
        print("\n" + "-" * 70)
        print("ANSWER")
        print("-" * 70)
        
        # Display answer
        answer_lines = answer.answer_text.split('\n')
        for line in answer_lines[:15]:  # Limit display
            print(f"  {line}")
        
        if len(answer_lines) > 15:
            print(f"  ... ({len(answer_lines) - 15} more lines)")
        
        # =====================================================
        # SECTION 5: CITATIONS
        # =====================================================
        if answer.citations:
            print("\n" + "-" * 70)
            print("CITATIONS")
            print("-" * 70)
            
            for c in answer.citations:
                mod_tag = f"[{c.modality}]"
                print(f"  {c.citation_id} {mod_tag} Page {c.page_number}")
                if c.table_id:
                    print(f"       {c.table_id}")
                if c.section_path:
                    print(f"       Section: {c.section_path}")
                excerpt = c.quoted_text[:60].replace('\n', ' ')
                print(f"       \"{excerpt}...\"")
        
        # =====================================================
        # SECTION 6: TIMING SUMMARY
        # =====================================================
        print("\n" + "-" * 70)
        print("TIMING")
        print("-" * 70)
        total_time = retrieval_time + gen_time
        print(f"  Retrieval: {retrieval_time:.0f}ms")
        print(f"  Generation: {gen_time:.0f}ms")
        print(f"  Total: {total_time:.0f}ms")


def main():
    """Run the demo."""
    # Check for vector index
    vector_path = Path("outputs/vectordb")
    if not (vector_path / "chunks.json").exists():
        print(f"ERROR: Vector index not found at {vector_path}")
        print("Run the embedding pipeline first.")
        return
    
    demo = RAGDemo(vector_store_path=str(vector_path))
    demo.run()


if __name__ == "__main__":
    main()
