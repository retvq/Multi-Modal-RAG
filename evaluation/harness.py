"""
Evaluation Harness for Multi-Modal RAG System

Runs benchmark queries across multiple modalities (text, table, figure)
and records comprehensive metrics for each query.

Metrics tracked:
- Retrieval latency
- Generation latency
- Total latency
- Modality accuracy (did we retrieve the expected modality?)
- Keyword match score (how many expected keywords appear in answer?)
- Pass/Fail status

Usage:
    python evaluation/harness.py [--use-llm] [--output PATH]
"""

import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.pipeline import RetrievalPipeline
from src.generation.generator import AnswerGenerator


@dataclass
class QueryResult:
    """Result of a single benchmark query."""
    query_id: str
    query: str
    category: str
    expected_modality: str
    
    # Timing
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float
    
    # Retrieved context
    retrieved_modalities: List[str]
    num_chunks: int
    top_scores: List[float]
    
    # Answer
    answer: str
    citations: List[str]
    
    # Evaluation
    modality_match: bool
    keyword_score: float
    keywords_found: List[str]
    keywords_missing: List[str]
    passed: bool


class EvaluationHarness:
    """
    Runs evaluation benchmarks across multiple modalities.
    
    Features:
    - Loads benchmark queries from JSON
    - Runs each query through the RAG pipeline
    - Records detailed timing and accuracy metrics
    - Generates comprehensive results report
    """
    
    def __init__(
        self,
        vector_store_path: str = "./outputs/vectordb",
        queries_path: str = "./evaluation/test_queries.json",
        use_llm: bool = False,
        collection_name: str = "chunks"
    ):
        self.vector_store_path = vector_store_path
        self.queries_path = queries_path
        self.use_llm = use_llm
        self.collection_name = collection_name
        
        # Initialize pipelines
        self.retrieval = RetrievalPipeline(
            vector_store_path=vector_store_path,
            use_mock=False,
            collection_name=collection_name
        )
        self.generator = AnswerGenerator(use_llm=use_llm)
        
        # Results storage
        self.results: List[QueryResult] = []
    
    def load_queries(self) -> List[Dict[str, Any]]:
        """Load benchmark queries from JSON file."""
        queries_file = Path(self.queries_path)
        if not queries_file.exists():
            raise FileNotFoundError(f"Queries file not found: {queries_file}")
        
        with open(queries_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def evaluate_query(self, query_spec: Dict[str, Any]) -> QueryResult:
        """Run a single benchmark query and record metrics."""
        query_id = query_spec["id"]
        query = query_spec["query"]
        expected_modality = query_spec["expected_modality"]
        expected_keywords = query_spec.get("expected_keywords", [])
        category = query_spec.get("category", "general")
        
        # --- Retrieval ---
        retrieval_start = time.perf_counter()
        retrieval_result = self.retrieval.retrieve(query, top_k=5)
        retrieval_time = (time.perf_counter() - retrieval_start) * 1000
        
        # Extract modalities from retrieved chunks
        retrieved_modalities = []
        top_scores = []
        for chunk in retrieval_result.results:
            modality = chunk.modality.lower()  # Normalize to lowercase
            retrieved_modalities.append(modality)
            top_scores.append(chunk.score)
        
        # --- Generation ---
        # Convert RankedChunk dataclasses to dicts for generator compatibility
        chunks_as_dicts = [asdict(c) for c in retrieval_result.results]
        
        generation_start = time.perf_counter()
        answer_result = self.generator.generate(query, chunks_as_dicts)
        generation_time = (time.perf_counter() - generation_start) * 1000
        
        total_time = retrieval_time + generation_time
        
        # --- Evaluation ---
        # Modality match: check if expected modality appears in retrieved chunks
        if expected_modality == "mixed":
            modality_match = len(set(retrieved_modalities)) > 1
        else:
            modality_match = expected_modality in retrieved_modalities
        
        # Keyword matching
        answer_lower = answer_result.answer_text.lower()
        keywords_found = [kw for kw in expected_keywords if kw.lower() in answer_lower]
        keywords_missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
        keyword_score = len(keywords_found) / len(expected_keywords) if expected_keywords else 1.0
        
        # Pass/Fail: modality match AND at least 50% keyword match
        passed = modality_match and keyword_score >= 0.5
        
        # Build citations list
        citations = [
            f"Page {c.page_number}: {c.modality}"
            for c in retrieval_result.results[:3]
        ]
        
        return QueryResult(
            query_id=query_id,
            query=query,
            category=category,
            expected_modality=expected_modality,
            retrieval_time_ms=round(retrieval_time, 2),
            generation_time_ms=round(generation_time, 2),
            total_time_ms=round(total_time, 2),
            retrieved_modalities=retrieved_modalities,
            num_chunks=len(retrieval_result.results),
            top_scores=[round(s, 4) for s in top_scores[:3]],
            answer=answer_result.answer_text[:500],  # Truncate for storage
            citations=citations,
            modality_match=modality_match,
            keyword_score=round(keyword_score, 3),
            keywords_found=keywords_found,
            keywords_missing=keywords_missing,
            passed=passed
        )
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run all benchmark queries and compile results."""
        queries = self.load_queries()
        self.results = []
        
        print(f"\n{'='*60}")
        print(f"  MULTI-MODAL RAG EVALUATION SUITE")
        print(f"  Running {len(queries)} benchmark queries...")
        print(f"{'='*60}\n")
        
        for i, query_spec in enumerate(queries, 1):
            print(f"[{i}/{len(queries)}] {query_spec['id']}: {query_spec['query'][:50]}...")
            
            try:
                result = self.evaluate_query(query_spec)
                self.results.append(result)
                
                status = "[PASS]" if result.passed else "[FAIL]"
                print(f"         {status} | {result.total_time_ms:.0f}ms | Keywords: {result.keyword_score:.0%}")
                
            except Exception as e:
                print(f"         [ERROR] {e}")
                # Create failed result
                self.results.append(QueryResult(
                    query_id=query_spec["id"],
                    query=query_spec["query"],
                    category=query_spec.get("category", "general"),
                    expected_modality=query_spec["expected_modality"],
                    retrieval_time_ms=0,
                    generation_time_ms=0,
                    total_time_ms=0,
                    retrieved_modalities=[],
                    num_chunks=0,
                    top_scores=[],
                    answer=f"ERROR: {e}",
                    citations=[],
                    modality_match=False,
                    keyword_score=0.0,
                    keywords_found=[],
                    keywords_missing=query_spec.get("expected_keywords", []),
                    passed=False
                ))
        
        return self.compile_report()
    
    def compile_report(self) -> Dict[str, Any]:
        """Compile evaluation results into a comprehensive report."""
        if not self.results:
            return {"error": "No results to compile"}
        
        # Overall metrics
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        
        avg_retrieval = sum(r.retrieval_time_ms for r in self.results) / total
        avg_generation = sum(r.generation_time_ms for r in self.results) / total
        avg_total = sum(r.total_time_ms for r in self.results) / total
        avg_keyword_score = sum(r.keyword_score for r in self.results) / total
        
        # Per-modality breakdown
        modality_stats = {}
        for modality in ["text", "table", "figure", "mixed"]:
            modality_results = [r for r in self.results if r.expected_modality == modality]
            if modality_results:
                modality_stats[modality] = {
                    "total": len(modality_results),
                    "passed": sum(1 for r in modality_results if r.passed),
                    "pass_rate": sum(1 for r in modality_results if r.passed) / len(modality_results),
                    "avg_latency_ms": sum(r.total_time_ms for r in modality_results) / len(modality_results),
                    "avg_keyword_score": sum(r.keyword_score for r in modality_results) / len(modality_results)
                }
        
        # Per-category breakdown
        category_stats = {}
        categories = set(r.category for r in self.results)
        for category in categories:
            cat_results = [r for r in self.results if r.category == category]
            if cat_results:
                category_stats[category] = {
                    "total": len(cat_results),
                    "passed": sum(1 for r in cat_results if r.passed),
                    "pass_rate": sum(1 for r in cat_results if r.passed) / len(cat_results)
                }
        
        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_queries": total,
                "use_llm": self.use_llm,
                "vector_store": self.vector_store_path
            },
            "summary": {
                "passed": passed,
                "failed": failed,
                "pass_rate": passed / total,
                "avg_retrieval_ms": round(avg_retrieval, 2),
                "avg_generation_ms": round(avg_generation, 2),
                "avg_total_ms": round(avg_total, 2),
                "avg_keyword_score": round(avg_keyword_score, 3)
            },
            "modality_breakdown": modality_stats,
            "category_breakdown": category_stats,
            "results": [asdict(r) for r in self.results]
        }
    
    def save_results(self, output_path: Optional[str] = None) -> str:
        """Save results to JSON file."""
        if output_path is None:
            logs_dir = Path("./evaluation/logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = logs_dir / f"results_{timestamp}.json"
        
        report = self.compile_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        return str(output_path)
    
    def print_summary(self):
        """Print a summary of evaluation results."""
        report = self.compile_report()
        summary = report["summary"]
        
        print(f"\n{'='*60}")
        print(f"  EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"  Total Queries: {report['metadata']['total_queries']}")
        print(f"  Passed: {summary['passed']} ({summary['pass_rate']:.0%})")
        print(f"  Failed: {summary['failed']}")
        print(f"\n  Avg Latency:")
        print(f"    Retrieval: {summary['avg_retrieval_ms']:.0f}ms")
        print(f"    Generation: {summary['avg_generation_ms']:.0f}ms")
        print(f"    Total: {summary['avg_total_ms']:.0f}ms")
        print(f"\n  Avg Keyword Score: {summary['avg_keyword_score']:.0%}")
        
        print(f"\n  Modality Breakdown:")
        for modality, stats in report["modality_breakdown"].items():
            print(f"    {modality.capitalize()}: {stats['passed']}/{stats['total']} ({stats['pass_rate']:.0%})")
        
        print(f"{'='*60}\n")


def main():
    """Run evaluation from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG Evaluation Suite")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for generation")
    parser.add_argument("--output", type=str, help="Output path for results JSON")
    parser.add_argument("--queries", type=str, default="./evaluation/test_queries.json",
                        help="Path to benchmark queries JSON")
    parser.add_argument("--vector-store", type=str, default="./outputs/vectordb",
                        help="Path to vector store")
    
    args = parser.parse_args()
    
    harness = EvaluationHarness(
        vector_store_path=args.vector_store,
        queries_path=args.queries,
        use_llm=args.use_llm
    )
    
    harness.run_evaluation()
    output_path = harness.save_results(args.output)
    harness.print_summary()
    
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
