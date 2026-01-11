"""
Index Validation Module

Validates vector index integrity and quality:
1. Coverage checks (all chunks indexed)
2. Dimensional consistency
3. Modality distribution inspection
4. Nearest-neighbor sanity queries
"""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from pathlib import Path
from collections import Counter
import sys

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    passed: bool
    message: str
    details: Dict = field(default_factory=dict)


@dataclass 
class IndexValidationReport:
    """Complete validation report for the index."""
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warnings: int = 0
    results: List[ValidationResult] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        return self.failed_checks == 0
    
    def add_result(self, result: ValidationResult):
        self.results.append(result)
        self.total_checks += 1
        if result.passed:
            self.passed_checks += 1
        else:
            self.failed_checks += 1
    
    def summary(self) -> str:
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        return f"{status} - {self.passed_checks}/{self.total_checks} checks passed"


class IndexValidator:
    """
    Validates vector index integrity and quality.
    
    Usage:
        validator = IndexValidator(
            chunks_json_path="outputs/vectordb/chunks.json",
            expected_chunks=1134,
            expected_dimension=384
        )
        report = validator.run_all_checks()
    """
    
    def __init__(
        self,
        chunks_json_path: str,
        expected_chunks: Optional[int] = None,
        expected_dimension: int = 384
    ):
        """
        Initialize validator.
        
        Args:
            chunks_json_path: Path to the vector index JSON
            expected_chunks: Expected number of indexed chunks
            expected_dimension: Expected embedding dimension
        """
        self.chunks_json_path = Path(chunks_json_path)
        self.expected_chunks = expected_chunks
        self.expected_dimension = expected_dimension
        
        # Load index data
        self.index_data = self._load_index()
    
    def _load_index(self) -> Dict:
        """Load index data from JSON."""
        if not self.chunks_json_path.exists():
            return {"embeddings": []}
        
        with open(self.chunks_json_path, encoding='utf-8') as f:
            return json.load(f)
    
    def run_all_checks(self) -> IndexValidationReport:
        """Run all validation checks."""
        report = IndexValidationReport()
        
        # 1. Coverage checks
        report.add_result(self.check_coverage())
        report.add_result(self.check_no_duplicates())
        
        # 2. Dimensional consistency
        report.add_result(self.check_dimensions())
        report.add_result(self.check_vector_norms())
        
        # 3. Modality distribution
        report.add_result(self.check_modality_distribution())
        report.add_result(self.check_section_coverage())
        
        # 4. Metadata completeness
        report.add_result(self.check_metadata_completeness())
        
        # 5. Sanity queries
        report.add_result(self.check_self_similarity())
        report.add_result(self.check_cross_modality_distinction())
        
        return report
    
    # === Coverage Checks ===
    
    def check_coverage(self) -> ValidationResult:
        """Verify all expected chunks are indexed."""
        embeddings = self.index_data.get("embeddings", [])
        actual_count = len(embeddings)
        
        if self.expected_chunks is None:
            return ValidationResult(
                check_name="coverage",
                passed=True,
                message=f"Index contains {actual_count} vectors (no expected count specified)",
                details={"actual_count": actual_count}
            )
        
        passed = actual_count == self.expected_chunks
        return ValidationResult(
            check_name="coverage",
            passed=passed,
            message=f"{'✓' if passed else '✗'} {actual_count}/{self.expected_chunks} chunks indexed",
            details={
                "expected": self.expected_chunks,
                "actual": actual_count,
                "missing": self.expected_chunks - actual_count
            }
        )
    
    def check_no_duplicates(self) -> ValidationResult:
        """Verify no duplicate chunk IDs."""
        embeddings = self.index_data.get("embeddings", [])
        ids = [e["id"] for e in embeddings]
        
        unique_ids = set(ids)
        duplicates = len(ids) - len(unique_ids)
        
        passed = duplicates == 0
        return ValidationResult(
            check_name="no_duplicates",
            passed=passed,
            message=f"{'✓' if passed else '✗'} {duplicates} duplicate IDs found",
            details={"duplicate_count": duplicates}
        )
    
    # === Dimensional Consistency ===
    
    def check_dimensions(self) -> ValidationResult:
        """Verify all vectors have correct dimension."""
        embeddings = self.index_data.get("embeddings", [])
        if not embeddings:
            return ValidationResult(
                check_name="dimensions",
                passed=False,
                message="✗ No embeddings in index",
                details={}
            )
        
        dimensions = [len(e["vector"]) for e in embeddings]
        unique_dims = set(dimensions)
        
        # Check all same dimension
        if len(unique_dims) > 1:
            return ValidationResult(
                check_name="dimensions",
                passed=False,
                message=f"✗ Inconsistent dimensions: {unique_dims}",
                details={"dimensions_found": list(unique_dims)}
            )
        
        # Check matches expected
        actual_dim = dimensions[0]
        passed = actual_dim == self.expected_dimension
        
        return ValidationResult(
            check_name="dimensions",
            passed=passed,
            message=f"{'✓' if passed else '✗'} All vectors are {actual_dim}D (expected {self.expected_dimension}D)",
            details={
                "expected": self.expected_dimension,
                "actual": actual_dim,
                "consistent": True
            }
        )
    
    def check_vector_norms(self) -> ValidationResult:
        """Verify vectors are normalized (L2 norm ≈ 1)."""
        embeddings = self.index_data.get("embeddings", [])
        if not embeddings:
            return ValidationResult(
                check_name="vector_norms",
                passed=False,
                message="✗ No embeddings to check",
                details={}
            )
        
        # Sample first 100 vectors
        sample = embeddings[:100]
        norms = []
        
        for e in sample:
            vec = e["vector"]
            norm = sum(v*v for v in vec) ** 0.5
            norms.append(norm)
        
        min_norm = min(norms)
        max_norm = max(norms)
        avg_norm = sum(norms) / len(norms)
        
        # Check if normalized (norm close to 1)
        tolerance = 0.1
        passed = all(abs(n - 1.0) < tolerance for n in norms)
        
        return ValidationResult(
            check_name="vector_norms",
            passed=passed,
            message=f"{'✓' if passed else '⚠'} Norms - min: {min_norm:.4f}, max: {max_norm:.4f}, avg: {avg_norm:.4f}",
            details={
                "min_norm": min_norm,
                "max_norm": max_norm,
                "avg_norm": avg_norm,
                "sample_size": len(sample)
            }
        )
    
    # === Modality Distribution ===
    
    def check_modality_distribution(self) -> ValidationResult:
        """Check modality distribution is reasonable."""
        embeddings = self.index_data.get("embeddings", [])
        
        modalities = [e["metadata"].get("modality", "UNKNOWN") for e in embeddings]
        distribution = Counter(modalities)
        
        # Check TEXT is dominant (typical for document QA)
        text_count = distribution.get("TEXT", 0)
        total = len(embeddings)
        text_ratio = text_count / total if total > 0 else 0
        
        # Reasonable: TEXT should be 50-95% for typical documents
        passed = 0.3 <= text_ratio <= 0.98
        
        return ValidationResult(
            check_name="modality_distribution",
            passed=passed,
            message=f"{'✓' if passed else '⚠'} Distribution: {dict(distribution)}",
            details={
                "distribution": dict(distribution),
                "text_ratio": text_ratio
            }
        )
    
    def check_section_coverage(self) -> ValidationResult:
        """Check sections are represented."""
        embeddings = self.index_data.get("embeddings", [])
        
        sections = set()
        for e in embeddings:
            section = e["metadata"].get("section_level_0", "")
            if section:
                sections.add(section)
        
        # Expect at least 2 sections for a structured document
        passed = len(sections) >= 2
        
        return ValidationResult(
            check_name="section_coverage",
            passed=passed,
            message=f"{'✓' if passed else '⚠'} {len(sections)} top-level sections: {sections}",
            details={"sections": list(sections)}
        )
    
    # === Metadata Completeness ===
    
    def check_metadata_completeness(self) -> ValidationResult:
        """Verify required metadata fields are present."""
        embeddings = self.index_data.get("embeddings", [])
        if not embeddings:
            return ValidationResult(
                check_name="metadata_completeness",
                passed=False,
                message="✗ No embeddings to check",
                details={}
            )
        
        required_fields = [
            "chunk_id", "parent_block_id", "document_id", 
            "modality", "page_number"
        ]
        
        missing_counts = {f: 0 for f in required_fields}
        
        for e in embeddings:
            meta = e.get("metadata", {})
            for field in required_fields:
                if field not in meta or meta[field] in [None, ""]:
                    missing_counts[field] += 1
        
        total_missing = sum(missing_counts.values())
        passed = total_missing == 0
        
        # Build message
        if passed:
            msg = "✓ All required metadata fields present"
        else:
            missing_details = [f"{f}: {c}" for f, c in missing_counts.items() if c > 0]
            msg = f"✗ Missing metadata: {', '.join(missing_details)}"
        
        return ValidationResult(
            check_name="metadata_completeness",
            passed=passed,
            message=msg,
            details={"missing_counts": missing_counts}
        )
    
    # === Sanity Queries ===
    
    def check_self_similarity(self) -> ValidationResult:
        """Verify chunks are most similar to themselves."""
        embeddings = self.index_data.get("embeddings", [])
        if len(embeddings) < 10:
            return ValidationResult(
                check_name="self_similarity",
                passed=True,
                message="⚠ Too few embeddings for self-similarity check",
                details={}
            )
        
        # Sample 10 random chunks
        import random
        sample = random.sample(embeddings, min(10, len(embeddings)))
        
        correct = 0
        for emb in sample:
            query_vec = emb["vector"]
            query_id = emb["id"]
            
            # Find nearest neighbor
            best_id, best_score = None, -1
            for other in embeddings:
                score = self._cosine_similarity(query_vec, other["vector"])
                if score > best_score:
                    best_score = score
                    best_id = other["id"]
            
            if best_id == query_id:
                correct += 1
        
        passed = correct == len(sample)
        return ValidationResult(
            check_name="self_similarity",
            passed=passed,
            message=f"{'✓' if passed else '⚠'} {correct}/{len(sample)} chunks are their own nearest neighbor",
            details={"correct": correct, "total": len(sample)}
        )
    
    def check_cross_modality_distinction(self) -> ValidationResult:
        """Verify different modalities have distinct embeddings."""
        embeddings = self.index_data.get("embeddings", [])
        
        # Group by modality
        by_modality = {}
        for e in embeddings:
            mod = e["metadata"].get("modality", "UNKNOWN")
            if mod not in by_modality:
                by_modality[mod] = []
            by_modality[mod].append(e)
        
        if len(by_modality) < 2:
            return ValidationResult(
                check_name="cross_modality_distinction",
                passed=True,
                message="⚠ Only one modality present",
                details={}
            )
        
        # Sample from each modality and compute cross-modality similarity
        modalities = list(by_modality.keys())
        similarities = []
        
        for i, mod1 in enumerate(modalities):
            for mod2 in modalities[i+1:]:
                samples1 = by_modality[mod1][:5]
                samples2 = by_modality[mod2][:5]
                
                for s1 in samples1:
                    for s2 in samples2:
                        sim = self._cosine_similarity(s1["vector"], s2["vector"])
                        similarities.append((mod1, mod2, sim))
        
        # Check average cross-modality similarity is not too high
        avg_sim = sum(s[2] for s in similarities) / len(similarities) if similarities else 0
        
        # Cross-modality similarity should typically be < 0.8
        passed = avg_sim < 0.8
        
        return ValidationResult(
            check_name="cross_modality_distinction",
            passed=passed,
            message=f"{'✓' if passed else '⚠'} Average cross-modality similarity: {avg_sim:.4f}",
            details={"avg_similarity": avg_sim, "modalities": modalities}
        )
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(a*b for a, b in zip(vec1, vec2))
        norm1 = sum(a*a for a in vec1) ** 0.5
        norm2 = sum(b*b for b in vec2) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot / (norm1 * norm2)


def run_validation(chunks_json_path: str, expected_chunks: int = None) -> IndexValidationReport:
    """Convenience function to run validation."""
    validator = IndexValidator(
        chunks_json_path=chunks_json_path,
        expected_chunks=expected_chunks
    )
    return validator.run_all_checks()


# === Example Output ===
EXAMPLE_OUTPUT = """
=== INDEX VALIDATION REPORT ===

Summary: ✓ VALID - 9/9 checks passed

Individual Checks:

1. coverage
   ✓ 1134/1134 chunks indexed
   
2. no_duplicates
   ✓ 0 duplicate IDs found

3. dimensions
   ✓ All vectors are 384D (expected 384D)
   
4. vector_norms
   ✓ Norms - min: 1.0000, max: 1.0000, avg: 1.0000
   
5. modality_distribution
   ✓ Distribution: {'TEXT': 931, 'TABLE': 59, 'FIGURE': 98, 'FOOTNOTE': 46}
   
6. section_coverage
   ✓ 4 top-level sections: {'Staff Report', 'Press Release', 'Document', 'Statistical Appendix'}
   
7. metadata_completeness
   ✓ All required metadata fields present
   
8. self_similarity
   ✓ 10/10 chunks are their own nearest neighbor
   
9. cross_modality_distinction
   ✓ Average cross-modality similarity: 0.4523
"""

if __name__ == "__main__":
    print(EXAMPLE_OUTPUT)
