"""
Test index validation checks.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.embedding.validator import IndexValidator, run_validation


def main():
    """Run index validation."""
    
    print("=" * 70)
    print("INDEX VALIDATION")
    print("=" * 70)
    
    # Check for vector index
    chunks_path = Path("outputs/vectordb/chunks.json")
    if not chunks_path.exists():
        print(f"ERROR: Vector index not found at {chunks_path}")
        return
    
    # Run validation
    print("\nRunning validation checks...\n")
    
    validator = IndexValidator(
        chunks_json_path=str(chunks_path),
        expected_chunks=1134,  # From embedding test
        expected_dimension=384
    )
    
    report = validator.run_all_checks()
    
    # Print results
    print(f"Summary: {report.summary()}")
    print("\n" + "-" * 50)
    print("Individual Checks:")
    print("-" * 50)
    
    for i, result in enumerate(report.results, 1):
        print(f"\n{i}. {result.check_name}")
        print(f"   {result.message}")
        
        # Print interesting details
        if result.details:
            if "distribution" in result.details:
                pass  # Already in message
            elif "sections" in result.details:
                pass  # Already in message
            elif "missing_counts" in result.details and any(v > 0 for v in result.details["missing_counts"].values()):
                for field, count in result.details["missing_counts"].items():
                    if count > 0:
                        print(f"      - {field}: {count} missing")
    
    print("\n" + "=" * 70)
    print(f"VALIDATION {'PASSED' if report.is_valid else 'FAILED'}")
    print("=" * 70)
    
    return report.is_valid


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
