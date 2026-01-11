"""
Embedding Sanity Validation

Tests that real embeddings produce meaningful similarity scores:
- Similar content should have higher similarity
- Unrelated content should have lower similarity
- No NaN, zero-norm, or identical vectors
"""

import json
import math
from pathlib import Path


def load_vectors(path: Path) -> list:
    """Load vectors from JSON store."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("embeddings", [])


def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    
    if norm_a == 0 or norm_b == 0:
        return float('nan')
    
    return dot / (norm_a * norm_b)


def check_vector_health(vec: list, chunk_id: str) -> dict:
    """Check vector for NaN, zero-norm, etc."""
    norm = math.sqrt(sum(x * x for x in vec))
    has_nan = any(math.isnan(x) for x in vec)
    is_zero = norm == 0
    is_unit = abs(norm - 1.0) < 0.01
    
    return {
        "chunk_id": chunk_id[:50],
        "dimension": len(vec),
        "norm": norm,
        "has_nan": has_nan,
        "is_zero": is_zero,
        "is_unit": is_unit,
    }


def main():
    print("=" * 70)
    print("EMBEDDING SANITY VALIDATION")
    print("=" * 70)
    
    # Load vectors
    vector_path = Path("outputs/vectordb/chunks.json")
    if not vector_path.exists():
        print(f"\n[ERROR] Vector store not found at {vector_path}")
        return
    
    vectors = load_vectors(vector_path)
    print(f"\nLoaded {len(vectors)} vectors")
    
    # Find chunks by modality
    text_chunks = [v for v in vectors if v["metadata"].get("modality") == "TEXT"]
    table_chunks = [v for v in vectors if v["metadata"].get("modality") == "TABLE"]
    
    print(f"  TEXT chunks: {len(text_chunks)}")
    print(f"  TABLE chunks: {len(table_chunks)}")
    
    # Select test chunks
    # 1. TEXT chunk about GDP/growth (fiscal content)
    gdp_chunk = None
    for v in text_chunks:
        if "gdp" in v["document"].lower() or "growth" in v["document"].lower():
            gdp_chunk = v
            break
    
    # 2. TABLE chunk (preferably with GDP data)
    table_chunk = None
    for v in table_chunks:
        if "gdp" in v["document"].lower() or "macroeconomic" in v["document"].lower():
            table_chunk = v
            break
    if not table_chunk and table_chunks:
        table_chunk = table_chunks[0]
    
    # 3. Unrelated TEXT chunk - MUST be different from gdp_chunk
    # Look for content that is NOT about economics - e.g., dates, headers, TOC
    unrelated_chunk = None
    for v in text_chunks:
        # Skip if same as GDP chunk
        if gdp_chunk and v["id"] == gdp_chunk["id"]:
            continue
        
        doc_lower = v["document"].lower()
        # Find structural/metadata content that's not about economy
        if len(v["document"]) < 100:  # Short, likely header/date
            if "international monetary fund" in doc_lower or "january" in doc_lower or "contents" in doc_lower:
                unrelated_chunk = v
                break
    
    # Second pass: find climate-specific content
    if not unrelated_chunk:
        for v in text_chunks:
            if gdp_chunk and v["id"] == gdp_chunk["id"]:
                continue
            doc_lower = v["document"].lower()
            if "climate" in doc_lower and "gdp" not in doc_lower and "growth" not in doc_lower:
                unrelated_chunk = v
                break
    
    # Fallback: use last TEXT chunk that's different from gdp_chunk
    if not unrelated_chunk:
        for v in reversed(text_chunks):
            if gdp_chunk and v["id"] != gdp_chunk["id"]:
                unrelated_chunk = v
                break
    
    if not all([gdp_chunk, table_chunk, unrelated_chunk]):
        print("\n[ERROR] Could not find required test chunks")
        return
    
    # Print test chunks
    print("\n" + "-" * 70)
    print("TEST CHUNKS")
    print("-" * 70)
    
    print(f"\n1. GDP TEXT chunk:")
    print(f"   ID: {gdp_chunk['id'][:60]}...")
    print(f"   Page: {gdp_chunk['metadata'].get('page_number')}")
    print(f"   Preview: {gdp_chunk['document'][:80]}...")
    
    print(f"\n2. TABLE chunk:")
    print(f"   ID: {table_chunk['id'][:60]}...")
    print(f"   Page: {table_chunk['metadata'].get('page_number')}")
    print(f"   Preview: {table_chunk['document'][:80]}...")
    
    print(f"\n3. UNRELATED TEXT chunk:")
    print(f"   ID: {unrelated_chunk['id'][:60]}...")
    print(f"   Page: {unrelated_chunk['metadata'].get('page_number')}")
    print(f"   Preview: {unrelated_chunk['document'][:80]}...")
    
    # Check vector health
    print("\n" + "-" * 70)
    print("VECTOR HEALTH")
    print("-" * 70)
    
    for chunk, name in [(gdp_chunk, "GDP TEXT"), (table_chunk, "TABLE"), (unrelated_chunk, "UNRELATED")]:
        health = check_vector_health(chunk["vector"], chunk["id"])
        status = "HEALTHY" if health["is_unit"] and not health["has_nan"] else "ISSUES"
        print(f"\n  {name}:")
        print(f"    Dimension: {health['dimension']}")
        print(f"    Norm: {health['norm']:.6f}")
        print(f"    Has NaN: {health['has_nan']}")
        print(f"    Is unit vector: {health['is_unit']}")
        print(f"    Status: {status}")
    
    # Compute similarities
    print("\n" + "-" * 70)
    print("SIMILARITY SCORES")
    print("-" * 70)
    
    sim_gdp_table = cosine_similarity(gdp_chunk["vector"], table_chunk["vector"])
    sim_gdp_unrelated = cosine_similarity(gdp_chunk["vector"], unrelated_chunk["vector"])
    sim_table_unrelated = cosine_similarity(table_chunk["vector"], unrelated_chunk["vector"])
    sim_gdp_gdp = cosine_similarity(gdp_chunk["vector"], gdp_chunk["vector"])
    
    print(f"\n  GDP TEXT <-> TABLE (related):      {sim_gdp_table:.4f}")
    print(f"  GDP TEXT <-> UNRELATED:            {sim_gdp_unrelated:.4f}")
    print(f"  TABLE <-> UNRELATED:               {sim_table_unrelated:.4f}")
    print(f"  GDP TEXT <-> GDP TEXT (self):      {sim_gdp_gdp:.4f}")
    
    # Verdict
    print("\n" + "-" * 70)
    print("VERDICT")
    print("-" * 70)
    
    checks = []
    
    # Check 1: Similar content higher than unrelated
    if sim_gdp_table > sim_gdp_unrelated:
        checks.append(("[PASS] GDP-TABLE similarity > GDP-UNRELATED similarity", True))
    else:
        checks.append(("[FAIL] GDP-TABLE similarity should be > GDP-UNRELATED", False))
    
    # Check 2: Self-similarity is 1.0
    if abs(sim_gdp_gdp - 1.0) < 0.001:
        checks.append(("[PASS] Self-similarity = 1.0", True))
    else:
        checks.append(("[FAIL] Self-similarity should be 1.0", False))
    
    # Check 3: No NaN
    has_nan = any(math.isnan(s) for s in [sim_gdp_table, sim_gdp_unrelated, sim_table_unrelated])
    if not has_nan:
        checks.append(("[PASS] No NaN similarity scores", True))
    else:
        checks.append(("[FAIL] Found NaN similarity scores", False))
    
    # Check 4: Vectors are not identical (except self)
    if sim_gdp_table < 0.999 and sim_gdp_unrelated < 0.999:
        checks.append(("[PASS] Vectors are not identical", True))
    else:
        checks.append(("[FAIL] Vectors appear identical", False))
    
    # Check 5: Reasonable similarity range
    if 0 < sim_gdp_unrelated < 1 and 0 < sim_gdp_table < 1:
        checks.append(("[PASS] Similarities in valid range (0, 1)", True))
    else:
        checks.append(("[FAIL] Similarities outside valid range", False))
    
    for msg, passed in checks:
        print(f"\n  {msg}")
    
    all_passed = all(p for _, p in checks)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("FINAL VERDICT: PASS")
    else:
        print("FINAL VERDICT: FAIL")
    print("=" * 70)


if __name__ == "__main__":
    main()
