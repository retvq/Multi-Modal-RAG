"""
Embedding Pipeline End-to-End Test

Tests the complete flow:
1. Load ingested blocks from Phase 3
2. Chunk blocks using Phase 4 rules
3. Generate embeddings
4. Store in vector index
5. Verify retrieval
"""

import json
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.schemas.content_block import ContentBlock, Modality
from src.embedding.chunker import Chunker
from src.embedding.pipeline import EmbeddingPipeline


def load_blocks_from_json(json_path: str) -> list:
    """Load ContentBlocks from ingested JSON."""
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    
    blocks = []
    for item in data:
        block = ContentBlock(
            block_id=item["block_id"],
            document_id=item["document_id"],
            modality=Modality(item["modality"]),
            content=item["content"],
            page_number=item["page_number"],
            section_hierarchy=item.get("section_hierarchy", []),
            extraction_confidence=item.get("extraction_confidence", 1.0),
        )
        blocks.append(block)
    
    return blocks


def main():
    """Run embedding pipeline test."""
    
    print("=" * 70)
    print("EMBEDDING PIPELINE TEST")
    print("=" * 70)
    
    # Check for ingested blocks
    blocks_path = Path("outputs/ingested/blocks.json")
    if not blocks_path.exists():
        print(f"ERROR: No ingested blocks found at {blocks_path}")
        print("Run the ingestion pipeline first.")
        return
    
    # Step 1: Load blocks
    print("\n[1] Loading ingested blocks...")
    blocks = load_blocks_from_json(str(blocks_path))
    print(f"    Loaded {len(blocks)} ContentBlocks")
    
    # Count by modality
    modality_counts = {}
    for b in blocks:
        mod = b.modality.value
        modality_counts[mod] = modality_counts.get(mod, 0) + 1
    for mod, count in sorted(modality_counts.items()):
        print(f"    - {mod}: {count}")
    
    # Step 2: Chunk blocks
    print("\n[2] Chunking blocks...")
    chunker = Chunker(max_tokens=512, target_tokens=400)
    chunks = chunker.chunk_batch(blocks)
    print(f"    Created {len(chunks)} chunks from {len(blocks)} blocks")
    
    # Chunk statistics
    chunk_modalities = {}
    for c in chunks:
        mod = c.modality.value
        chunk_modalities[mod] = chunk_modalities.get(mod, 0) + 1
    for mod, count in sorted(chunk_modalities.items()):
        print(f"    - {mod}: {count}")
    
    # Example chunk
    if chunks:
        example = chunks[0]
        print(f"\n    Example chunk:")
        print(f"    chunk_id: {example.chunk_id[:50]}...")
        print(f"    modality: {example.modality.value}")
        print(f"    section_path: {example.section_path}")
        print(f"    content_length: {example.content_length}")
        print(f"    embedding_input preview:")
        print(f"    {example.embedding_input[:150]}...")
    
    # Step 3: Generate embeddings and store
    print("\n[3] Embedding and storing...")
    pipeline = EmbeddingPipeline(
        output_dir="./outputs/vectordb",
        use_mock=True  # Use mock for testing without model
    )
    
    results = pipeline.embed_and_store(chunks)
    
    print(f"    Model: {results['model']}")
    print(f"    Dimension: {results['dimension']}")
    print(f"    Using mock: {results['using_mock']}")
    print(f"    Successful: {results['successful']}")
    print(f"    Failed: {results['failed']}")
    print(f"    Duration: {results['duration_seconds']:.2f}s")
    print(f"    Vectors in index: {results['vectors_in_index']}")
    
    # Step 4: Test retrieval
    print("\n[4] Testing retrieval...")
    
    # Create a test query
    test_query = "What is Qatar's GDP growth projection?"
    query_embedding = pipeline.model.encode([test_query])[0]
    
    # Query the store
    search_results = pipeline.store.query(
        query_embedding=query_embedding,
        n_results=5
    )
    
    print(f"    Query: '{test_query}'")
    print(f"    Top 5 results:")
    
    for i, (chunk_id, distance, metadata, doc) in enumerate(zip(
        search_results["ids"][0],
        search_results["distances"][0],
        search_results["metadatas"][0],
        search_results["documents"][0]
    )):
        print(f"\n    {i+1}. {metadata['modality']} | Page {metadata['page_number']}")
        print(f"       Score: {1-distance:.4f}")
        print(f"       Section: {metadata['section_path']}")
        print(f"       Content: {doc[:80]}...")
    
    # Step 5: Test filtered retrieval
    print("\n\n[5] Testing filtered retrieval (TABLE only)...")
    
    filtered_results = pipeline.store.query(
        query_embedding=query_embedding,
        n_results=3,
        where={"modality": "TABLE"}
    )
    
    if filtered_results["ids"][0]:
        print(f"    Found {len(filtered_results['ids'][0])} TABLE results")
        for i, metadata in enumerate(filtered_results["metadatas"][0]):
            print(f"    {i+1}. {metadata.get('table_id', 'N/A')} | Page {metadata['page_number']}")
    else:
        print("    No TABLE results found")
    
    # Summary
    print("\n\n" + "=" * 70)
    print("EMBEDDING PIPELINE TEST COMPLETE")
    print("=" * 70)
    print(f"\n    Blocks ingested: {len(blocks)}")
    print(f"    Chunks created: {len(chunks)}")
    print(f"    Vectors stored: {results['vectors_in_index']}")
    print(f"    Failures: {results['failures']}")
    
    print("\n    Output files:")
    print(f"    - outputs/vectordb/chunks.json")
    if results['failures'] > 0:
        print(f"    - outputs/vectordb/embedding_failures.json")


if __name__ == "__main__":
    main()
