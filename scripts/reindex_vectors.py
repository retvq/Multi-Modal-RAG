"""
Re-index Vector Store with Real Embeddings

This script:
1. Loads existing blocks from ingestion output
2. Chunks the blocks
3. Deletes old mock-embedding index
4. Re-embeds all chunks using sentence-transformers
5. Stores new vectors with identical chunk IDs and metadata
"""

import os
import sys
import json
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(Path(__file__).parent))

from src.embedding.pipeline import EmbeddingPipeline
from src.embedding.chunker import Chunker
from src.schemas.content_block import ContentBlock, Modality

# Paths
BLOCKS_INPUT = Path("outputs/ingested/blocks.json")
VECTORDB_DIR = Path("outputs/vectordb")
OLD_INDEX = VECTORDB_DIR / "chunks.json"


def load_blocks_from_json(path: Path) -> list:
    """Load blocks from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
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
    print("=" * 70)
    print("VECTOR STORE RE-INDEXING")
    print("=" * 70)
    
    # Step 1: Check for blocks input
    if not BLOCKS_INPUT.exists():
        print(f"\n[ERROR] Blocks not found at {BLOCKS_INPUT}")
        print("Run ingestion pipeline first.")
        return
    
    print(f"\n[1/5] Loading blocks from {BLOCKS_INPUT}...")
    blocks = load_blocks_from_json(BLOCKS_INPUT)
    print(f"      Loaded {len(blocks)} blocks")
    
    # Step 2: Chunk the blocks
    print(f"\n[2/5] Chunking blocks...")
    chunker = Chunker()
    chunks = chunker.chunk_batch(blocks)
    print(f"      Created {len(chunks)} chunks")
    
    # Step 3: Delete old index
    print(f"\n[3/5] Removing old mock index...")
    if OLD_INDEX.exists():
        os.remove(OLD_INDEX)
        print(f"      Deleted {OLD_INDEX}")
    else:
        print("      No existing index found")
    
    # Also remove ChromaDB files if present
    for f in VECTORDB_DIR.glob("chroma*"):
        print(f"      Removing {f}")
        if f.is_dir():
            import shutil
            shutil.rmtree(f)
        else:
            os.remove(f)
    
    # Step 4: Re-embed with real embeddings
    print(f"\n[4/5] Embedding with sentence-transformers (all-MiniLM-L6-v2)...")
    print("      This may take 1-2 minutes on CPU...")
    
    pipeline = EmbeddingPipeline(
        output_dir=str(VECTORDB_DIR),
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
        use_mock=False  # REAL EMBEDDINGS
    )
    
    result = pipeline.embed_and_store(chunks)
    
    # Step 5: Report results
    print(f"\n[5/5] Re-indexing complete!")
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Total chunks:      {result['total_chunks']}")
    print(f"  Successful:        {result['successful']}")
    print(f"  Failed:            {result['failed']}")
    print(f"  Vectors in index:  {result['vectors_in_index']}")
    print(f"  Embedding model:   {result['model']}")
    print(f"  Dimension:         {result['dimension']}")
    print(f"  Using mock:        {result['using_mock']}")
    print(f"  Duration:          {result['duration_seconds']:.1f}s")
    
    # Verification
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
    # Check dimension
    if result['dimension'] == 384:
        print("  [PASS] Embedding dimension = 384")
    else:
        print(f"  [FAIL] Embedding dimension = {result['dimension']}")
    
    # Check mock status
    if not result['using_mock']:
        print("  [PASS] Using real embeddings (not mock)")
    else:
        print("  [FAIL] Still using mock embeddings")
    
    # Check chunk count
    if result['vectors_in_index'] == len(chunks):
        print(f"  [PASS] All {len(chunks)} chunks indexed")
    else:
        print(f"  [FAIL] Expected {len(chunks)}, got {result['vectors_in_index']}")
    
    # Sample ID verification
    if chunks:
        sample_id = chunks[0].chunk_id
        print(f"  [INFO] Sample chunk ID: {sample_id[:50]}...")


if __name__ == "__main__":
    main()
