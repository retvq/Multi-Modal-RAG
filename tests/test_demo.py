"""
Demo test with single query (non-interactive).
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from demo import RAGDemo


def main():
    vector_path = Path("outputs/vectordb")
    if not (vector_path / "chunks.json").exists():
        print("ERROR: Vector index not found")
        return
    
    demo = RAGDemo(vector_store_path=str(vector_path))
    
    # Test query
    print("\n" + "=" * 70)
    print("TEST QUERY")
    print("=" * 70)
    
    demo.process_query("What is the weather in Doha?")


if __name__ == "__main__":
    main()
