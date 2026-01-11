"""Test underspecified summary guard."""
import sys
sys.path.insert(0, ".")
from demo import RAGDemo

d = RAGDemo()
print("QUERY 1: summary?")
d.process_query("summary?")
print("\nQUERY 2: please summarize")
d.process_query("please summarize")
