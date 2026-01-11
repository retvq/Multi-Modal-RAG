"""Test underspecified summary guard variations."""
import sys
sys.path.insert(0, ".")
from demo import RAGDemo

d = RAGDemo()
print("QUERY 1: what is summary")
d.process_query("what is summary")
print("\nQUERY 2: give me a full summary")
d.process_query("give me a full summary")
print("\nQUERY 3: summarize the document please")
d.process_query("summarize the document please")
