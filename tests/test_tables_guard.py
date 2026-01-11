"""Test 'all tables' guard."""
import sys
sys.path.insert(0, ".")
from demo import RAGDemo

d = RAGDemo()
print("QUERY: show me all tables")
d.process_query("show me all tables")
