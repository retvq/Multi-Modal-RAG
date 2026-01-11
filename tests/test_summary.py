"""Test summary quality with file output."""

import sys
sys.path.insert(0, ".")

# Capture output
import io
from contextlib import redirect_stdout

from demo import RAGDemo

d = RAGDemo()

# Capture
f = io.StringIO()
with redirect_stdout(f):
    d.process_query("Summarize the economic outlook")

output = f.getvalue()

# Write to file
with open("test_output.txt", "w", encoding="utf-8") as out:
    out.write(output)

print("Output written to test_output.txt")
print("\n--- KEY SECTIONS ---")

# Extract key parts
lines = output.split("\n")
in_answer = False
in_citations = False

for line in lines:
    if "ANSWER" in line and "---" not in line:
        in_answer = True
        continue
    if "CITATIONS" in line and "---" not in line:
        in_answer = False
        in_citations = True
        continue
    if "TIMING" in line:
        in_citations = False
        continue
    
    if in_answer and line.strip():
        print(f"ANSWER: {line}")
    if in_citations and line.strip() and not line.startswith("-"):
        print(f"CITATION: {line}")
