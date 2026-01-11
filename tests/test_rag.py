"""Quick test script for RAG pipeline"""
import warnings
warnings.filterwarnings('ignore')

from src.retrieval.pipeline import RetrievalPipeline
from src.generation.generator import AnswerGenerator

# Initialize
print("Initializing RAG system...")
retrieval = RetrievalPipeline(vector_store_path='./outputs/vectordb', use_mock=False)
generator = AnswerGenerator(use_llm=True)

if generator._llm_available:
    print("LLM: Gemini enabled")
else:
    print("LLM: Not available (using rule-based)")

# Test query
query = "What are the fiscal policy recommendations?"
print(f"\n{'='*60}")
print(f"QUERY: {query}")
print('='*60)

# Retrieve
result = retrieval.retrieve(query, top_k=5, method='reranked')
chunks = [{
    'chunk_id': c.chunk_id,
    'content': c.content,
    'modality': c.modality,
    'page_number': c.page_number,
    'section_path': c.section_path,
    'table_id': c.table_id,
    'figure_id': c.figure_id
} for c in result.results]

print(f"\nRetrieved {len(chunks)} chunks:")
for i, c in enumerate(chunks[:3], 1):
    print(f"  {i}. [{c['modality']}] Page {c['page_number']} - {c['content'][:60]}...")

# Generate
answer = generator.generate(query, chunks)

print(f"\n{'='*60}")
print("ANSWER:")
print('='*60)
print(answer.answer_text)

print(f"\nType: {answer.answer_type.value}")
print(f"Confidence: {answer.confidence.value}")
print(f"Citations: {len(answer.citations)}")

if answer.citations:
    print("\nSources:")
    for cit in answer.citations:
        print(f"  - Page {cit.page_number} ({cit.modality})")
