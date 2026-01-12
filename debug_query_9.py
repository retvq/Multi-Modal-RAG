import time
from src.retrieval.pipeline import RetrievalPipeline
from src.generation.generator import AnswerGenerator, classify_intent
from src.generation.llm_client import GeminiClient

def debug_query_9():
    query = "Compare GDP projections mentioned in text with those in Table 1"
    print(f"Query: {query}")
    
    # Check intent
    intent = classify_intent(query)
    print(f"Classified Intent: {intent}")
    
    # Retrieve
    print("Retrieving...")
    retrieval = RetrievalPipeline(vector_store_path="./outputs/vectordb", collection_name="chunks")
    retrieval_result = retrieval.retrieve(query, top_k=5)
    print(f"Retrieved {len(retrieval_result.results)} chunks")
    
    from dataclasses import asdict
    chunks = [asdict(c) for c in retrieval_result.results]
    for i, c in enumerate(chunks):
        print(f"Chunk {i}: {c.get('modality')} - {c.get('chunk_id')}")
        
    # Generate
    print("Generating...")
    start_time = time.time()
    
    # Initialize generator manually to inspect
    client = GeminiClient(model_name="gemini-3-flash-preview")
    print(f"Using model: {client.model_name}")
    
    # Call generate_answer manually to isolate LLM
    try:
        response = client.generate_answer(query, chunks)
        print("Generation Complete!")
        print(f"Answer: {response.answer_text[:100]}...")
        print(f"Time: {time.time() - start_time:.2f}s")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_query_9()
