"""
Embedding Pipeline

Generates embeddings for chunks and stores them in a vector index.

Features:
- Deterministic embedding (same input -> same output)
- Batch processing for efficiency
- Fallback to mock embeddings if model unavailable
- Error logging for failures
"""

import json
import hashlib
import time
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Callable
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from src.embedding.chunker import Chunk


# Try to import embedding libraries
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


@dataclass
class EmbeddingResult:
    """Result of embedding a single chunk."""
    chunk_id: str
    success: bool
    vector: Optional[List[float]] = None
    error: Optional[str] = None
    embedding_time_ms: float = 0


@dataclass
class EmbeddingFailure:
    """Record of a failed embedding."""
    chunk_id: str
    reason: str
    input_preview: str
    extraction_confidence: float


class EmbeddingModel:
    """
    Wrapper for embedding model with fallback support.
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_mock: bool = False
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: Sentence-transformer model name
            use_mock: Force mock embeddings (for testing)
        """
        self.model_name = model_name
        self.use_mock = use_mock
        self.dimension = 384  # Default for MiniLM
        self._model = None
        
        if not use_mock and HAS_SENTENCE_TRANSFORMERS:
            try:
                self._model = SentenceTransformer(model_name)
                self.dimension = self._model.get_sentence_embedding_dimension()
            except Exception as e:
                print(f"[EmbeddingModel] Failed to load {model_name}: {e}")
                print("[EmbeddingModel] Using mock embeddings")
                self.use_mock = True
        else:
            self.use_mock = True
    
    def encode(self, texts: List[str]) -> List[List[float]]:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        
        Note:
            Truncation is handled by the model at the token level,
            not by character slicing. This ensures:
            - Proper tokenization before truncation
            - No mid-word or mid-token cuts
            - Consistent behavior across text lengths
        """
        if self.use_mock:
            return self._mock_encode(texts)
        
        # Real embeddings with normalization for cosine similarity
        # 
        # normalize_embeddings=True ensures all vectors have unit length (L2 norm = 1)
        # This is critical for cosine similarity because:
        # - cosine_sim(a, b) = dot(a, b) / (||a|| * ||b||)
        # - When ||a|| = ||b|| = 1, cosine_sim reduces to dot(a, b)
        # - Pre-normalization prevents scale drift and NaN from zero-magnitude vectors
        #
        # Mock embeddings already normalize in _mock_encode(), so this only affects real
        embeddings = self._model.encode(
            texts, 
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalization for cosine stability
        )
        return [emb.tolist() for emb in embeddings]
    
    def _mock_encode(self, texts: List[str]) -> List[List[float]]:
        """
        Generate deterministic mock embeddings for testing.
        
        Uses hash of text to generate consistent pseudo-random vectors.
        
        Note:
            - Empty/whitespace text uses placeholder to avoid zero vectors
            - All vectors are normalized (non-zero magnitude)
            - Zero vectors are NEVER returned
        """
        embeddings = []
        for text in texts:
            # Handle empty/whitespace text - use placeholder to avoid zero vectors
            effective_text = text.strip()
            if not effective_text:
                effective_text = "__EMPTY_CHUNK_PLACEHOLDER__"
            
            # Generate deterministic hash
            text_hash = hashlib.sha256(effective_text.encode()).digest()
            
            # Convert to floats in [-1, 1] range
            vector = []
            for i in range(self.dimension):
                byte_idx = i % len(text_hash)
                value = (text_hash[byte_idx] / 127.5) - 1.0
                vector.append(value)
            
            # Normalize - guaranteed non-zero since hash produces varied bytes
            norm = sum(v*v for v in vector) ** 0.5
            if norm > 0:
                vector = [v / norm for v in vector]
            else:
                # Fallback: should never happen, but ensure non-zero
                vector = [1.0 / (self.dimension ** 0.5)] * self.dimension
            
            embeddings.append(vector)
        
        return embeddings


class VectorStore:
    """
    Vector storage with metadata support.
    
    Uses ChromaDB if available, otherwise JSON file storage.
    """
    
    def __init__(
        self,
        collection_name: str = "chunks",
        persist_directory: str = "./outputs/vectordb"
    ):
        """
        Initialize vector store.
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self._client = None
        self._collection = None
        self._use_json = False
        
        if HAS_CHROMADB:
            try:
                self._client = chromadb.PersistentClient(
                    path=str(self.persist_directory)
                )
                self._collection = self._client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e:
                print(f"[VectorStore] ChromaDB failed: {e}")
                self._use_json = True
        else:
            self._use_json = True
        
        # JSON fallback storage
        self._json_path = self.persist_directory / f"{collection_name}.json"
        self._json_data = {"embeddings": [], "metadata": {}}
        if self._use_json and self._json_path.exists():
            with open(self._json_path) as f:
                self._json_data = json.load(f)
    
    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        documents: List[str]
    ):
        """
        Add vectors to the store.
        """
        if self._use_json:
            for i, chunk_id in enumerate(ids):
                self._json_data["embeddings"].append({
                    "id": chunk_id,
                    "vector": embeddings[i],
                    "metadata": metadatas[i],
                    "document": documents[i]
                })
            with open(self._json_path, "w") as f:
                json.dump(self._json_data, f)
        else:
            self._collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
    
    def count(self) -> int:
        """Get number of stored vectors."""
        if self._use_json:
            return len(self._json_data["embeddings"])
        return self._collection.count()
    
    def query(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        where: Optional[Dict] = None
    ) -> Dict:
        """
        Query the store.
        """
        if self._use_json:
            # Simple cosine similarity for JSON fallback
            results = []
            for entry in self._json_data["embeddings"]:
                # Apply filters
                if where:
                    match = all(
                        entry["metadata"].get(k) == v 
                        for k, v in where.items()
                    )
                    if not match:
                        continue
                
                # Compute cosine similarity
                vec = entry["vector"]
                dot = sum(a*b for a, b in zip(query_embedding, vec))
                results.append((dot, entry))
            
            # Sort by similarity
            results.sort(key=lambda x: x[0], reverse=True)
            results = results[:n_results]
            
            return {
                "ids": [[r[1]["id"] for r in results]],
                "distances": [[1 - r[0] for r in results]],
                "metadatas": [[r[1]["metadata"] for r in results]],
                "documents": [[r[1]["document"] for r in results]]
            }
        else:
            return self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where
            )


class EmbeddingPipeline:
    """
    Main pipeline for embedding chunks and storing in vector index.
    
    Usage:
        pipeline = EmbeddingPipeline(output_dir="./outputs/vectordb")
        results = pipeline.embed_and_store(chunks)
    """
    
    def __init__(
        self,
        output_dir: str = "./outputs/vectordb",
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        use_mock: bool = False,
        collection_name: str = "chunks"
    ):
        """
        Initialize embedding pipeline.
        
        Args:
            output_dir: Directory for vector store
            model_name: Sentence-transformer model name
            batch_size: Batch size for embedding
            use_mock: Use mock embeddings for testing
            collection_name: ChromaDB collection name (for namespace separation)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = EmbeddingModel(model_name=model_name, use_mock=use_mock)
        self.store = VectorStore(
            collection_name=collection_name,
            persist_directory=str(self.output_dir)
        )
        
        self.batch_size = batch_size
        self.failures: List[EmbeddingFailure] = []
    
    def embed_and_store(self, chunks: List[Chunk]) -> Dict:
        """
        Embed all chunks and store in vector index.
        
        Returns:
            Summary statistics
        """
        start_time = time.time()
        
        total_chunks = len(chunks)
        successful = 0
        failed = 0
        
        # Process in batches
        for i in range(0, total_chunks, self.batch_size):
            batch = chunks[i:i+self.batch_size]
            
            # Check for low-confidence inputs
            for chunk in batch:
                if chunk.extraction_confidence < 0.5:
                    self._log_failure(chunk, "Low extraction confidence")
            
            # Prepare inputs
            ids = [c.chunk_id for c in batch]
            texts = [c.embedding_input for c in batch]
            metadatas = [self._build_metadata(c) for c in batch]
            documents = [c.content for c in batch]
            
            try:
                # Generate embeddings
                embeddings = self.model.encode(texts)
                
                # Store
                self.store.add(
                    ids=ids,
                    embeddings=embeddings,
                    metadatas=metadatas,
                    documents=documents
                )
                
                successful += len(batch)
                
            except Exception as e:
                # Log failures
                for chunk in batch:
                    self._log_failure(chunk, str(e))
                failed += len(batch)
        
        duration = time.time() - start_time
        
        # Write failures log
        self._write_failures_log()
        
        return {
            "total_chunks": total_chunks,
            "successful": successful,
            "failed": failed,
            "failures": len(self.failures),
            "duration_seconds": duration,
            "vectors_in_index": self.store.count(),
            "model": self.model.model_name,
            "dimension": self.model.dimension,
            "using_mock": self.model.use_mock
        }
    
    def _build_metadata(self, chunk: Chunk) -> Dict:
        """Build metadata dict for storage."""
        return {
            "chunk_id": chunk.chunk_id,
            "parent_block_id": chunk.parent_block_id,
            "document_id": chunk.document_id,
            "modality": chunk.modality.value,
            "page_number": chunk.page_number,
            "section_path": chunk.section_path,
            "section_level_0": chunk.section_hierarchy[0] if chunk.section_hierarchy else "",
            "section_level_1": chunk.section_hierarchy[1] if len(chunk.section_hierarchy) > 1 else "",
            "table_id": chunk.table_id or "",
            "figure_id": chunk.figure_id or "",
            "chunk_index": chunk.chunk_index,
            "total_chunks": chunk.total_chunks,
            "extraction_confidence": chunk.extraction_confidence,
        }
    
    def _log_failure(self, chunk: Chunk, reason: str):
        """Log embedding failure."""
        self.failures.append(EmbeddingFailure(
            chunk_id=chunk.chunk_id,
            reason=reason,
            input_preview=chunk.embedding_input[:100],
            extraction_confidence=chunk.extraction_confidence
        ))
    
    def _write_failures_log(self):
        """Write failures to log file."""
        if self.failures:
            log_path = self.output_dir / "embedding_failures.json"
            with open(log_path, "w") as f:
                json.dump([asdict(f) for f in self.failures], f, indent=2)


# Example indexed record
EXAMPLE_INDEXED_RECORD = """
Example indexed record in vector store:

{
  "id": "qatar_p039_table_9ee09b77_000_chunk_0_a1b2c3d4",
  "vector": [0.0234, -0.0891, 0.1456, ...],  // 384 dimensions
  "metadata": {
    "chunk_id": "qatar_p039_table_9ee09b77_000_chunk_0_a1b2c3d4",
    "parent_block_id": "qatar_p039_table_9ee09b77_000",
    "document_id": "qatar_test_doc",
    "modality": "TABLE",
    "page_number": 39,
    "section_path": "Staff Report > Statistical Appendix",
    "section_level_0": "Staff Report",
    "section_level_1": "Statistical Appendix",
    "table_id": "Table 1",
    "figure_id": "",
    "chunk_index": 0,
    "total_chunks": 1,
    "extraction_confidence": 1.0
  },
  "document": "Table 1. Qatar: Selected Macroeconomic Indicators, 2020-29\\n\\n| Indicator | 2020 | 2021 |..."
}
"""

if __name__ == "__main__":
    print(EXAMPLE_INDEXED_RECORD)
