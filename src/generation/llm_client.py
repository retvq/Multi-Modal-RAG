"""
Gemini LLM Client for RAG Generation.

This module provides a clean abstraction over Google's Generative AI API
for use in the RAG pipeline. It enforces structured output to maintain
citation accuracy.
"""

import os
import json
import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# Load .env file if present
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

import google.generativeai as genai


@dataclass
class LLMResponse:
    """Structured response from the LLM."""
    answer_text: str
    cited_chunks: List[int]  # Indices of chunks actually used
    confidence: str  # "high", "medium", "low", "none"
    reasoning: Optional[str] = None  # Internal chain-of-thought (not shown to user)


class GeminiClient:
    """
    Gemini LLM client for grounded answer generation.
    
    Key Design Principles:
    1. GROUNDED: Only use information from provided context
    2. CITED: Every claim must reference a chunk index
    3. STRUCTURED: Output is JSON for reliable parsing
    """
    
    # System instruction for grounded RAG
    SYSTEM_INSTRUCTION = """You are a precise document analysis assistant for IMF financial reports.

STRICT RULES:
1. ONLY use information from the provided CONTEXT chunks
2. NEVER add information not in the context
3. ALWAYS cite sources using [X] where X is the chunk index (0-based)
4. If the context doesn't contain the answer, say "The provided documents do not contain this information"
5. Be concise but complete

TABLE DATA HANDLING:
When the query asks about table data, economic indicators, or numerical values:
1. Present actual data values from the table context
2. Format response with the years as headers: **Years:** 2020, 2021, 2022, ...
3. List each indicator with its values, e.g.:
   * **Nominal GDP:** 525.7, 654.2, 858.0, ...
   * **Real GDP (percent):** -3.6, 1.6, 4.2, ...
4. Include the actual numbers, not descriptions of what the table contains

SPECIAL HANDLING FOR FOOTNOTES:
- When a FOOTNOTE chunk is relevant, explain its contextual relevance
- Connect the footnote back to what it clarifies or qualifies in the main text
- Format: "According to footnote [X], [explanation of what it clarifies]"

OUTPUT FORMAT (JSON):
{
    "answer": "Your answer text with [0] inline citations [1] like this",
    "cited_chunks": [0, 1],
    "confidence": "high|medium|low|none",
    "reasoning": "Brief internal note on how you derived the answer"
}"""

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: str = None):
        """
        Initialize the Gemini client.
        
        Args:
            model_name: Model to use. Options:
                - "gemini-1.5-flash": Fast, high limits (recommended)
                - "gemini-3-flash": Latest flash model
            api_key: Optional API key override
        """
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY")
            
        if not api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Get your key from https://aistudio.google.com/apikey"
            )
        
        genai.configure(api_key=api_key)
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=self.SYSTEM_INSTRUCTION,
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temperature for factual accuracy
                top_p=0.95,
                response_mime_type="application/json",
            ),
        )
        self.model_name = model_name
    
    def generate_answer(
        self,
        query: str,
        chunks: List[Dict],
        max_chunks: int = 10
    ) -> LLMResponse:
        """
        Generate a grounded answer from retrieved chunks.
        
        Args:
            query: User's question
            chunks: Retrieved chunks with 'content', 'page_number', 'modality', 'section_path'
            max_chunks: Maximum chunks to include in context (to respect token limits)
        
        Returns:
            LLMResponse with answer, citations, and confidence
        """
        # Format context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks[:max_chunks]):
            page = chunk.get("page_number", "?")
            modality = chunk.get("modality", "TEXT")
            section = chunk.get("section_path", "")
            content = chunk.get("content", "")[:2000]  # Truncate very long chunks
            
            context_parts.append(
                f"[CHUNK {i}] (Page {page}, {modality}, {section})\n{content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build the prompt
        prompt = f"""CONTEXT:
{context}

QUESTION: {query}

Generate a grounded answer using ONLY the context above. Output valid JSON."""

        try:
            response = self.model.generate_content(prompt)
            
            # Parse JSON response
            response_text = response.text.strip()
            
            # Robust JSON extraction: Find first '{' and last '}'
            try:
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    response_text = response_text[start_idx : end_idx + 1]
                
                parsed = json.loads(response_text)
                
                return LLMResponse(
                    answer_text=parsed.get("answer", ""),
                    cited_chunks=parsed.get("cited_chunks", []),
                    confidence=parsed.get("confidence", "medium"),
                    reasoning=parsed.get("reasoning"),
                )
            except json.JSONDecodeError as e:
                # If extraction failed, try stripping markdown as fallback
                clean_text = re.sub(r'^```json?\n?', '', response.text.strip())
                clean_text = re.sub(r'\n?```$', '', clean_text)
                try:
                    parsed = json.loads(clean_text)
                    return LLMResponse(
                        answer_text=parsed.get("answer", ""),
                        cited_chunks=parsed.get("cited_chunks", []),
                        confidence=parsed.get("confidence", "medium"),
                        reasoning=parsed.get("reasoning"),
                    )
                except json.JSONDecodeError:
                    raise e
            
        except json.JSONDecodeError as e:
            # Fallback: Return raw text if JSON parsing fails
            return LLMResponse(
                answer_text=response.text if response else "Failed to generate response",
                cited_chunks=[],
                confidence="low",
                reasoning=f"JSON parse error: {e}",
            )
        except Exception as e:
            return LLMResponse(
                answer_text=f"Error generating response: {str(e)}",
                cited_chunks=[],
                confidence="none",
                reasoning=str(e),
            )
    
    def is_available(self) -> bool:
        """Check if the API is available and configured."""
        try:
            # Use a simple model without JSON mode for testing
            test_model = genai.GenerativeModel(self.model_name)
            response = test_model.generate_content("Say OK")
            return response.text is not None and len(response.text.strip()) > 0
        except Exception:
            return False
    
    def generate_briefing(
        self,
        chunks: List[Dict],
        focus_area: str = None,
        max_chunks: int = 15
    ) -> LLMResponse:
        """
        Generate an executive briefing/summary from document chunks.
        
        Args:
            chunks: List of retrieved chunks to summarize
            focus_area: Optional focus (e.g., "fiscal policy", "economic outlook")
            max_chunks: Maximum chunks to include
        
        Returns:
            LLMResponse with structured summary
        """
        # Format context
        context_parts = []
        for i, chunk in enumerate(chunks[:max_chunks]):
            page = chunk.get("page_number", "?")
            modality = chunk.get("modality", "TEXT")
            section = chunk.get("section_path", "")
            content = chunk.get("content", "")[:2000]
            
            context_parts.append(
                f"[CHUNK {i}] (Page {page}, {modality}, {section})\n{content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Build briefing prompt
        focus_instruction = f"Focus specifically on: {focus_area}" if focus_area else "Provide a comprehensive overview"
        
        prompt = f"""CONTEXT:
{context}

TASK: Generate an executive briefing/summary of the above document content.
{focus_instruction}

OUTPUT FORMAT (JSON):
{{
    "answer": "A structured executive summary with [0] inline citations. Use bullet points for key findings. Include: 1) Overview, 2) Key Points, 3) Recommendations (if applicable)",
    "cited_chunks": [0, 1, 2],
    "confidence": "high|medium|low",
    "reasoning": "Brief note on what was covered"
}}

Generate a clear, well-organized briefing grounded ONLY in the provided context."""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Handle markdown code blocks
            if response_text.startswith("```"):
                response_text = re.sub(r'^```json?\n?', '', response_text)
                response_text = re.sub(r'\n?```$', '', response_text)
            
            parsed = json.loads(response_text)
            
            return LLMResponse(
                answer_text=parsed.get("answer", ""),
                cited_chunks=parsed.get("cited_chunks", []),
                confidence=parsed.get("confidence", "medium"),
                reasoning=parsed.get("reasoning"),
            )
            
        except json.JSONDecodeError as e:
            return LLMResponse(
                answer_text=response.text if response else "Failed to generate briefing",
                cited_chunks=[],
                confidence="low",
                reasoning=f"JSON parse error: {e}",
            )
        except Exception as e:
            return LLMResponse(
                answer_text=f"Error generating briefing: {str(e)}",
                cited_chunks=[],
                confidence="none",
                reasoning=str(e),
            )


# Convenience function for quick testing
def test_gemini_connection():
    """Test if Gemini API is properly configured."""
    try:
        client = GeminiClient()
        print(f"  Model: {client.model_name}")
        print(f"  Testing connection...")
        
        if client.is_available():
            print("✓ Gemini API connected successfully!")
            return True
        else:
            print("✗ Gemini API connection test failed")
            return False
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
        return False
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return False


if __name__ == "__main__":
    test_gemini_connection()
