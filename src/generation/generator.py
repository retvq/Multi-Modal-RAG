"""
Answer Generation Module (Fixed)

Generates grounded answers with citations from retrieved chunks.

Features:
- Rule-based intent classification
- Intent-specific guardrails
- Table cell extraction for numeric lookups
- Immediate stop on valid extraction
- Citation deduplication
"""

import re
import json
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple
from enum import Enum
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))


# =============================================================================
# INTENT CLASSIFICATION
# =============================================================================

class QueryIntent(Enum):
    NUMERIC_LOOKUP = "numeric_lookup"
    TABLE_LOOKUP = "table_lookup"
    TEXT_SUMMARY = "text_summary"
    POLICY_QUESTION = "policy_question"
    UNKNOWN = "unknown"


# Regex patterns for each intent (order matters - most specific first)
INTENT_PATTERNS = {
    QueryIntent.TABLE_LOOKUP: [
        r"table\s*\d+",
        r"(show|display|get)\s+(me\s+)?table",
        r"(data|info|information)\s+(from|in)\s+table",
    ],
    QueryIntent.POLICY_QUESTION: [
        r"\b(recommend|recommendation|recommendations|policy|policies)\b",
        r"\b(should|advised|stance|position)\b",
        r"(authorities|staff|imf).*(view|opinion|assess|recommend)",
        r"what (should|must|needs to)",
    ],
    QueryIntent.NUMERIC_LOOKUP: [
        # Requires year + numeric indicator
        r"(what|how).*(rate|growth|balance|deficit|surplus|gdp|inflation|percent|%).*\b(20\d{2})\b",
        r"\b(20\d{2})\b.*(what|how).*(rate|growth|balance|deficit|surplus|gdp|inflation|percent|%)",
        r"how (much|many).*(percent|%|billion|million|rate)",
        r"what (is|was|are|were) (the|qatar'?s?)\s+\w+.*(rate|growth|%|percent).*\b(20\d{2})\b",
    ],
    QueryIntent.TEXT_SUMMARY: [
        r"\b(explain|describe|summarize|outline)\b",
        r"\b(summary|overview|synopsis)\b",
        r"tell me (about|regarding)",
        r"(main|key|overall)\s+(points|findings|conclusions|takeaways)",
    ],
}


def classify_intent(query: str) -> QueryIntent:
    """
    Classify query intent using regex patterns only.
    
    Order:
    1. TABLE_LOOKUP (most explicit - "Table 1")
    2. POLICY_QUESTION (specific vocabulary)
    3. NUMERIC_LOOKUP (requires year + indicator)
    4. TEXT_SUMMARY (explanation verbs)
    5. UNKNOWN (fallback)
    """
    query_lower = query.lower()
    
    # Check patterns in priority order
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, query_lower):
                return intent
    
    return QueryIntent.UNKNOWN


# In-domain keyword allowlist for UNKNOWN intent filtering
IN_DOMAIN_KEYWORDS = {
    # Core economic terms
    "economy", "economic", "fiscal", "monetary", "financial",
    "gdp", "growth", "inflation", "deficit", "surplus", "revenue", "expenditure",
    "budget", "debt", "reserve", "currency", "exchange", "trade", "export", "import",
    # Qatar-specific
    "qatar", "qatari", "doha", "gcc", "gulf",
    # Policy terms
    "policy", "policies", "reform", "regulation", "tax", "vat",
    "government", "authorities", "imf", "staff", "article",
    # Report-specific
    "outlook", "forecast", "projection", "risk", "indicator", "table", "figure",
    "annex", "appendix", "sector", "bank", "investment", "infrastructure",
    # Common economic indicators
    "rate", "percent", "billion", "million", "index", "ratio",
}


# Explicitly unsupported topics - queries with these are always out-of-domain
EXPLICITLY_UNSUPPORTED_TOPICS = {
    # Weather/climate (non-economic sense)
    "weather", "temperature", "rain", "sunny", "cloudy",
    # Sports/entertainment
    "football", "soccer", "sports", "movie", "music", "concert",
    # Travel/tourism (non-economic sense)
    "hotel", "flight", "restaurant", "tourist", "vacation", "tourism",
    # Personal/general
    "recipe", "health", "doctor", "hospital", "school",
    # Technology (consumer sense)
    "phone", "laptop", "computer", "game", "gaming",
    # Demographics (not in document)
    "population", "census", "demographics",
}


# Section path priorities for TEXT_SUMMARY
PREFERRED_SECTIONS = {
    "outlook", "outlook and risks", "staff report", "context",
    "recent economic developments", "policies",
}
DEPRIORITIZED_SECTIONS = {
    "selected issues", "box", "appendix", "annex", "statistical",
}


# Expanded policy detection patterns
POLICY_PATTERNS = [
    r"\brecommend",
    r"\brecommendation",
    r"\bshould\b",
    r"\bauthorities'",
    r"\bstaff's assessment",
    r"\bcommitment to",
    r"\bfiscal discipline",
    r"\bpolicy stance",
    r"\bmedium-term fiscal",
    r"\bconsolidation",
    r"\burge",
    r"\bsuggest",
    r"\bencourage",
    r"\badvised",
    r"\bis critical",
    r"\bis essential",
    r"\bis important",
]


def is_query_in_domain(query: str) -> bool:
    """
    Check if a query is within the document's domain.
    
    Uses a blocklist + allowlist approach:
    1. If any blocklist keyword is present: OUT OF DOMAIN
    2. If any allowlist keyword is present: IN DOMAIN
    3. Otherwise: OUT OF DOMAIN
    
    Returns True if query is related to economics/Qatar/IMF reports.
    """
    query_lower = query.lower()
    
    # Step 1: Check blocklist first
    for keyword in EXPLICITLY_UNSUPPORTED_TOPICS:
        if keyword in query_lower:
            return False
    
    # Step 2: Check allowlist
    for keyword in IN_DOMAIN_KEYWORDS:
        if keyword in query_lower:
            return True
    
    # Step 3: Default to out-of-domain
    return False


# =============================================================================
# GUARDRAILS PER INTENT
# =============================================================================

@dataclass
class Guardrails:
    allowed_modalities: List[str]
    modality_priority: List[str]
    max_chunks: int
    max_length: int  # characters
    stop_on_first: bool


GUARDRAILS = {
    QueryIntent.NUMERIC_LOOKUP: Guardrails(
        allowed_modalities=["TABLE", "TEXT"],
        modality_priority=["TABLE", "TEXT"],
        max_chunks=1,
        max_length=100,
        stop_on_first=True,
    ),
    QueryIntent.TABLE_LOOKUP: Guardrails(
        allowed_modalities=["TABLE"],
        modality_priority=["TABLE"],
        max_chunks=1,
        max_length=500,
        stop_on_first=True,
    ),
    QueryIntent.TEXT_SUMMARY: Guardrails(
        allowed_modalities=["TEXT"],  # TEXT only, no FIGURE
        modality_priority=["TEXT"],
        max_chunks=5,  # Check more chunks to find 2-3 good sentences
        max_length=400,
        stop_on_first=False,
    ),
    QueryIntent.POLICY_QUESTION: Guardrails(
        allowed_modalities=["TEXT"],
        modality_priority=["TEXT"],
        max_chunks=2,
        max_length=200,
        stop_on_first=True,
    ),
    QueryIntent.UNKNOWN: Guardrails(
        allowed_modalities=["TEXT", "TABLE", "FIGURE", "FOOTNOTE"],
        modality_priority=["TEXT", "TABLE", "FIGURE", "FOOTNOTE"],
        max_chunks=1,
        max_length=150,
        stop_on_first=True,
    ),
}


# =============================================================================
# CITATION HANDLING
# =============================================================================

class AnswerType(Enum):
    DIRECT = "direct"
    PARTIAL = "partial"
    NOT_FOUND = "not_found"
    CONFLICTING = "conflicting"


class AnswerConfidence(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"


@dataclass
class Citation:
    """Single citation linking claim to source."""
    citation_id: str
    chunk_id: str
    page_number: int
    modality: str
    section_path: str
    table_id: Optional[str] = None
    figure_id: Optional[str] = None
    quoted_text: str = ""
    
    def format_short(self) -> str:
        """Format citation for inline reference."""
        if self.table_id:
            return f"{self.table_id}, Page {self.page_number}"
        elif self.figure_id:
            return f"{self.figure_id}, Page {self.page_number}"
        else:
            return f"Page {self.page_number}"


class CitationRegistry:
    """Maintains unique citations with first-appearance ordering."""
    
    def __init__(self):
        self._citations: OrderedDict = OrderedDict()
        self._id_to_marker: Dict[str, str] = {}
        self._next_marker: int = 1
    
    def add(self, citation: Citation) -> str:
        """Add citation, return marker. Returns existing marker if duplicate."""
        chunk_id = citation.chunk_id
        
        if chunk_id in self._citations:
            return self._id_to_marker[chunk_id]
        
        marker = f"[{self._next_marker}]"
        self._citations[chunk_id] = citation
        self._id_to_marker[chunk_id] = marker
        citation.citation_id = marker
        self._next_marker += 1
        
        return marker
    
    def get_all(self) -> List[Citation]:
        """Return all citations in order of first appearance."""
        return list(self._citations.values())
    
    def __len__(self) -> int:
        return len(self._citations)


@dataclass
class Answer:
    """Schema-valid answer object."""
    answer_id: str
    query: str
    answer_text: str
    answer_type: AnswerType
    confidence: AnswerConfidence
    citations: List[Citation]
    sources_used: int
    sources_available: int
    generation_time_ms: float
    intent: QueryIntent = QueryIntent.UNKNOWN
    
    def to_dict(self) -> Dict:
        return {
            "answer_id": self.answer_id,
            "query": self.query,
            "answer_text": self.answer_text,
            "answer_type": self.answer_type.value,
            "confidence": self.confidence.value,
            "citations": [asdict(c) for c in self.citations],
            "sources_used": self.sources_used,
            "sources_available": self.sources_available,
            "generation_time_ms": self.generation_time_ms,
            "intent": self.intent.value,
        }


# =============================================================================
# TABLE EXTRACTION
# =============================================================================

def parse_markdown_table(content: str) -> List[List[str]]:
    """Parse markdown table into list of rows."""
    lines = content.strip().split('\n')
    rows = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('|-') or line.startswith('|--'):
            continue
        if '|' in line:
            cells = [c.strip() for c in line.split('|')]
            cells = [c for c in cells if c]
            if cells:
                rows.append(cells)
    
    return rows


def extract_numeric_from_table(
    content: str,
    query_tokens: List[str],
    page_number: int,
    table_id: Optional[str],
    chunk_id: str,
    section_path: str
) -> Optional[Tuple[str, Citation]]:
    """
    Extract a single numeric value from table based on query tokens.
    Returns (value, citation) or None.
    """
    rows = parse_markdown_table(content)
    if len(rows) < 2:
        return None
    
    header_row = rows[0]
    data_rows = rows[1:]
    
    # Find target year from query
    target_year = None
    for token in query_tokens:
        if re.match(r"20\d{2}", token):
            target_year = token
            break
    
    if not target_year:
        return None
    
    # Find year column
    target_col = None
    for col_idx, header in enumerate(header_row):
        if target_year in header:
            target_col = col_idx
            break
    
    if target_col is None:
        return None
    
    # Find matching row by keywords
    keyword_tokens = [t.lower() for t in query_tokens if not re.match(r"20\d{2}", t)]
    keyword_tokens = [t for t in keyword_tokens if len(t) > 2]  # Skip short words
    
    best_row = None
    best_score = 0
    
    for row in data_rows:
        if not row:
            continue
        
        row_header = row[0].lower()
        score = sum(1 for kw in keyword_tokens if kw in row_header)
        
        if score > best_score:
            best_score = score
            best_row = row
    
    if best_row is None or best_score == 0:
        return None
    
    if target_col >= len(best_row):
        return None
    
    value = best_row[target_col].strip()
    row_label = best_row[0].strip()
    
    if not value or value in ["-", "...", "n/a", ""]:
        return None
    
    citation = Citation(
        citation_id="",
        chunk_id=chunk_id,
        page_number=page_number,
        modality="TABLE",
        section_path=section_path,
        table_id=table_id,
        quoted_text=f"{row_label} | {target_year} | {value}",
    )
    
    return (value, row_label, citation)


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

# Causal verbs that indicate analytical content
CAUSAL_VERBS = {
    "driven by", "supported by", "reflects", "due to", "as a result of",
    "contributes to", "led to", "resulted in", "attributed to", "caused by",
    "boosted by", "offset by", "constrained by", "underpinned by",
}

# Macroeconomic drivers that indicate substantive content
MACRO_DRIVERS = {
    "growth", "inflation", "fiscal", "buffers", "external", "hydrocarbon",
    "non-hydrocarbon", "exports", "imports", "surplus", "deficit", "gdp",
    "monetary", "revenue", "expenditure", "investment", "consumption",
}

# Hard rejection phrases for summary sentences
SUMMARY_REJECT_PHRASES = {
    "this box", "summarizes", "figure", "table", "staff concur",
    "authorities broadly concur", "broadly concurred", "concurred with",
    "outlined in", "appendix", "methodology", "dataset", "authors",
    "selected issues", "see box", "see figure", "see table",
}


def is_summary_worthy(sentence: str, section_path: str = "") -> bool:
    """
    Determine if a sentence is worthy of inclusion in a TEXT_SUMMARY.
    
    A sentence is VALID only if it satisfies at least 2 of:
    1. Contains causal verbs
    2. Mentions macroeconomic drivers
    3. Has narrative structure (>= 12 words)
    
    Hard rejections (immediate False):
    - Starts with number or bullet
    - Contains any SUMMARY_REJECT_PHRASES
    - Comes from appendix/statistical sections
    """
    sent_lower = sentence.lower().strip()
    
    # =====================================================
    # HARD REJECTIONS
    # =====================================================
    
    # Reject if starts with number/bullet
    if re.match(r'^[\d•\-\*]', sentence.strip()):
        return False
    
    # Reject if contains any rejection phrase
    for phrase in SUMMARY_REJECT_PHRASES:
        if phrase in sent_lower:
            return False
    
    # Reject if from appendix/statistical sections
    section_lower = section_path.lower()
    if any(x in section_lower for x in ["appendix", "statistical", "annex"]):
        return False
    
    # Reject very short sentences
    if len(sentence) < 30:
        return False
    
    # =====================================================
    # POSITIVE SIGNALS (need at least 2)
    # =====================================================
    score = 0
    
    # Signal 1: Contains causal verbs
    has_causal = any(verb in sent_lower for verb in CAUSAL_VERBS)
    if has_causal:
        score += 1
    
    # Signal 2: Mentions macroeconomic drivers
    macro_count = sum(1 for driver in MACRO_DRIVERS if driver in sent_lower)
    if macro_count >= 1:
        score += 1
    
    # Signal 3: Has narrative structure (>= 12 words)
    word_count = len(sentence.split())
    if word_count >= 12:
        score += 1
    
    return score >= 2


def compute_token_overlap(sent1: str, sent2: str) -> float:
    """
    Compute token overlap ratio between two sentences.
    
    Returns overlap ratio (0.0 to 1.0).
    Uses set intersection over smaller set size.
    """
    tokens1 = set(sent1.lower().split())
    tokens2 = set(sent2.lower().split())
    
    # Remove common stopwords
    stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                 "to", "of", "in", "for", "on", "with", "at", "by", "from",
                 "and", "or", "but", "as", "if", "that", "this", "it"}
    tokens1 -= stopwords
    tokens2 -= stopwords
    
    if not tokens1 or not tokens2:
        return 0.0
    
    intersection = tokens1 & tokens2
    smaller_set = min(len(tokens1), len(tokens2))
    
    return len(intersection) / smaller_set if smaller_set > 0 else 0.0


def extract_relevant_sentence(
    content: str,
    query_tokens: List[str]
) -> Optional[str]:
    """
    Extract first relevant declarative sentence from text content.
    
    Rejects sentences that:
    - Start with numbers ("1 This box...", "2 In 2023...")
    - Mention methodology, datasets, or authors
    - Are too short (< 20 chars)
    """
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    keyword_tokens = [t.lower() for t in query_tokens if len(t) > 2]
    
    # Patterns to reject
    reject_patterns = [
        r'^\d+\s',  # Starts with number
        r'^This box',  # Box reference
        r'^Box \d+',  # Box reference
        r'^See ',  # Cross-reference
        r'\bauthors?\b',  # Methodology
        r'\bdataset\b',  # Methodology
        r'\bmethodology\b',  # Methodology
        r'\boutlined in\b',  # Cross-reference
    ]
    
    for sent in sentences:
        sent = sent.strip()
        
        # Skip short sentences
        if len(sent) < 20:
            continue
        
        # Skip rejected patterns
        should_reject = False
        for pattern in reject_patterns:
            if re.search(pattern, sent, re.IGNORECASE):
                should_reject = True
                break
        if should_reject:
            continue
        
        # Check for keyword matches
        sent_lower = sent.lower()
        matches = sum(1 for kw in keyword_tokens if kw in sent_lower)
        if matches >= 2:
            return sent
    
    return None


def extract_policy_sentence(content: str) -> Optional[Tuple[str, str]]:
    """
    Extract sentence containing policy language.
    
    Uses expanded POLICY_PATTERNS for detection.
    Returns (sentence, attribution) where attribution is 'authorities' or 'staff' or 'general'.
    """
    sentences = re.split(r'(?<=[.!?])\s+', content)
    
    for sent in sentences:
        sent = sent.strip()
        
        # Skip short sentences
        if len(sent) < 20:
            continue
        
        sent_lower = sent.lower()
        
        # Check for policy patterns
        is_policy = False
        for pattern in POLICY_PATTERNS:
            if re.search(pattern, sent_lower):
                is_policy = True
                break
        
        if is_policy:
            # Determine attribution
            if "authorities" in sent_lower:
                attribution = "authorities"
            elif "staff" in sent_lower:
                attribution = "staff"
            else:
                attribution = "general"
            
            return (sent, attribution)
    
    return None


# =============================================================================
# TEXT-TABLE FALLBACK (for TEXT chunks that look like tables)
# =============================================================================

def detect_text_table(content: str) -> Tuple[bool, float]:
    """
    Detect if TEXT chunk contains table-like data.
    
    Returns (is_table_like, confidence) where confidence > 0.5 means table-like.
    
    Detection criteria (need at least 2 of 4):
    1. Contains ≥2 4-digit years
    2. Multiple numeric values per line
    3. Row-style labels (GDP, inflation, etc.)
    4. Consistent line structure
    """
    lines = content.strip().split('\n')
    score = 0
    
    # Criterion 1: Contains ≥2 4-digit years
    years_found = len(re.findall(r'\b20\d{2}\b', content))
    if years_found >= 2:
        score += 1
    
    # Criterion 2: Multiple numeric values per line
    lines_with_multiple_numbers = 0
    for line in lines:
        numbers = re.findall(r'-?\d+\.?\d*', line)
        if len(numbers) >= 2:
            lines_with_multiple_numbers += 1
    if lines_with_multiple_numbers >= 2:
        score += 1
    
    # Criterion 3: Row-style labels (economic indicators)
    indicator_patterns = [
        r'(?i)(gdp|growth|inflation|deficit|surplus|export|import)',
        r'(?i)(revenue|expenditure|balance|account|reserve)',
        r'(?i)(rate|percent|ratio|index)',
    ]
    indicators_found = 0
    for pattern in indicator_patterns:
        if re.search(pattern, content):
            indicators_found += 1
    if indicators_found >= 2:
        score += 1
    
    # Criterion 4: Consistent line structure (similar length lines)
    if len(lines) >= 3:
        lengths = [len(l) for l in lines if len(l) > 10]
        if lengths:
            avg_len = sum(lengths) / len(lengths)
            consistent = sum(1 for l in lengths if abs(l - avg_len) < avg_len * 0.5)
            if consistent >= len(lengths) * 0.5:
                score += 1
    
    confidence = score / 4.0
    is_table_like = score >= 2
    
    return is_table_like, confidence


def parse_text_table(content: str) -> Optional[Dict]:
    """
    Parse TEXT content into pseudo-table structure.
    
    Returns dict with:
        - header_years: List of years found in header
        - rows: List of {label: str, values: dict[year: value]}
    
    Returns None if parsing fails.
    """
    lines = content.strip().split('\n')
    if len(lines) < 2:
        return None
    
    # Find header line with years
    header_line = None
    header_line_idx = -1
    for i, line in enumerate(lines[:5]):  # Check first 5 lines
        years = re.findall(r'\b(20\d{2})\b', line)
        if len(years) >= 2:
            header_line = line
            header_line_idx = i
            break
    
    if header_line is None:
        return None
    
    # Extract years and their approximate positions
    header_years = []
    for match in re.finditer(r'\b(20\d{2})\b', header_line):
        header_years.append({
            'year': match.group(1),
            'pos': match.start()
        })
    
    if len(header_years) < 2:
        return None
    
    # Parse data rows
    rows = []
    for line in lines[header_line_idx + 1:]:
        line = line.strip()
        if not line or len(line) < 5:
            continue
        
        # Extract row label (first text segment)
        label_match = re.match(r'^([A-Za-z][^0-9\n]{2,40})', line)
        if not label_match:
            continue
        
        label = label_match.group(1).strip()
        
        # Extract numeric values and try to align with years
        # Find all numbers with their positions
        values = {}
        for year_info in header_years:
            year = year_info['year']
            pos = year_info['pos']
            
            # Look for a number near this position in the line
            # Numbers should be within ±10 characters of the year position
            search_window = line[max(0, pos - 5):min(len(line), pos + 15)]
            num_match = re.search(r'-?\d+\.?\d*', search_window)
            if num_match:
                values[year] = num_match.group()
        
        if values:
            rows.append({
                'label': label,
                'values': values
            })
    
    if not rows:
        return None
    
    return {
        'header_years': [y['year'] for y in header_years],
        'rows': rows
    }


def extract_numeric_from_text_table(
    content: str,
    query_tokens: List[str],
    page_number: int,
    chunk_id: str,
    section_path: str
) -> Optional[Tuple[str, str, Citation]]:
    """
    Extract numeric value from TEXT chunk that looks like a table.
    
    This is a fallback for when TABLE extraction fails.
    Only called for NUMERIC_LOOKUP intent.
    
    Returns (value, row_label, citation) or None.
    """
    # Step 1: Check if content looks like a table
    is_table_like, confidence = detect_text_table(content)
    
    if not is_table_like or confidence < 0.5:
        return None
    
    # Step 2: Parse into pseudo-table
    parsed = parse_text_table(content)
    
    if parsed is None:
        return None
    
    # Step 3: Find target year from query
    target_year = None
    for token in query_tokens:
        if re.match(r'20\d{2}', token):
            target_year = token
            break
    
    if not target_year:
        return None
    
    # Check if target year exists in table
    if target_year not in parsed['header_years']:
        return None
    
    # Step 4: Find matching row by keywords
    keyword_tokens = [t.lower() for t in query_tokens if not re.match(r'20\d{2}', t)]
    keyword_tokens = [t for t in keyword_tokens if len(t) > 2]
    
    best_row = None
    best_score = 0
    
    for row in parsed['rows']:
        label_lower = row['label'].lower()
        score = sum(1 for kw in keyword_tokens if kw in label_lower)
        
        if score > best_score and target_year in row['values']:
            best_score = score
            best_row = row
    
    if best_row is None or best_score == 0:
        return None
    
    # Step 5: Extract value
    value = best_row['values'].get(target_year)
    
    if not value or value in ['-', '...', 'n/a', '']:
        return None
    
    row_label = best_row['label']
    
    # Step 6: Create citation (honest: TEXT modality, not Table)
    citation = Citation(
        citation_id="",
        chunk_id=chunk_id,
        page_number=page_number,
        modality="TEXT",  # Honest: this is TEXT, not TABLE
        section_path=section_path,
        table_id=None,  # No table_id since it's TEXT
        quoted_text=f"{row_label} {target_year} {value}",
    )
    
    return (value, row_label, citation)


# =============================================================================
# ANSWER GENERATOR
# =============================================================================

class AnswerGenerator:
    """
    Generates answers from retrieved chunks with intent-based guardrails.
    
    Supports two modes:
    - use_llm=False: Deterministic rule-based extraction (fast, predictable)
    - use_llm=True: LLM-powered generation via Gemini (more natural, flexible)
    """
    
    def __init__(self, use_llm: bool = False, llm_client=None, api_key: str = None):
        self.use_llm = use_llm
        self.llm_client = llm_client
        
        # Try to initialize LLM client if use_llm is True
        if self.use_llm and self.llm_client is None:
            try:
                from src.generation.llm_client import GeminiClient
                self.llm_client = GeminiClient(api_key=api_key)
                self._llm_available = True
            except Exception as e:
                print(f"Warning: LLM client failed to initialize: {e}")
                self._llm_available = False
        else:
            self._llm_available = self.llm_client is not None
    
    def generate(self, query: str, chunks: List[Dict]) -> Answer:
        """Generate answer from query and retrieved chunks."""
        start_time = time.time()
        
        # Step 1: Classify intent
        intent = classify_intent(query)
        guardrails = GUARDRAILS[intent]
        
        # Step 2: Check for empty context
        if not chunks:
            return self._not_found(query, intent, "No relevant documents retrieved.", start_time)
        
        # =====================================================
        # LLM-BASED GENERATION PATH
        # =====================================================
        if self.use_llm and self._llm_available:
            try:
                return self._generate_with_llm(query, chunks, intent, start_time)
            except Exception as e:
                # Fallback to rule-based on LLM error
                print(f"LLM generation failed, falling back to rule-based: {e}")
        
        # =====================================================
        # RULE-BASED GENERATION PATH (original logic)
        # =====================================================
        
        # Step 3: Filter and sort by allowed modalities
        filtered = [c for c in chunks if c.get("modality") in guardrails.allowed_modalities]
        
        if not filtered:
            return self._not_found(query, intent, "No chunks match allowed modalities.", start_time)
        
        # Sort by modality priority
        def priority_key(c):
            mod = c.get("modality", "TEXT")
            try:
                return guardrails.modality_priority.index(mod)
            except ValueError:
                return 999
        
        filtered.sort(key=priority_key)
        
        # Step 4: Generate based on intent
        query_tokens = query.replace("?", "").replace("'s", " ").split()
        
        # Guard: Ambiguous "all tables" query
        if re.search(r"\ball\s+tables\b|\btables\b", query.lower()):
            return Answer(
                answer_id=str(uuid.uuid4())[:8],
                query=query,
                answer_text="Multiple tables detected. Please specify a table number (e.g., 'Show me Table 1').",
                answer_type=AnswerType.NOT_FOUND,
                confidence=AnswerConfidence.NONE,
                citations=[],
                sources_used=0,
                sources_available=len(chunks),
                generation_time_ms=(time.time() - start_time) * 1000,
                intent=QueryIntent.TABLE_LOOKUP,
            )
        
        if intent == QueryIntent.NUMERIC_LOOKUP:
            return self._generate_numeric(query, filtered, guardrails, query_tokens, start_time)
        elif intent == QueryIntent.TABLE_LOOKUP:
            return self._generate_table(query, filtered, guardrails, start_time)
        elif intent == QueryIntent.TEXT_SUMMARY:
            return self._generate_summary(query, filtered, guardrails, query_tokens, start_time)
        elif intent == QueryIntent.POLICY_QUESTION:
            return self._generate_policy(query, filtered, guardrails, start_time)
        else:
            return self._generate_fallback(query, filtered, guardrails, query_tokens, start_time)
    
    def _generate_with_llm(
        self, query: str, chunks: List[Dict], intent: QueryIntent, start_time: float
    ) -> Answer:
        """
        Generate answer using Gemini LLM.
        
        The LLM receives the top 10 chunks and generates a grounded answer
        with citations. Citations are mapped back to our Citation schema.
        """
        from src.generation.llm_client import LLMResponse
        
        # Call LLM
        llm_response: LLMResponse = self.llm_client.generate_answer(
            query=query,
            chunks=chunks,
            max_chunks=10
        )
        
        # Map LLM citations to our Citation objects
        registry = CitationRegistry()
        
        for chunk_idx in llm_response.cited_chunks:
            if 0 <= chunk_idx < len(chunks):
                chunk = chunks[chunk_idx]
                citation = Citation(
                    citation_id="",
                    chunk_id=chunk.get("chunk_id", f"chunk_{chunk_idx}"),
                    page_number=chunk.get("page_number", 0),
                    modality=chunk.get("modality", "TEXT"),
                    section_path=chunk.get("section_path", ""),
                    table_id=chunk.get("table_id"),
                    figure_id=chunk.get("figure_id"),
                    quoted_text=chunk.get("content", "")[:100],
                )
                registry.add(citation)
        
        # Map LLM confidence to our enum
        confidence_map = {
            "high": AnswerConfidence.HIGH,
            "medium": AnswerConfidence.MEDIUM,
            "low": AnswerConfidence.LOW,
            "none": AnswerConfidence.NONE,
        }
        confidence = confidence_map.get(llm_response.confidence.lower(), AnswerConfidence.MEDIUM)
        
        # Determine answer type
        if not llm_response.answer_text or "do not contain" in llm_response.answer_text.lower():
            answer_type = AnswerType.NOT_FOUND
            confidence = AnswerConfidence.NONE
        elif len(registry) > 0:
            answer_type = AnswerType.DIRECT
        else:
            answer_type = AnswerType.PARTIAL
        
        return Answer(
            answer_id=str(uuid.uuid4())[:8],
            query=query,
            answer_text=llm_response.answer_text,
            answer_type=answer_type,
            confidence=confidence,
            citations=registry.get_all(),
            sources_used=len(registry),
            sources_available=len(chunks),
            generation_time_ms=(time.time() - start_time) * 1000,
            intent=intent,
        )

    
    def _generate_numeric(
        self, query: str, chunks: List[Dict], 
        guardrails: Guardrails, tokens: List[str], start_time: float
    ) -> Answer:
        """
        Generate NUMERIC_LOOKUP answer.
        
        Flow:
        1. Try TABLE modality extraction (primary)
        2. If NOT_FOUND, try TEXT-table fallback (for tables ingested as TEXT)
        3. If still NOT_FOUND, return proper refusal
        """
        registry = CitationRegistry()
        
        # =====================================================
        # PHASE 1: Try TABLE modality (primary extraction)
        # =====================================================
        for chunk in chunks[:guardrails.max_chunks + 2]:
            if chunk.get("modality") == "TABLE":
                result = extract_numeric_from_table(
                    chunk.get("content", ""),
                    tokens,
                    chunk.get("page_number", 0),
                    chunk.get("table_id"),
                    chunk.get("chunk_id", ""),
                    chunk.get("section_path", ""),
                )
                
                if result:
                    value, row_label, citation = result
                    marker = registry.add(citation)
                    answer_text = f"{row_label} is {value} {marker}."
                    
                    return Answer(
                        answer_id=str(uuid.uuid4())[:8],
                        query=query,
                        answer_text=answer_text,
                        answer_type=AnswerType.DIRECT,
                        confidence=AnswerConfidence.HIGH,
                        citations=registry.get_all(),
                        sources_used=1,
                        sources_available=len(chunks),
                        generation_time_ms=(time.time() - start_time) * 1000,
                        intent=QueryIntent.NUMERIC_LOOKUP,
                    )
        
        # =====================================================
        # PHASE 2: TEXT-Table Fallback
        # Only runs if TABLE extraction failed
        # =====================================================
        for chunk in chunks[:guardrails.max_chunks + 5]:  # Check more chunks
            if chunk.get("modality") == "TEXT":
                result = extract_numeric_from_text_table(
                    chunk.get("content", ""),
                    tokens,
                    chunk.get("page_number", 0),
                    chunk.get("chunk_id", ""),
                    chunk.get("section_path", ""),
                )
                
                if result:
                    value, row_label, citation = result
                    marker = registry.add(citation)
                    answer_text = f"{row_label} is {value} {marker}."
                    
                    return Answer(
                        answer_id=str(uuid.uuid4())[:8],
                        query=query,
                        answer_text=answer_text,
                        answer_type=AnswerType.DIRECT,
                        confidence=AnswerConfidence.MEDIUM,  # Lower confidence for fallback
                        citations=registry.get_all(),
                        sources_used=1,
                        sources_available=len(chunks),
                        generation_time_ms=(time.time() - start_time) * 1000,
                        intent=QueryIntent.NUMERIC_LOOKUP,
                    )
        
        # =====================================================
        # PHASE 3: Not Found
        # =====================================================
        return self._not_found(query, QueryIntent.NUMERIC_LOOKUP, 
                               "No matching numeric value found.", start_time)
    
    def _format_table_for_display(self, table_text: str) -> str:
        """
        Deterministic formatter to align years and values.
        Falls back to original text on any error.
        """
        try:
            # Split and clean lines
            lines = [line.strip() for line in table_text.split('\n') if line.strip()]
            if not lines:
                return table_text

            # 1. Identify year header (multiple 4-digit years)
            year_pattern = r'\b(20\d{2})\b'
            header_idx = -1
            years = []

            for i, line in enumerate(lines):
                matches = re.findall(year_pattern, line)
                # Heuristic: at least 3 years to identify a time-series header
                if len(matches) >= 3:
                     header_idx = i
                     years = matches
                     break
            
            if header_idx == -1:
                return table_text

            # 2. Build formatted table
            formatted_lines = []
            
            # Header
            # Label col width 30. Value col width 12.
            header_str = f"{'Label':<30}" + "".join([f"{y:>12}" for y in years])
            formatted_lines.append(header_str)
            formatted_lines.append("-" * len(header_str))
            
            # 3. Process data rows
            # Regex for numeric values (integers, floats, with commas, negatives)
            val_pattern = r'-?(?:(?:\d{1,3}(?:,\d{3})+)|(?:\d+))(?:\.\d+)?'
            
            for line in lines[header_idx+1:]:
                # Extract all potential numeric values
                vals = re.findall(val_pattern, line)
                
                # Only include rows where number of values >= number of years
                if len(vals) >= len(years):
                    # Take the LAST N values corresponding to the N years
                    row_vals = vals[-len(years):]
                    
                    # Determine Label:
                    # Search for the substring corresponding to the values to split label/data
                    escaped_vals = [re.escape(v) for v in row_vals]
                    seq_pattern = r'\s+'.join(escaped_vals)
                    
                    match = re.search(seq_pattern, line)
                    
                    label = ""
                    if match:
                        label = line[:match.start()].strip()
                    else:
                        # Fallback: if identifying sequence fails, skip safe
                        continue

                    if not label:
                        label = "Row"

                    # Truncate label to fit 29 chars
                    if len(label) > 29:
                        label_str = label[:27] + ".."
                    else:
                        label_str = label
                    
                    row_str = f"{label_str:<30}" + "".join([f"{v:>12}" for v in row_vals])
                    formatted_lines.append(row_str)
            
            if len(formatted_lines) <= 2: # Only header found
                 return table_text
                 
            return "\n".join(formatted_lines)
            
        except Exception:
            return table_text

    def _generate_table(
        self, query: str, chunks: List[Dict],
        guardrails: Guardrails, start_time: float
    ) -> Answer:
        """
        Generate TABLE_LOOKUP answer.
        
        Accepts a chunk as a match if ANY of:
        1. chunk.table_id matches requested table
        2. chunk.content.lower() contains "table {N}."
        3. chunk.content.lower() starts with "table {N}"
        """
        registry = CitationRegistry()
        
        # Extract table ID from query
        table_match = re.search(r"table\s*(\d+)", query.lower())
        if not table_match:
            return self._not_found(query, QueryIntent.TABLE_LOOKUP,
                                   "No table number specified.", start_time)
        
        target_num = table_match.group(1)
        target_patterns = [
            f"table {target_num}.",
            f"table {target_num}:",
            f"table {target_num} ",
        ]
        
        for chunk in chunks:
            chunk_table = chunk.get("table_id", "") or ""
            content_lower = chunk.get("content", "").lower()
            
            # Check 1: table_id match
            is_match = f"table {target_num}" in chunk_table.lower()
            
            # Check 2: content contains "table N."
            if not is_match:
                for pattern in target_patterns:
                    if pattern in content_lower:
                        is_match = True
                        break
            
            # Check 3: content starts with "table N"
            if not is_match:
                stripped = content_lower.strip()
                if stripped.startswith(f"table {target_num}"):
                    is_match = True
            
            if is_match:
                # Found matching table
                table_name = chunk_table if chunk_table else f"Table {target_num}"
                
                citation = Citation(
                    citation_id="",
                    chunk_id=chunk.get("chunk_id", ""),
                    page_number=chunk.get("page_number", 0),
                    modality=chunk.get("modality", "TABLE"),
                    section_path=chunk.get("section_path", ""),
                    table_id=table_name,
                    quoted_text=chunk.get("content", "")[:100],
                )
                marker = registry.add(citation)
                
                # Format table logic
                content = chunk.get("content", "")
                formatted_table = self._format_table_for_display(content)
                
                # If formatting failed/returned raw, apply legacy truncation
                if formatted_table == content:
                    lines = content.split("\n")[:15]
                    formatted_table = "\n".join(lines)
                
                answer_text = f"{table_name} (Page {chunk.get('page_number', 0)}):\n\n{formatted_table}\n\n{marker}"
                
                return Answer(
                    answer_id=str(uuid.uuid4())[:8],
                    query=query,
                    answer_text=answer_text,
                    answer_type=AnswerType.DIRECT,
                    confidence=AnswerConfidence.HIGH,
                    citations=registry.get_all(),
                    sources_used=1,
                    sources_available=len(chunks),
                    generation_time_ms=(time.time() - start_time) * 1000,
                    intent=QueryIntent.TABLE_LOOKUP,
                )
        
        return self._not_found(query, QueryIntent.TABLE_LOOKUP,
                               f"Table {target_num} not found.", start_time)
    
    def _generate_summary(
        self, query: str, chunks: List[Dict],
        guardrails: Guardrails, tokens: List[str], start_time: float
    ) -> Answer:
        """
        Generate TEXT_SUMMARY answer using two-pass selection.
        
        PASS 1: Collect all valid candidates from TEXT chunks
        PASS 2: Score candidates deterministically
        PASS 3: Select top-ranked with diversity constraints
        
        This fixes the ordering bug where low-quality sentences
        occupied slots before higher-quality analytical content.
        """
        
        # =====================================================
        # GUARD: Underspecified TEXT_SUMMARY detection
        # =====================================================
        # Filter strictly logic tokens
        lower_tokens = [t.lower() for t in tokens if t not in ["?", ".", ",", "!"]]
        
        # Criteria 1: Length <= 2 OR contains only generic words
        is_short = len(lower_tokens) <= 2
        
        generic_words = {
            "summary", "summarize", "summarise", "brief", "overview", 
            "the", "a", "an", "is", "of", "in", "to", "for", "please",
            "what", "give", "me", "can", "you", "document", "text", "full", "complete"
        }
        is_generic = all(t in generic_words for t in lower_tokens)
        
        # Criteria 2: No domain anchor keywords present
        anchors = ["outlook", "fiscal", "policy", "report", "risks", "economy", "reform", "gdp", "debt", "growth", "inflation", "sector"]
        has_anchor = any(anchor in query.lower() for anchor in anchors)
        
        if (is_short or is_generic) and not has_anchor:
             return Answer(
                answer_id=str(uuid.uuid4())[:8],
                query=query,
                answer_text="Please specify what you want summarized (for example, “Summarize the economic outlook”).",
                answer_type=AnswerType.NOT_FOUND,
                confidence=AnswerConfidence.NONE,
                citations=[],
                sources_used=0,
                sources_available=len(chunks),
                generation_time_ms=(time.time() - start_time) * 1000,
                intent=QueryIntent.TEXT_SUMMARY,
            )
        
        # =====================================================
        # PASS 1: Candidate Collection
        # =====================================================
        # Each candidate: {text, page, section, chunk_id, has_causal, has_macro, has_penalty}
        candidates = []
        
        text_chunks = [c for c in chunks if c.get("modality") == "TEXT"]
        
        for chunk in text_chunks[:guardrails.max_chunks + 10]:  # Check many chunks
            page_num = chunk.get("page_number", 0)
            section_path = chunk.get("section_path", "")
            chunk_id = chunk.get("chunk_id", "")
            content = chunk.get("content", "")
            
            # Split into sentences
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
            for sent in sentences:
                sent = sent.strip()
                
                # Skip short sentences
                if len(sent) < 30:
                    continue
                
                # Apply is_summary_worthy filter (includes hard rejections)
                if not is_summary_worthy(sent, section_path):
                    continue
                
                sent_lower = sent.lower()
                
                # Check keyword relevance (at least 1 query token)
                keyword_tokens = [t.lower() for t in tokens if len(t) > 2]
                matches = sum(1 for kw in keyword_tokens if kw in sent_lower)
                if matches < 1:
                    continue
                
                # Compute flags for scoring
                has_causal = any(verb in sent_lower for verb in CAUSAL_VERBS)
                has_macro = any(driver in sent_lower for driver in MACRO_DRIVERS)
                has_penalty = any(phrase in sent_lower for phrase in SUMMARY_REJECT_PHRASES)
                
                candidates.append({
                    "text": sent,
                    "page": page_num,
                    "section": section_path,
                    "chunk_id": chunk_id,
                    "has_causal": has_causal,
                    "has_macro": has_macro,
                    "has_penalty": has_penalty,
                })
        
        # =====================================================
        # PASS 2: Candidate Scoring (deterministic)
        # =====================================================
        def score_candidate(cand):
            score = 0
            
            # Positive signals
            if cand["has_causal"]:
                score += 3
            if cand["has_macro"]:
                score += 2
            
            # Section bonuses
            section_lower = cand["section"].lower()
            if any(pref in section_lower for pref in PREFERRED_SECTIONS):
                score += 1
            
            # Penalties
            if cand["has_penalty"]:
                score -= 2
            if "press release" in section_lower:
                score -= 1
            
            return score
        
        # Score and sort candidates
        for cand in candidates:
            cand["score"] = score_candidate(cand)
        
        candidates.sort(key=lambda c: (-c["score"], c["page"]))  # Highest score first, then by page
        
        # =====================================================
        # PASS 3: Final Selection with Diversity Constraints
        # =====================================================
        registry = CitationRegistry()
        selected = []  # List of (text, page, marker)
        used_pages = set()
        
        for cand in candidates:
            # Constraint 1: Max 1 sentence per page
            if cand["page"] in used_pages:
                continue
            
            # Constraint 2: Token overlap < 60% with already selected
            has_overlap = False
            for existing_text, _, _ in selected:
                overlap = compute_token_overlap(cand["text"], existing_text)
                if overlap > 0.6:
                    has_overlap = True
                    break
            
            if has_overlap:
                continue
            
            # This candidate passes - add it
            citation = Citation(
                citation_id="",
                chunk_id=cand["chunk_id"],
                page_number=cand["page"],
                modality="TEXT",
                section_path=cand["section"],
                quoted_text=cand["text"][:80],
            )
            marker = registry.add(citation)
            
            selected.append((cand["text"], cand["page"], marker))
            used_pages.add(cand["page"])
            
            # Stop at 3 sentences
            if len(selected) >= 3:
                break
        
        # =====================================================
        # Enforce minimum 2 sentences
        # =====================================================
        if len(selected) < 2:
            return self._not_found(query, QueryIntent.TEXT_SUMMARY,
                                   "Insufficient summary-quality content found.", start_time)
        
        # =====================================================
        # DEFENSIVE DEDUPLICATION (last-resort guard)
        # =====================================================
        seen_normalized = set()
        deduped_selected = []
        for text, page, marker in selected:
            # Normalize: lowercase, strip punctuation
            import string
            normalized = text.lower().translate(str.maketrans('', '', string.punctuation)).strip()
            if normalized not in seen_normalized:
                seen_normalized.add(normalized)
                deduped_selected.append((text, page, marker))
        
        selected = deduped_selected
        
        # =====================================================
        # SINGLE-WRITE BUFFER: Build answer_text exactly once
        # =====================================================
        answer_text = " ".join(f"{text} {marker}" for text, _, marker in selected)
        # DO NOT mutate answer_text after this point
        
        # =====================================================
        # ASSERTION: answer_text must NOT contain citation blocks
        # =====================================================
        assert "CITATIONS" not in answer_text, "Citation block leaked into answer_text"
        assert "TIMING" not in answer_text, "Timing block leaked into answer_text"
        
        return Answer(
            answer_id=str(uuid.uuid4())[:8],
            query=query,
            answer_text=answer_text,
            answer_type=AnswerType.DIRECT,
            confidence=AnswerConfidence.MEDIUM,
            citations=registry.get_all(),
            sources_used=len(registry),
            sources_available=len(chunks),
            generation_time_ms=(time.time() - start_time) * 1000,
            intent=QueryIntent.TEXT_SUMMARY,
        )
    
    def _generate_policy(
        self, query: str, chunks: List[Dict],
        guardrails: Guardrails, start_time: float
    ) -> Answer:
        """
        Generate POLICY_QUESTION answer.
        
        Uses expanded POLICY_PATTERNS for detection.
        Stops after first valid policy sentence.
        Never paraphrases or merges multiple policies.
        """
        registry = CitationRegistry()
        
        # Check more chunks to find policy content
        for chunk in chunks[:guardrails.max_chunks + 5]:
            result = extract_policy_sentence(chunk.get("content", ""))
            
            if result:
                sent, attribution = result
                
                citation = Citation(
                    citation_id="",
                    chunk_id=chunk.get("chunk_id", ""),
                    page_number=chunk.get("page_number", 0),
                    modality="TEXT",
                    section_path=chunk.get("section_path", ""),
                    quoted_text=sent[:80],
                )
                marker = registry.add(citation)
                answer_text = f"{sent} {marker}"
                
                return Answer(
                    answer_id=str(uuid.uuid4())[:8],
                    query=query,
                    answer_text=answer_text,
                    answer_type=AnswerType.DIRECT,
                    confidence=AnswerConfidence.HIGH,
                    citations=registry.get_all(),
                    sources_used=1,
                    sources_available=len(chunks),
                    generation_time_ms=(time.time() - start_time) * 1000,
                    intent=QueryIntent.POLICY_QUESTION,
                )
        
        return self._not_found(query, QueryIntent.POLICY_QUESTION,
                               "No policy recommendation found.", start_time)
    
    def _generate_fallback(
        self, query: str, chunks: List[Dict],
        guardrails: Guardrails, tokens: List[str], start_time: float
    ) -> Answer:
        """
        Generate UNKNOWN intent answer.
        
        STRICT REFUSAL RULE:
        - If query is out-of-domain: immediately return NOT_FOUND
        - If query is in-domain: attempt partial extraction
        """
        # =====================================================
        # GATE CHECK: Refuse out-of-domain queries immediately
        # =====================================================
        if not is_query_in_domain(query):
            return Answer(
                answer_id=str(uuid.uuid4())[:8],
                query=query,
                answer_text="I cannot answer this question. The retrieved documents do not contain information related to this topic.",
                answer_type=AnswerType.NOT_FOUND,
                confidence=AnswerConfidence.NONE,
                citations=[],  # No citations for out-of-domain
                sources_used=0,
                sources_available=len(chunks),
                generation_time_ms=(time.time() - start_time) * 1000,
                intent=QueryIntent.UNKNOWN,
            )
        
        # =====================================================
        # IN-DOMAIN: Attempt partial extraction
        # =====================================================
        registry = CitationRegistry()
        
        if chunks:
            chunk = chunks[0]
            content = chunk.get("content", "")
            
            # Get first sentence
            sentences = re.split(r'(?<=[.!?])\s+', content)
            first_sent = sentences[0] if sentences else content[:100]
            
            citation = Citation(
                citation_id="",
                chunk_id=chunk.get("chunk_id", ""),
                page_number=chunk.get("page_number", 0),
                modality=chunk.get("modality", "TEXT"),
                section_path=chunk.get("section_path", ""),
                quoted_text=first_sent[:80],
            )
            marker = registry.add(citation)
            
            answer_text = f"Based on retrieved content: {first_sent} {marker}"
            
            return Answer(
                answer_id=str(uuid.uuid4())[:8],
                query=query,
                answer_text=answer_text,
                answer_type=AnswerType.PARTIAL,
                confidence=AnswerConfidence.LOW,
                citations=registry.get_all(),
                sources_used=1,
                sources_available=len(chunks),
                generation_time_ms=(time.time() - start_time) * 1000,
                intent=QueryIntent.UNKNOWN,
            )
        
        return self._not_found(query, QueryIntent.UNKNOWN, "No content available.", start_time)
    
    def _not_found(
        self, query: str, intent: QueryIntent, reason: str, start_time: float
    ) -> Answer:
        """Create NOT_FOUND answer."""
        return Answer(
            answer_id=str(uuid.uuid4())[:8],
            query=query,
            answer_text=f"I cannot answer this question. {reason}",
            answer_type=AnswerType.NOT_FOUND,
            confidence=AnswerConfidence.NONE,
            citations=[],
            sources_used=0,
            sources_available=0,
            generation_time_ms=(time.time() - start_time) * 1000,
            intent=intent,
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

if __name__ == "__main__":
    # Test intent classification
    test_queries = [
        ("What is GDP growth for 2024?", QueryIntent.NUMERIC_LOOKUP),
        ("Show me Table 1", QueryIntent.TABLE_LOOKUP),
        ("Summarize the outlook", QueryIntent.TEXT_SUMMARY),
        ("What are the policy recommendations?", QueryIntent.POLICY_QUESTION),
        ("Hello", QueryIntent.UNKNOWN),
    ]
    
    print("Intent Classification Test:")
    for query, expected in test_queries:
        result = classify_intent(query)
        status = "PASS" if result == expected else "FAIL"
        print(f"  {status}: '{query}' -> {result.value}")
