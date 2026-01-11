"""
Text Processing Utilities

Common text operations used across extraction and normalization:
- Whitespace normalization
- Paragraph number detection
- Heading detection heuristics
"""

import re
from typing import Optional, Tuple


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace while preserving semantic structure.
    
    - Collapses multiple spaces to single space
    - Normalizes line breaks
    - Strips leading/trailing whitespace
    - Preserves paragraph breaks (double newline)
    
    Args:
        text: Raw extracted text
        
    Returns:
        Normalized text
    """
    if not text:
        return ""
    
    # Replace various whitespace characters with standard space
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    
    # Collapse multiple spaces (but not newlines) to single space
    text = re.sub(r" +", " ", text)
    
    # Normalize line endings
    text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 consecutive newlines
    
    # Strip each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    
    return text.strip()


def detect_paragraph_number(text: str) -> Tuple[Optional[int], str]:
    """
    Detect and extract official paragraph numbering.
    
    IMF documents use paragraph numbers like "1.", "2.", "37."
    at the start of paragraphs. This function extracts the number
    and returns the remaining text.
    
    Args:
        text: Text that may start with paragraph number
        
    Returns:
        (paragraph_number or None, text_without_number)
        
    Examples:
        "1. The economy grew..." → (1, "The economy grew...")
        "Without number..." → (None, "Without number...")
    """
    # Pattern: Start of string, optional whitespace, number, period, space
    pattern = r"^\s*(\d{1,3})\.\s+"
    
    match = re.match(pattern, text)
    if match:
        para_num = int(match.group(1))
        remaining = text[match.end():]
        return (para_num, remaining)
    
    return (None, text)


def detect_heading_level(
    text: str, 
    font_size: Optional[float] = None,
    is_bold: bool = False,
    is_uppercase: bool = False
) -> Optional[int]:
    """
    Detect if text is a heading and determine its level.
    
    Uses multiple signals:
    - Section patterns (A., B., I., II., Annex I)
    - Font size (larger = higher level)
    - Text styling (bold, uppercase)
    
    Args:
        text: Text content
        font_size: Font size in points (if available)
        is_bold: Whether text is bold
        is_uppercase: Whether text is uppercase
        
    Returns:
        Heading level 1-6, or None if not a heading
    """
    text_stripped = text.strip()
    
    # Level 1: Main sections (CONTEXT, OUTLOOK, STAFF APPRAISAL)
    level1_patterns = [
        r"^(CONTEXT|OUTLOOK|POLICIES|STAFF APPRAISAL)",
        r"^KEY ISSUES",
    ]
    for pattern in level1_patterns:
        if re.match(pattern, text_stripped, re.IGNORECASE):
            return 1
    
    # Level 2: Lettered sections (A. Fiscal Policy, B. Financial Sector)
    if re.match(r"^[A-Z]\.\s+", text_stripped):
        return 2
    
    # Level 2: Annexes (Annex I, Annex II)
    if re.match(r"^Annex\s+[IVX]+", text_stripped, re.IGNORECASE):
        return 2
    
    # Level 3: Numbered subsections within annexes
    if re.match(r"^\d{1,2}\.\s+[A-Z]", text_stripped):
        # Could be either paragraph or subsection
        # If short and ends without period, likely heading
        if len(text_stripped) < 100 and not text_stripped.endswith("."):
            return 3
    
    # Font-based heuristics (if available)
    if font_size:
        if font_size >= 14 and is_bold:
            return 1
        elif font_size >= 12 and is_bold:
            return 2
        elif font_size >= 11 and is_bold:
            return 3
    
    return None


def is_header_or_footer(text: str, y_position: float, page_height: float) -> bool:
    """
    Detect if text is likely a header or footer.
    
    Uses both position and content patterns:
    - Position: top 5% or bottom 5% of page
    - Content: page numbers, "QATAR", "INTERNATIONAL MONETARY FUND"
    
    Args:
        text: Text content
        y_position: Y position in page coordinates
        page_height: Total page height
        
    Returns:
        True if likely header/footer
    """
    # Position check (top/bottom 5%)
    relative_y = y_position / page_height
    if relative_y < 0.05 or relative_y > 0.95:
        return True
    
    # Content patterns for IMF documents
    text_stripped = text.strip()
    header_patterns = [
        r"^INTERNATIONAL MONETARY FUND$",
        r"^QATAR$",
        r"^\d{1,3}$",  # Standalone page number
        r"^IMF Country Report",
    ]
    
    for pattern in header_patterns:
        if re.match(pattern, text_stripped, re.IGNORECASE):
            return True
    
    return False


def clean_extracted_text(text: str) -> str:
    """
    Clean text after extraction.
    
    Handles common extraction artifacts:
    - Hyphenation at line breaks
    - Ligature issues
    - Control characters
    
    Args:
        text: Raw extracted text
        
    Returns:
        Cleaned text
    """
    # Remove control characters except newline and tab
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    
    # Fix hyphenation at line breaks (word- \n continued → word continued)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    
    # Fix common ligature issues
    text = text.replace("ﬁ", "fi")
    text = text.replace("ﬂ", "fl")
    text = text.replace("ﬀ", "ff")
    
    return text
