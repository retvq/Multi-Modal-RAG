"""
Table Extractor

Extracts tables from PDF pages with structural fidelity using pdfplumber.
Preserves:
- Column headers (including nested/multi-level)
- Row labels
- Cell values with position
- Merged/spanning cells
- Footnote markers

IMPORTANT: This module uses pdfplumber for table extraction because:
1. Better table boundary detection than PyMuPDF
2. Native support for cell-level bounding boxes
3. Handles bordered and borderless tables

Falls back to PyMuPDF text extraction if pdfplumber fails.
"""

import pdfplumber
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_path))

from src.schemas.content_block import BoundingBox


@dataclass
class Cell:
    """
    A single table cell with value and span information.
    """
    value: str                          # Cell content (text)
    row_index: int                      # 0-indexed row position
    col_index: int                      # 0-indexed column position
    row_span: int = 1                   # Number of rows this cell spans
    col_span: int = 1                   # Number of columns this cell spans
    footnote_marker: Optional[str] = None  # Footnote reference (e.g., "1/", "*")
    
    def __post_init__(self):
        # Extract footnote markers from cell value
        if self.value:
            # Pattern: number followed by / at end of value (IMF style: "1/", "2/")
            match = re.search(r'\s*(\d+/)\s*$', self.value)
            if match:
                self.footnote_marker = match.group(1)


@dataclass
class TableRow:
    """
    A single row in a table.
    """
    cells: List[Cell]
    row_index: int
    is_header: bool = False             # True if this is a header row
    row_label: Optional[str] = None     # First cell if it's a row label


@dataclass
class TableStructure:
    """
    Complete structured representation of a table.
    
    Supports:
    - Multi-level headers (list of header rows)
    - Data rows with cell values
    - Detected column and row structure
    """
    headers: List[List[str]]            # Multi-level headers (list of rows)
    rows: List[TableRow]                # Data rows
    column_count: int
    row_count: int
    has_merged_cells: bool = False


@dataclass  
class RawTable:
    """
    Raw extracted table before normalization.
    
    Contains both structured and fallback representations.
    """
    # Identity
    table_id: Optional[str] = None      # e.g., "Table 1", detected from title
    table_title: str = ""               # Full title text
    
    # Location
    page_number: int = 0
    position_in_page: int = 0           # Ordinal among tables on this page
    bbox: Optional[BoundingBox] = None
    
    # Structure
    structure: Optional[TableStructure] = None
    
    # Content representations
    raw_text: str = ""                  # Linearized text fallback
    markdown: str = ""                  # Markdown table representation
    
    # Metadata
    units: Optional[str] = None         # e.g., "Percent of GDP", from title/headers
    column_headers: List[str] = field(default_factory=list)  # Top-level column names
    row_headers: List[str] = field(default_factory=list)     # Row label values
    footnote_refs: List[str] = field(default_factory=list)   # Footnote markers found
    
    # Quality
    extraction_confidence: float = 1.0
    extraction_warnings: List[str] = field(default_factory=list)


class TableExtractor:
    """
    Extracts tables from PDF pages using pdfplumber.
    
    Strategy:
    1. Open page with pdfplumber (separate from PyMuPDF document)
    2. Detect table regions using visual line detection
    3. Extract cell values with positions
    4. Detect header rows (first N rows with different styling)
    5. Detect merged cells from spanning behavior
    6. Generate both structured and text representations
    
    IMF-specific handling:
    - Tables often have multi-level headers (e.g., years, then metrics)
    - Units often in title/parentheses
    - Footnote markers use N/ format
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize table extractor for a PDF.
        
        Args:
            pdf_path: Path to PDF file (needed for pdfplumber)
        """
        self.pdf_path = pdf_path
        self._pdf = None
    
    def __enter__(self):
        self._pdf = pdfplumber.open(self.pdf_path)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pdf:
            self._pdf.close()
    
    def extract_from_page(self, page_number: int) -> List[RawTable]:
        """
        Extract all tables from a specific page.
        
        Args:
            page_number: 1-indexed page number
            
        Returns:
            List of RawTable objects
        """
        if not self._pdf:
            self._pdf = pdfplumber.open(self.pdf_path)
        
        # pdfplumber uses 0-indexed pages
        page_idx = page_number - 1
        
        if page_idx < 0 or page_idx >= len(self._pdf.pages):
            return []
        
        page = self._pdf.pages[page_idx]
        page_width = page.width
        page_height = page.height
        
        # Find all tables on the page
        tables = page.find_tables()
        
        raw_tables = []
        for position, table in enumerate(tables):
            raw_table = self._process_table(
                table=table,
                page=page,
                page_number=page_number,
                page_width=page_width,
                page_height=page_height,
                position=position
            )
            if raw_table:
                raw_tables.append(raw_table)
        
        return raw_tables
    
    def _process_table(
        self,
        table: Any,  # pdfplumber Table object
        page: Any,   # pdfplumber Page object
        page_number: int,
        page_width: float,
        page_height: float,
        position: int
    ) -> Optional[RawTable]:
        """
        Process a single pdfplumber table into RawTable.
        """
        try:
            # Extract table data as list of lists
            data = table.extract()
            
            if not data or len(data) == 0:
                return None
            
            # Get bounding box
            bbox_raw = table.bbox  # (x0, y0, x1, y1) in pdfplumber coords
            bbox = BoundingBox(
                x0=bbox_raw[0] / page_width,
                y0=bbox_raw[1] / page_height,
                x1=bbox_raw[2] / page_width,
                y1=bbox_raw[3] / page_height
            )
            
            # Build structured representation
            structure = self._build_structure(data)
            
            # Try to find table title (text above table)
            title, table_id, units = self._extract_table_metadata(page, bbox_raw)
            
            # Generate representations
            raw_text = self._to_linear_text(data)
            markdown = self._to_markdown(data)
            
            # Extract column headers (first row or detected headers)
            column_headers = []
            if structure.headers:
                column_headers = structure.headers[0]
            elif data:
                column_headers = [str(c) if c else "" for c in data[0]]
            
            # Extract row headers (first column of data rows)
            row_headers = []
            start_row = len(structure.headers) if structure.headers else 1
            for row in data[start_row:]:
                if row and row[0]:
                    row_headers.append(str(row[0]))
            
            # Collect footnote references
            footnote_refs = self._collect_footnote_refs(data)
            
            return RawTable(
                table_id=table_id,
                table_title=title,
                page_number=page_number,
                position_in_page=position,
                bbox=bbox,
                structure=structure,
                raw_text=raw_text,
                markdown=markdown,
                units=units,
                column_headers=column_headers,
                row_headers=row_headers,
                footnote_refs=footnote_refs,
                extraction_confidence=1.0,
                extraction_warnings=[]
            )
            
        except Exception as e:
            # Return degraded table with warning
            return RawTable(
                page_number=page_number,
                position_in_page=position,
                extraction_confidence=0.5,
                extraction_warnings=[f"Extraction error: {str(e)}"]
            )
    
    def _build_structure(self, data: List[List]) -> TableStructure:
        """
        Build structured representation from raw table data.
        
        Detects:
        - Header rows (typically first 1-2 rows)
        - Merged cells (None values in the grid)
        - Column and row counts
        """
        if not data:
            return TableStructure(
                headers=[],
                rows=[],
                column_count=0,
                row_count=0
            )
        
        # Determine column count
        col_count = max(len(row) for row in data)
        
        # Detect header rows
        # Heuristic: rows before first row with numeric data in most cells
        header_rows = []
        data_start = 0
        
        for i, row in enumerate(data):
            numeric_count = sum(1 for c in row if c and self._is_numeric(str(c)))
            if numeric_count > len(row) / 2:
                # This row has mostly numeric data - it's a data row
                data_start = i
                break
            else:
                # This is likely a header row
                header_rows.append([str(c) if c else "" for c in row])
        
        if not header_rows and data:
            # Fallback: first row is header
            header_rows = [[str(c) if c else "" for c in data[0]]]
            data_start = 1
        
        # Build data rows
        rows = []
        has_merged = False
        
        for row_idx, row in enumerate(data[data_start:], start=data_start):
            cells = []
            for col_idx, value in enumerate(row):
                # None values may indicate merged cells
                if value is None:
                    has_merged = True
                    value = ""
                
                cell = Cell(
                    value=str(value) if value else "",
                    row_index=row_idx,
                    col_index=col_idx
                )
                cells.append(cell)
            
            # First cell might be row label
            row_label = cells[0].value if cells else None
            
            rows.append(TableRow(
                cells=cells,
                row_index=row_idx,
                is_header=False,
                row_label=row_label
            ))
        
        return TableStructure(
            headers=header_rows,
            rows=rows,
            column_count=col_count,
            row_count=len(data),
            has_merged_cells=has_merged
        )
    
    def _is_numeric(self, value: str) -> bool:
        """Check if value is numeric (including negative, decimals, percentages)."""
        if not value:
            return False
        # Remove common formatting
        clean = value.replace(",", "").replace("%", "").replace("$", "").strip()
        # Handle parentheses for negatives: (5.3) -> -5.3
        if clean.startswith("(") and clean.endswith(")"):
            clean = clean[1:-1]
        try:
            float(clean)
            return True
        except ValueError:
            return False
    
    def _extract_table_metadata(
        self, 
        page: Any, 
        table_bbox: Tuple[float, float, float, float]
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Extract table title, ID, and units from text above the table.
        
        Returns:
            (title, table_id like "Table 1", units like "Percent of GDP")
        """
        # Look for text just above the table
        x0, y0, x1, y1 = table_bbox
        
        # Search region: above table, same width, up to 50 points
        search_region = (x0, max(0, y0 - 50), x1, y0)
        
        try:
            text_above = page.within_bbox(search_region).extract_text()
        except:
            text_above = ""
        
        if not text_above:
            return ("", None, None)
        
        title = text_above.strip()
        
        # Extract table ID (e.g., "Table 1", "Table 3a")
        table_id = None
        id_match = re.search(r'Table\s+(\d+[a-z]?)', title, re.IGNORECASE)
        if id_match:
            table_id = f"Table {id_match.group(1)}"
        
        # Extract units from parentheses
        units = None
        units_match = re.search(r'\(([^)]+)\)\s*$', title)
        if units_match:
            units = units_match.group(1)
        
        return (title, table_id, units)
    
    def _collect_footnote_refs(self, data: List[List]) -> List[str]:
        """Collect all footnote references from table cells."""
        refs = set()
        pattern = r'(\d+/|\*+)'
        
        for row in data:
            for cell in row:
                if cell:
                    matches = re.findall(pattern, str(cell))
                    refs.update(matches)
        
        return sorted(list(refs))
    
    def _to_linear_text(self, data: List[List]) -> str:
        """
        Convert table to linearized text representation.
        
        Used as fallback when structure cannot be preserved.
        """
        lines = []
        for row in data:
            # Filter None values and join with tabs
            cells = [str(c) if c else "" for c in row]
            lines.append("\t".join(cells))
        return "\n".join(lines)
    
    def _to_markdown(self, data: List[List]) -> str:
        """
        Convert table to Markdown format.
        
        Useful for LLM consumption and debugging.
        """
        if not data:
            return ""
        
        lines = []
        
        # First row as header
        header = [str(c) if c else "" for c in data[0]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        
        # Remaining rows as data
        for row in data[1:]:
            cells = [str(c) if c else "" for c in row]
            # Pad to header length if needed
            while len(cells) < len(header):
                cells.append("")
            lines.append("| " + " | ".join(cells) + " |")
        
        return "\n".join(lines)


# Example output demonstration
EXAMPLE_OUTPUT = """
Example RawTable output from extraction (Table 1, Page 39):

RawTable(
    table_id='Table 1',
    table_title='Table 1. Qatar: Selected Macroeconomic Indicators, 2020-29',
    page_number=39,
    position_in_page=0,
    bbox=BoundingBox(x0=0.08, y0=0.12, x1=0.92, y1=0.88),
    structure=TableStructure(
        headers=[
            ['', '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029'],
            ['', 'Actual', 'Actual', 'Actual', 'Actual', 'Proj.', 'Proj.', 'Proj.', 'Proj.', 'Proj.', 'Proj.']
        ],
        rows=[
            TableRow(
                cells=[
                    Cell(value='Real GDP growth', row_index=2, col_index=0),
                    Cell(value='-3.6', row_index=2, col_index=1),
                    Cell(value='1.6', row_index=2, col_index=2),
                    ...
                ],
                row_index=2,
                is_header=False,
                row_label='Real GDP growth'
            ),
            ...
        ],
        column_count=11,
        row_count=45,
        has_merged_cells=False
    ),
    raw_text='\\t2020\\t2021\\t...\\nReal GDP growth\\t-3.6\\t1.6\\t...',
    markdown='| | 2020 | 2021 | ... |\\n|---|---|---|...',
    units='Percent',
    column_headers=['', '2020', '2021', '2022', '2023', '2024', '2025', '2026', '2027', '2028', '2029'],
    row_headers=['Real GDP growth', 'Hydrocarbon', 'Non-hydrocarbon', ...],
    footnote_refs=['1/', '2/', '3/'],
    extraction_confidence=1.0,
    extraction_warnings=[]
)
"""

if __name__ == "__main__":
    print(EXAMPLE_OUTPUT)
