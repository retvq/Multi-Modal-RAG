"""
Figure Extractor

Extracts figures and charts from PDF pages using PyMuPDF.
Captures:
- Image data (saved to disk)
- Bounding boxes
- Captions (text near figures)
- Visible text labels (via OCR if needed)

IMPORTANT: This extractor identifies and extracts figures but does NOT:
- Interpret chart contents
- Summarize visual data
- Generate descriptions

Those tasks are for downstream vision-language processing.
"""

import fitz  # PyMuPDF
import re
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_path))

from src.ingestion.loaders.pdf_loader import PageInfo
from src.schemas.content_block import BoundingBox


# Try to import OCR, but make it optional
try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


@dataclass
class PanelInfo:
    """
    Information about a single panel within a multi-panel figure.
    
    Multi-panel figures (like Figure 1-5 in IMF reports) contain
    multiple independent charts arranged in a grid.
    """
    panel_index: int                    # Position in grid (0-indexed)
    panel_title: Optional[str] = None   # Title for this panel
    bbox: Optional[BoundingBox] = None  # Location within parent figure
    image_path: Optional[str] = None    # Path to extracted panel image


@dataclass
class RawFigure:
    """
    Raw extracted figure before normalization.
    
    Contains image data, metadata, and extracted text.
    """
    # Identity
    figure_id: Optional[str] = None     # e.g., "Figure 1", "Text Figure 8"
    figure_title: str = ""              # Full title text
    
    # Location
    page_number: int = 0
    position_in_page: int = 0
    bbox: Optional[BoundingBox] = None
    
    # Image data
    image_path: Optional[str] = None    # Path to saved image file
    image_bytes: Optional[bytes] = None # Raw image bytes (before saving)
    image_width: int = 0
    image_height: int = 0
    
    # Figure type classification
    figure_type: str = "UNKNOWN"        # LINE_CHART, BAR_CHART, TABLE, DIAGRAM, etc.
    is_multi_panel: bool = False
    panel_count: int = 1
    panels: List[PanelInfo] = field(default_factory=list)
    
    # Extracted text
    caption: str = ""                   # Caption/source text below figure
    extracted_labels: List[str] = field(default_factory=list)  # OCR text from image
    
    # Quality
    extraction_confidence: float = 1.0
    extraction_warnings: List[str] = field(default_factory=list)
    ocr_applied: bool = False


class FigureExtractor:
    """
    Extracts figures from PDF pages using PyMuPDF.
    
    Strategy:
    1. Find all images on the page
    2. Filter out small images (logos, icons)
    3. Look for captions/titles near each image
    4. Detect multi-panel layouts
    5. Extract visible text via OCR (optional)
    6. Save images to disk
    
    IMF-specific handling:
    - Figures often have "Figure N" or "Text Figure N" titles
    - Multi-panel figures common (2x3 grids)
    - Source attributions at bottom of figures
    """
    
    def __init__(
        self,
        output_dir: str,
        min_image_size: int = 100,      # Minimum dimension to consider
        run_ocr: bool = False,           # Whether to run OCR on figures
        extract_panels: bool = True      # Whether to detect multi-panel layouts
    ):
        """
        Initialize figure extractor.
        
        Args:
            output_dir: Directory to save extracted images
            min_image_size: Minimum width/height to extract (filters icons)
            run_ocr: Whether to run Tesseract OCR on figures
            extract_panels: Whether to detect and separate multi-panel figures
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_image_size = min_image_size
        self.run_ocr = run_ocr and HAS_TESSERACT
        self.extract_panels = extract_panels
    
    def extract(self, page: PageInfo, document_id: str) -> List[RawFigure]:
        """
        Extract all figures from a single page.
        
        Args:
            page: PageInfo from document loader
            document_id: For naming output files
            
        Returns:
            List of RawFigure objects
        """
        fitz_page = page.fitz_page
        page_width = page.width
        page_height = page.height
        
        raw_figures = []
        
        # Method 1: Extract embedded images
        image_list = fitz_page.get_images(full=True)
        
        for img_idx, img_info in enumerate(image_list):
            raw_figure = self._process_image(
                page=fitz_page,
                img_info=img_info,
                img_idx=img_idx,
                page_number=page.page_number,
                page_width=page_width,
                page_height=page_height,
                document_id=document_id
            )
            
            if raw_figure:
                raw_figures.append(raw_figure)
        
        # Method 2: Find figure regions by looking for "Figure" text
        # This catches vector graphics that aren't embedded as images
        text_figures = self._find_figure_regions(
            page=fitz_page,
            page_number=page.page_number,
            page_width=page_width,
            page_height=page_height,
            document_id=document_id,
            existing_figures=raw_figures
        )
        
        raw_figures.extend(text_figures)
        
        # Sort by position on page
        raw_figures.sort(key=lambda f: (f.bbox.y0 if f.bbox else 0, f.bbox.x0 if f.bbox else 0))
        
        # Re-number positions
        for i, fig in enumerate(raw_figures):
            fig.position_in_page = i
        
        return raw_figures
    
    def _process_image(
        self,
        page: fitz.Page,
        img_info: tuple,
        img_idx: int,
        page_number: int,
        page_width: float,
        page_height: float,
        document_id: str
    ) -> Optional[RawFigure]:
        """
        Process a single embedded image.
        """
        xref = img_info[0]
        
        try:
            # Extract image data
            base_image = page.parent.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Get image dimensions
            img_width = base_image.get("width", 0)
            img_height = base_image.get("height", 0)
            
            # Filter small images (likely icons/logos)
            if img_width < self.min_image_size or img_height < self.min_image_size:
                return None
            
            # Get image location on page
            img_rects = page.get_image_rects(xref)
            if not img_rects:
                return None
            
            rect = img_rects[0]
            bbox = BoundingBox(
                x0=rect.x0 / page_width,
                y0=rect.y0 / page_height,
                x1=rect.x1 / page_width,
                y1=rect.y1 / page_height
            )
            
            # Save image to disk
            img_ext = base_image.get("ext", "png")
            img_filename = f"{document_id}_p{page_number:03d}_fig{img_idx:02d}.{img_ext}"
            img_path = self.output_dir / img_filename
            
            with open(img_path, "wb") as f:
                f.write(image_bytes)
            
            # Look for caption/title near the image
            title, figure_id, caption = self._find_figure_text(
                page=page,
                img_rect=rect,
                page_height=page_height
            )
            
            # Detect if multi-panel
            is_multi_panel, panel_count = self._detect_multi_panel(
                img_width=img_width,
                img_height=img_height,
                title=title
            )
            
            # Run OCR if enabled
            extracted_labels = []
            ocr_applied = False
            ocr_confidence = 0.0
            if self.run_ocr:
                extracted_labels, ocr_confidence = self._run_ocr(image_bytes)
                ocr_applied = True
            
            # Factor OCR confidence into overall extraction confidence
            base_confidence = 1.0
            if ocr_applied and ocr_confidence > 0:
                # Blend: high OCR confidence boosts, low OCR confidence reduces slightly
                base_confidence = 0.8 + (ocr_confidence / 100.0) * 0.2
            
            return RawFigure(
                figure_id=figure_id,
                figure_title=title,
                page_number=page_number,
                position_in_page=img_idx,
                bbox=bbox,
                image_path=str(img_path),
                image_bytes=None,  # Don't keep in memory
                image_width=img_width,
                image_height=img_height,
                figure_type=self._classify_figure_type(title),
                is_multi_panel=is_multi_panel,
                panel_count=panel_count,
                caption=caption,
                extracted_labels=extracted_labels,
                extraction_confidence=base_confidence,
                extraction_warnings=[f"OCR confidence: {ocr_confidence:.1f}%"] if ocr_applied else [],
                ocr_applied=ocr_applied
            )
            
        except Exception as e:
            return RawFigure(
                page_number=page_number,
                position_in_page=img_idx,
                extraction_confidence=0.5,
                extraction_warnings=[f"Image extraction error: {str(e)}"]
            )
    
    def _find_figure_regions(
        self,
        page: fitz.Page,
        page_number: int,
        page_width: float,
        page_height: float,
        document_id: str,
        existing_figures: List[RawFigure]
    ) -> List[RawFigure]:
        """
        Find figure regions by looking for "Figure" titles in text.
        
        This catches vector graphics and charts that aren't embedded images.
        """
        figures = []
        
        # Search for figure title patterns
        text_instances = page.search_for("Figure")
        text_instances.extend(page.search_for("Text Figure"))
        
        for rect in text_instances:
            # Check if this region overlaps with already-extracted images
            overlaps = False
            for existing in existing_figures:
                if existing.bbox and self._rects_overlap(
                    (rect.x0/page_width, rect.y0/page_height, rect.x1/page_width, rect.y1/page_height),
                    (existing.bbox.x0, existing.bbox.y0, existing.bbox.x1, existing.bbox.y1)
                ):
                    overlaps = True
                    break
            
            if not overlaps:
                # Extract the title text
                title_rect = fitz.Rect(rect.x0, rect.y0, page_width * 0.9, rect.y1 + 20)
                title_text = page.get_textbox(title_rect).strip()
                
                # Extract figure ID
                figure_id = self._extract_figure_id(title_text)
                
                if figure_id:
                    # Estimate figure region (title + content below)
                    fig_rect = fitz.Rect(
                        rect.x0 - 10,
                        rect.y0,
                        page_width * 0.9,
                        min(rect.y1 + 300, page_height * 0.9)  # Estimate height
                    )
                    
                    # Render region as image
                    try:
                        clip = fig_rect
                        mat = fitz.Matrix(2, 2)  # 2x zoom for quality
                        pix = page.get_pixmap(matrix=mat, clip=clip)
                        
                        img_filename = f"{document_id}_p{page_number:03d}_figregion{len(figures):02d}.png"
                        img_path = self.output_dir / img_filename
                        pix.save(str(img_path))
                        
                        bbox = BoundingBox(
                            x0=clip.x0 / page_width,
                            y0=clip.y0 / page_height,
                            x1=clip.x1 / page_width,
                            y1=clip.y1 / page_height
                        )
                        
                        figures.append(RawFigure(
                            figure_id=figure_id,
                            figure_title=title_text,
                            page_number=page_number,
                            bbox=bbox,
                            image_path=str(img_path),
                            image_width=pix.width,
                            image_height=pix.height,
                            figure_type="VECTOR",
                            extraction_confidence=0.8,
                            extraction_warnings=["Rendered from vector graphics"]
                        ))
                    except Exception as e:
                        pass  # Skip if rendering fails
        
        return figures
    
    def _find_figure_text(
        self,
        page: fitz.Page,
        img_rect: fitz.Rect,
        page_height: float
    ) -> Tuple[str, Optional[str], str]:
        """
        Find figure title and caption near an image.
        
        Returns:
            (title, figure_id, caption)
        """
        title = ""
        figure_id = None
        caption = ""
        
        # Look above the image for title (within 50 points)
        title_rect = fitz.Rect(
            img_rect.x0 - 10,
            max(0, img_rect.y0 - 50),
            img_rect.x1 + 10,
            img_rect.y0
        )
        title_text = page.get_textbox(title_rect).strip()
        
        if title_text:
            title = title_text
            figure_id = self._extract_figure_id(title_text)
        
        # Look below the image for caption/source (within 30 points)
        caption_rect = fitz.Rect(
            img_rect.x0 - 10,
            img_rect.y1,
            img_rect.x1 + 10,
            min(page_height, img_rect.y1 + 30)
        )
        caption = page.get_textbox(caption_rect).strip()
        
        return (title, figure_id, caption)
    
    def _extract_figure_id(self, text: str) -> Optional[str]:
        """Extract figure ID like 'Figure 1' or 'Text Figure 8' from text."""
        patterns = [
            r'(Text\s+Figure\s+\d+)',
            r'(Figure\s+\d+[a-z]?)',
            r'(Figure\s+[IVX]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _detect_multi_panel(
        self,
        img_width: int,
        img_height: int,
        title: str
    ) -> Tuple[bool, int]:
        """
        Detect if figure is a multi-panel layout.
        
        Heuristics:
        - Wide images (aspect ratio > 1.5) are often multi-column
        - Title mentions multiple subjects
        - IMF main figures (Figure 1-5) are typically 2x3 = 6 panels
        """
        # Check if it's a main IMF figure (typically 6 panels)
        if re.search(r'^Figure\s+[1-5]\b', title, re.IGNORECASE):
            if img_width > 400 and img_height > 300:
                return (True, 6)
        
        # Check aspect ratio
        aspect_ratio = img_width / max(img_height, 1)
        if aspect_ratio > 2.0:
            # Very wide - likely 2+ columns
            return (True, 2)
        
        return (False, 1)
    
    def _classify_figure_type(self, title: str) -> str:
        """Classify figure type based on title keywords."""
        title_lower = title.lower()
        
        if any(word in title_lower for word in ["chart", "trend", "growth"]):
            return "LINE_CHART"
        elif any(word in title_lower for word in ["bar", "comparison"]):
            return "BAR_CHART"
        elif any(word in title_lower for word in ["pie", "share", "composition"]):
            return "PIE_CHART"
        elif any(word in title_lower for word in ["map"]):
            return "MAP"
        elif any(word in title_lower for word in ["diagram", "flow", "process"]):
            return "DIAGRAM"
        else:
            return "CHART"  # Default for IMF figures
    
    def _run_ocr(self, image_bytes: bytes, min_confidence: float = 60.0) -> Tuple[List[str], float]:
        """
        Run OCR on image to extract visible text labels with confidence filtering.
        
        Args:
            image_bytes: Raw image bytes
            min_confidence: Minimum confidence threshold (0-100) for including text
        
        Returns:
            (extracted_labels, average_confidence)
        """
        if not HAS_TESSERACT:
            return [], 0.0
        
        try:
            import io
            import pandas as pd
            
            img = Image.open(io.BytesIO(image_bytes))
            
            # Use image_to_data for word-level confidence scores
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            # Filter by confidence threshold
            high_confidence_words = []
            confidences = []
            
            for i, conf in enumerate(ocr_data['conf']):
                # conf is -1 for non-word elements
                if conf >= min_confidence:
                    word = ocr_data['text'][i].strip()
                    if word:  # Non-empty text
                        high_confidence_words.append(word)
                        confidences.append(conf)
            
            # Group into lines (consecutive words on same line)
            lines = []
            current_line = []
            prev_line_num = -1
            
            for i, conf in enumerate(ocr_data['conf']):
                if conf >= min_confidence:
                    word = ocr_data['text'][i].strip()
                    line_num = ocr_data['line_num'][i]
                    
                    if word:
                        if line_num != prev_line_num and current_line:
                            lines.append(' '.join(current_line))
                            current_line = []
                        current_line.append(word)
                        prev_line_num = line_num
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Calculate average confidence
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            # Detect axis labels (common chart patterns)
            axis_labels = self._detect_axis_labels(lines)
            
            # Combine unique labels
            all_labels = list(set(lines + axis_labels))
            
            return all_labels, avg_confidence
            
        except Exception as e:
            # Fallback to simple OCR without confidence
            try:
                import io
                img = Image.open(io.BytesIO(image_bytes))
                text = pytesseract.image_to_string(img)
                labels = [line.strip() for line in text.split('\n') if line.strip()]
                return labels, 50.0  # Default medium confidence
            except Exception:
                return [], 0.0
    
    def _detect_axis_labels(self, lines: List[str]) -> List[str]:
        """
        Detect common axis labels and chart annotations.
        
        Patterns:
        - Years (2019, 2020, etc.)
        - Percentages (10%, -5%)
        - Months (Jan, Feb, etc.)
        - Numeric scales (0, 50, 100)
        """
        axis_labels = []
        
        for line in lines:
            # Year patterns
            if re.match(r'^20\d{2}$', line.strip()):
                axis_labels.append(line.strip())
            # Percentage patterns
            elif re.match(r'^-?\d+\.?\d*%?$', line.strip()):
                axis_labels.append(line.strip())
            # Month abbreviations
            elif line.strip().lower() in ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                           'jul', 'aug', 'sep', 'oct', 'nov', 'dec']:
                axis_labels.append(line.strip())
        
        return axis_labels
    
    def _rects_overlap(self, r1: tuple, r2: tuple) -> bool:
        """Check if two rectangles overlap."""
        return not (r1[2] < r2[0] or r2[2] < r1[0] or r1[3] < r2[1] or r2[3] < r1[1])


# Example output
EXAMPLE_OUTPUT = """
Example RawFigure for a multi-panel figure (Figure 1, Page 34):

RawFigure(
    figure_id='Figure 1',
    figure_title='Figure 1. Qatar: Real Sector Developments',
    page_number=34,
    position_in_page=0,
    bbox=BoundingBox(x0=0.08, y0=0.10, x1=0.92, y1=0.85),
    image_path='outputs/figures/qatar_p034_fig00.png',
    image_bytes=None,
    image_width=1200,
    image_height=900,
    figure_type='CHART',
    is_multi_panel=True,
    panel_count=6,
    panels=[
        PanelInfo(panel_index=0, panel_title='Real GDP Growth'),
        PanelInfo(panel_index=1, panel_title='Non-hydrocarbon Growth'),
        PanelInfo(panel_index=2, panel_title='PMI'),
        PanelInfo(panel_index=3, panel_title='Tourism'),
        PanelInfo(panel_index=4, panel_title='Construction'),
        PanelInfo(panel_index=5, panel_title='Employment'),
    ],
    caption='Sources: Haver Analytics, QCB, and IMF staff calculations.',
    extracted_labels=['2019', '2020', '2021', '2022', '2023', '2024', 'Percent'],
    extraction_confidence=1.0,
    extraction_warnings=[],
    ocr_applied=True
)
"""

if __name__ == "__main__":
    print(EXAMPLE_OUTPUT)
