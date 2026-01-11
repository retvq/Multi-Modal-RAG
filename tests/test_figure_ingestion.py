"""
Figure Ingestion Test Script

Demonstrates the figure ingestion pipeline on the Qatar IMF document.
Extracts figures from pages 34-38 (main figures).
"""

import sys
import os
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.extractors.figure_extractor import FigureExtractor
from src.ingestion.normalizers.figure_normalizer import FigureNormalizer


def main():
    """Run figure ingestion on Qatar document."""
    
    # Document path
    doc_path = Path(__file__).parent.parent / "data" / "qatar_test_doc.pdf"
    output_dir = Path(__file__).parent / "outputs" / "figures"
    
    if not doc_path.exists():
        print(f"ERROR: Document not found at {doc_path}")
        return
    
    print("=" * 70)
    print("FIGURE INGESTION TEST")
    print("=" * 70)
    
    # Step 1: Load document
    print("\n[1] Loading document...")
    try:
        doc = PDFLoader.load(str(doc_path))
        print(f"    Document ID: {doc.document_id}")
        print(f"    Output dir: {output_dir}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return
    
    # Step 2: Extract figures from pages 34-38 (main figures)
    print("\n[2] Extracting figures...")
    
    extractor = FigureExtractor(
        output_dir=str(output_dir),
        min_image_size=100,
        run_ocr=False,  # Skip OCR for speed
        extract_panels=True
    )
    
    all_figures = []
    
    for page_num in [34, 35, 36, 37, 38]:
        page = doc.get_page(page_num)
        if not page:
            continue
        
        figures = extractor.extract(page, doc.document_id)
        print(f"    Page {page_num}: {len(figures)} figure(s)")
        
        for fig in figures:
            all_figures.append(fig)
            print(f"      - {fig.figure_id or 'Untitled'}: {fig.image_width}x{fig.image_height}", end="")
            if fig.is_multi_panel:
                print(f" (multi-panel: {fig.panel_count} panels)", end="")
            print()
    
    print(f"\n    Total figures found: {len(all_figures)}")
    
    # Step 3: Show first figure details
    if all_figures:
        raw = all_figures[0]
        print(f"\n    --- First Figure Details ---")
        print(f"    Figure ID: {raw.figure_id}")
        print(f"    Title: {raw.figure_title[:80]}..." if len(raw.figure_title) > 80 else f"    Title: {raw.figure_title}")
        print(f"    Page: {raw.page_number}")
        print(f"    Type: {raw.figure_type}")
        print(f"    Multi-panel: {raw.is_multi_panel} ({raw.panel_count} panels)")
        print(f"    Image: {raw.image_width}x{raw.image_height}")
        print(f"    Image path: {raw.image_path}")
        print(f"    Caption: {raw.caption[:80]}..." if len(raw.caption) > 80 else f"    Caption: {raw.caption}")
        
        if raw.bbox:
            print(f"    BBox: ({raw.bbox.x0:.2f}, {raw.bbox.y0:.2f}) - ({raw.bbox.x1:.2f}, {raw.bbox.y1:.2f})")
        
        print(f"    OCR applied: {raw.ocr_applied}")
        print(f"    Labels: {raw.extracted_labels[:5]}" if raw.extracted_labels else "    Labels: []")
        print(f"    Confidence: {raw.extraction_confidence}")
        print(f"    Warnings: {raw.extraction_warnings}")
    
    # Step 4: Normalize to ContentBlocks
    print("\n\n[3] Normalizing to ContentBlocks...")
    
    if all_figures:
        normalizer = FigureNormalizer(document_id=doc.document_id)
        
        section_hierarchy = ["Staff Report", "Figures"]
        content_blocks = normalizer.normalize_batch(
            raw_figures=all_figures,
            section_hierarchy=section_hierarchy
        )
        print(f"    Created {len(content_blocks)} ContentBlocks")
        
        # Show first content block
        if content_blocks:
            block = content_blocks[0]
            print(f"\n    === ContentBlock ===")
            print(f"    block_id: {block.block_id}")
            print(f"    modality: {block.modality.value}")
            print(f"    page_number: {block.page_number}")
            print(f"    section_hierarchy: {block.section_hierarchy}")
            print(f"    citation_short: {block.citation_short}")
            print(f"    content_length: {block.content_length}")
            print(f"    extraction_confidence: {block.extraction_confidence}")
            print(f"    extraction_warnings: {block.extraction_warnings}")
            
            # Access figure metadata
            if hasattr(block, '_figure_metadata'):
                fm = block._figure_metadata
                print(f"\n    Figure Metadata:")
                print(f"      figure_id: {fm.figure_id}")
                print(f"      figure_type: {fm.figure_type}")
                print(f"      is_multi_panel: {fm.is_multi_panel}")
                print(f"      panel_count: {fm.panel_count}")
                print(f"      image_path: {fm.image_path}")
                print(f"      caption: {fm.caption[:50]}..." if len(fm.caption) > 50 else f"      caption: {fm.caption}")
            
            print(f"\n    Content (first 200 chars):")
            print(f"    {block.content[:200]}...")
            
            # Validate
            is_valid, errors = block.validate()
            print(f"\n    valid: {is_valid}")
            if errors:
                print(f"    errors: {errors}")
    
    # Step 5: List saved images
    print("\n\n[4] Saved images:")
    if output_dir.exists():
        images = list(output_dir.glob("*.png")) + list(output_dir.glob("*.jpg"))
        for img in images[:10]:
            size = img.stat().st_size / 1024
            print(f"    {img.name} ({size:.1f} KB)")
        if len(images) > 10:
            print(f"    ... and {len(images) - 10} more")
    
    # Cleanup
    doc.close()
    
    print("\n" + "=" * 70)
    print("FIGURE INGESTION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
