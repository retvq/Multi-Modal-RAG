"""
Table Ingestion Test Script

Demonstrates the table ingestion pipeline on the Qatar IMF document.
Runs extraction and normalization on table pages.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.ingestion.extractors.table_extractor import TableExtractor
from src.ingestion.normalizers.table_normalizer import TableNormalizer


def main():
    """Run table ingestion on Qatar document."""
    
    # Document path
    doc_path = Path(__file__).parent.parent / "data" / "qatar_test_doc.pdf"
    
    if not doc_path.exists():
        print(f"ERROR: Document not found at {doc_path}")
        return
    
    print("=" * 70)
    print("TABLE INGESTION TEST")
    print("=" * 70)
    
    # Step 1: Initialize extractor
    print("\n[1] Initializing table extractor...")
    
    with TableExtractor(str(doc_path)) as extractor:
        
        # Step 2: Extract tables from page 39 (Table 1: Macro Indicators)
        print("\n[2] Extracting tables from page 39...")
        
        raw_tables = extractor.extract_from_page(39)
        print(f"    Found {len(raw_tables)} tables")
        
        if not raw_tables:
            # Try page 40 as fallback
            print("    Trying page 40...")
            raw_tables = extractor.extract_from_page(40)
            print(f"    Found {len(raw_tables)} tables")
        
        if not raw_tables:
            print("    No tables found on pages 39-40. Trying page 42...")
            raw_tables = extractor.extract_from_page(42)
            print(f"    Found {len(raw_tables)} tables")
        
        if raw_tables:
            # Show first table details
            raw = raw_tables[0]
            print(f"\n    --- First Table ---")
            print(f"    Table ID: {raw.table_id}")
            print(f"    Title: {raw.table_title[:80]}..." if len(raw.table_title) > 80 else f"    Title: {raw.table_title}")
            print(f"    Page: {raw.page_number}")
            print(f"    Position: {raw.position_in_page}")
            print(f"    Units: {raw.units}")
            
            if raw.bbox:
                print(f"    BBox: ({raw.bbox.x0:.2f}, {raw.bbox.y0:.2f}) - ({raw.bbox.x1:.2f}, {raw.bbox.y1:.2f})")
            
            if raw.structure:
                print(f"\n    Structure:")
                print(f"      Columns: {raw.structure.column_count}")
                print(f"      Rows: {raw.structure.row_count}")
                print(f"      Header rows: {len(raw.structure.headers)}")
                print(f"      Has merged cells: {raw.structure.has_merged_cells}")
                
                if raw.structure.headers:
                    print(f"\n    Headers (nested):")
                    for i, header_row in enumerate(raw.structure.headers):
                        preview = header_row[:5]
                        print(f"      Level {i}: {preview}{'...' if len(header_row) > 5 else ''}")
            
            print(f"\n    Column headers: {raw.column_headers[:5]}{'...' if len(raw.column_headers) > 5 else ''}")
            print(f"    Row headers (first 5): {raw.row_headers[:5]}")
            print(f"    Footnote refs: {raw.footnote_refs}")
            print(f"    Confidence: {raw.extraction_confidence}")
            print(f"    Warnings: {raw.extraction_warnings}")
            
            # Show markdown preview
            print(f"\n    Markdown preview (first 500 chars):")
            print("-" * 50)
            print(raw.markdown[:500] if len(raw.markdown) > 500 else raw.markdown)
            print("-" * 50)
        
        # Step 3: Normalize to ContentBlocks
        print("\n\n[3] Normalizing to ContentBlocks...")
        
        if raw_tables:
            normalizer = TableNormalizer(document_id="qatar_test_doc")
            
            section_hierarchy = ["Staff Report", "Statistical Appendix"]
            content_blocks = normalizer.normalize_batch(
                raw_tables=raw_tables,
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
                print(f"    citation_full: {block.citation_full[:80]}..." if len(block.citation_full) > 80 else f"    citation_full: {block.citation_full}")
                print(f"    content_length: {block.content_length}")
                print(f"    extraction_confidence: {block.extraction_confidence}")
                
                # Access table metadata
                if hasattr(block, '_table_metadata'):
                    tm = block._table_metadata
                    print(f"\n    Table Metadata:")
                    print(f"      table_id: {tm.table_id}")
                    print(f"      column_count: {tm.column_count}")
                    print(f"      row_count: {tm.row_count}")
                    print(f"      has_nested_headers: {tm.has_nested_headers}")
                    print(f"      has_merged_cells: {tm.has_merged_cells}")
                    print(f"      units: {tm.units}")
                    print(f"      footnote_refs: {tm.footnote_refs}")
                
                # Validate
                is_valid, errors = block.validate()
                print(f"\n    valid: {is_valid}")
                if errors:
                    print(f"    errors: {errors}")
        
        # Step 4: Try multiple pages to find all tables
        print("\n\n[4] Scanning all table pages (39-45)...")
        
        all_tables = []
        for page_num in range(39, 46):
            tables = extractor.extract_from_page(page_num)
            if tables:
                print(f"    Page {page_num}: {len(tables)} table(s)")
                for t in tables:
                    all_tables.append(t)
                    print(f"      - {t.table_id or 'Untitled'}: {t.structure.column_count if t.structure else '?'} cols, {t.structure.row_count if t.structure else '?'} rows")
        
        print(f"\n    Total tables found: {len(all_tables)}")
    
    print("\n" + "=" * 70)
    print("TABLE INGESTION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
