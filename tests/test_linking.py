"""
Footnote and Cross-Reference Linking Test Script

Demonstrates:
- Footnote extraction from pages
- Cross-reference detection in text
- Relationship record creation
- Error handling for unresolved references
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.extractors.text_extractor import TextExtractor
from src.ingestion.extractors.table_extractor import TableExtractor
from src.ingestion.extractors.footnote_extractor import FootnoteExtractor, normalize_footnote_to_block
from src.ingestion.normalizers.text_normalizer import TextNormalizer
from src.ingestion.normalizers.table_normalizer import TableNormalizer
from src.ingestion.linkers.cross_reference_linker import CrossReferenceLinker
from src.ingestion.linkers.footnote_linker import FootnoteLinker


def main():
    """Test footnote and cross-reference linking."""
    
    doc_path = Path(__file__).parent.parent / "data" / "qatar_test_doc.pdf"
    
    if not doc_path.exists():
        print(f"ERROR: Document not found at {doc_path}")
        return
    
    print("=" * 70)
    print("FOOTNOTE AND CROSS-REFERENCE LINKING TEST")
    print("=" * 70)
    
    # Load document
    print("\n[1] Loading document...")
    doc = PDFLoader.load(str(doc_path))
    print(f"    Document ID: {doc.document_id}")
    
    # Step 1: Extract footnotes from table pages (39-45)
    print("\n[2] Extracting footnotes from pages 39-43...")
    
    fn_extractor = FootnoteExtractor()
    all_footnotes = []
    all_markers = []
    
    for page_num in [39, 40, 41, 42, 43]:
        page = doc.get_page(page_num)
        if page:
            footnotes, markers = fn_extractor.extract(page)
            print(f"    Page {page_num}: {len(footnotes)} footnotes, {len(markers)} markers")
            
            for fn in footnotes:
                all_footnotes.append(fn)
                print(f"      - [{fn.marker}] {fn.content[:50]}..." if len(fn.content) > 50 else f"      - [{fn.marker}] {fn.content}")
            
            all_markers.extend(markers)
    
    print(f"\n    Total: {len(all_footnotes)} footnotes, {len(all_markers)} markers")
    
    # Convert footnotes to ContentBlocks
    print("\n[3] Converting footnotes to ContentBlocks...")
    footnote_blocks = []
    for fn in all_footnotes:
        block = normalize_footnote_to_block(fn, doc.document_id, ["Document", "Footnotes"])
        footnote_blocks.append(block)
    
    if footnote_blocks:
        block = footnote_blocks[0]
        print(f"\n    Example ContentBlock:")
        print(f"    block_id: {block.block_id}")
        print(f"    modality: {block.modality.value}")
        print(f"    citation_short: {block.citation_short}")
        print(f"    content: {block.content[:80]}...")
    
    # Step 2: Extract text and tables for cross-reference detection
    print("\n\n[4] Extracting text blocks for cross-reference detection...")
    
    text_extractor = TextExtractor()
    text_normalizer = TextNormalizer(doc.document_id)
    
    all_blocks = list(footnote_blocks)  # Start with footnotes
    
    # Extract text from pages with cross-references (5-15)
    for page_num in [5, 6, 7, 8, 10, 12]:
        page = doc.get_page(page_num)
        if page:
            raw_blocks = text_extractor.extract(page)
            normalized = text_normalizer.normalize_batch(
                raw_blocks=[b for b in raw_blocks if not b.is_header_footer],
                section_hierarchy=["Staff Report"]
            )
            all_blocks.extend(normalized)
    
    print(f"    Extracted {len(all_blocks)} total blocks")
    
    # Add some table blocks for cross-reference targets
    print("\n[5] Adding table blocks as cross-reference targets...")
    
    with TableExtractor(str(doc_path)) as table_extractor:
        table_normalizer = TableNormalizer(doc.document_id)
        for page_num in [39, 40, 41]:
            raw_tables = table_extractor.extract_from_page(page_num)
            for raw in raw_tables:
                block = table_normalizer.normalize(raw, ["Staff Report", "Tables"])
                all_blocks.append(block)
    
    print(f"    Total blocks for linking: {len(all_blocks)}")
    
    # Step 3: Run cross-reference linker
    print("\n\n[6] Running cross-reference linker...")
    
    xref_linker = CrossReferenceLinker()
    xref_relationships, unresolved = xref_linker.link(all_blocks)
    
    print(f"    Created {len(xref_relationships)} cross-reference relationships")
    print(f"    Unresolved references: {len(unresolved)}")
    
    if xref_relationships:
        print(f"\n    Example Relationships:")
        for rel in xref_relationships[:3]:
            print(f"      {rel.reference_text}: {rel.source_block_id[:30]}... → {rel.target_block_id[:30]}...")
    
    if unresolved:
        print(f"\n    Unresolved (first 3):")
        for ur in unresolved[:3]:
            print(f"      '{ur.reference_text}' on page {ur.page_number}: {ur.reason}")
    
    # Step 4: Run footnote linker
    print("\n\n[7] Running footnote linker...")
    
    fn_linker = FootnoteLinker()
    fn_relationships, orphan_fns, orphan_markers = fn_linker.link(all_blocks)
    
    print(f"    Created {len(fn_relationships)} footnote relationships")
    print(f"    Orphan footnotes: {len(orphan_fns)}")
    print(f"    Orphan markers: {len(orphan_markers)}")
    
    if fn_relationships:
        print(f"\n    Example Footnote Relationships:")
        for rel in fn_relationships[:3]:
            print(f"      [{rel.reference_text}]: {rel.source_block_id[:25]}... → {rel.target_block_id[:25]}...")
    
    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total_rels = len(xref_relationships) + len(fn_relationships)
    print(f"\n    Total ContentBlocks: {len(all_blocks)}")
    print(f"    Total Relationships: {total_rels}")
    print(f"      - Cross-references: {len(xref_relationships)}")
    print(f"      - Footnote refs: {len(fn_relationships)}")
    print(f"\n    Unresolved:")
    print(f"      - Cross-refs: {len(unresolved)}")
    print(f"      - Orphan footnotes: {len(orphan_fns)}")
    print(f"      - Orphan markers: {len(orphan_markers)}")
    
    doc.close()
    
    print("\n" + "=" * 70)
    print("LINKING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
