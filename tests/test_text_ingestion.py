"""
Text Ingestion Test Script

Demonstrates the text ingestion pipeline on the Qatar IMF document.
Runs extraction and normalization on a sample page.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.ingestion.loaders.pdf_loader import PDFLoader
from src.ingestion.extractors.text_extractor import TextExtractor
from src.ingestion.normalizers.text_normalizer import TextNormalizer


def main():
    """Run text ingestion on Qatar document."""
    
    # Document path
    doc_path = Path(__file__).parent.parent / "data" / "qatar_test_doc.pdf"
    
    if not doc_path.exists():
        print(f"ERROR: Document not found at {doc_path}")
        return
    
    print("=" * 60)
    print("TEXT INGESTION TEST")
    print("=" * 60)
    
    # Step 1: Load document
    print("\n[1] Loading document...")
    try:
        doc = PDFLoader.load(str(doc_path))
        print(f"    Document ID: {doc.document_id}")
        print(f"    Total Pages: {doc.total_pages}")
    except Exception as e:
        print(f"    ERROR: {e}")
        return
    
    # Step 2: Extract text from page 5 (has KEY ISSUES section)
    print("\n[2] Extracting text from page 5...")
    extractor = TextExtractor()
    page = doc.get_page(5)
    
    if not page:
        print("    ERROR: Page 5 not found")
        return
    
    raw_blocks = extractor.extract(page)
    print(f"    Extracted {len(raw_blocks)} raw text blocks")
    
    # Show first 3 raw blocks
    print("\n    Sample raw blocks:")
    for i, block in enumerate(raw_blocks[:3]):
        print(f"\n    --- Block {i} ---")
        print(f"    Type: {block.text_type.value}")
        print(f"    Heading Level: {block.heading_level}")
        print(f"    Para Number: {block.paragraph_number}")
        print(f"    Is Header/Footer: {block.is_header_footer}")
        print(f"    Font Size: {block.font_size:.1f}")
        print(f"    Bold: {block.is_bold}")
        print(f"    Text: {block.text[:80]}...")
    
    # Step 3: Normalize to ContentBlocks
    print("\n\n[3] Normalizing to ContentBlocks...")
    normalizer = TextNormalizer(document_id=doc.document_id)
    
    # Simple section hierarchy for demo
    section_hierarchy = ["Staff Report", "Key Issues"]
    
    content_blocks = normalizer.normalize_batch(
        raw_blocks=[b for b in raw_blocks if not b.is_header_footer],
        section_hierarchy=section_hierarchy
    )
    print(f"    Created {len(content_blocks)} ContentBlocks")
    
    # Show first 3 content blocks with full detail
    print("\n    Sample ContentBlocks:")
    for i, block in enumerate(content_blocks[:3]):
        print(f"\n    === ContentBlock {i} ===")
        print(f"    block_id: {block.block_id}")
        print(f"    modality: {block.modality.value}")
        print(f"    page_number: {block.page_number}")
        print(f"    section_hierarchy: {block.section_hierarchy}")
        print(f"    citation_short: {block.citation_short}")
        print(f"    citation_section: {block.citation_section}")
        
        if block.text_metadata:
            print(f"    text_type: {block.text_metadata.text_type.value}")
            print(f"    paragraph_number: {block.text_metadata.paragraph_number}")
            print(f"    heading_level: {block.text_metadata.heading_level}")
            print(f"    has_cross_references: {block.text_metadata.has_cross_references}")
        
        if block.bounding_box:
            bb = block.bounding_box
            print(f"    bounding_box: ({bb.x0:.2f}, {bb.y0:.2f}) - ({bb.x1:.2f}, {bb.y1:.2f})")
        
        print(f"    content_length: {block.content_length}")
        print(f"    extraction_confidence: {block.extraction_confidence}")
        print(f"    content: {block.content[:100]}...")
        
        # Validate
        is_valid, errors = block.validate()
        print(f"    valid: {is_valid}")
        if errors:
            print(f"    errors: {errors}")
    
    # Step 4: Summary statistics
    print("\n\n[4] Summary Statistics")
    print("=" * 60)
    
    # Count by text type
    type_counts = {}
    for block in content_blocks:
        if block.text_metadata:
            text_type = block.text_metadata.text_type.value
            type_counts[text_type] = type_counts.get(text_type, 0) + 1
    
    print("\n    Blocks by type:")
    for text_type, count in sorted(type_counts.items()):
        print(f"      {text_type}: {count}")
    
    # Count paragraphs with numbers
    numbered = sum(1 for b in content_blocks 
                   if b.text_metadata and b.text_metadata.paragraph_number)
    print(f"\n    Numbered paragraphs: {numbered}")
    
    # Count with cross-references
    with_refs = sum(1 for b in content_blocks 
                    if b.text_metadata and b.text_metadata.has_cross_references)
    print(f"    With cross-references: {with_refs}")
    
    # Validation summary
    valid_count = sum(1 for b in content_blocks if b.validate()[0])
    print(f"\n    Valid blocks: {valid_count}/{len(content_blocks)}")
    
    # Cleanup
    doc.close()
    
    print("\n" + "=" * 60)
    print("TEXT INGESTION TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
