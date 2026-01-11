"""
Validation Test Script

Demonstrates schema validation with both valid and invalid blocks.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from src.schemas.content_block import ContentBlock, Modality, TextMetadata, TextType, BoundingBox
from src.schemas.relationship import Relationship, RelationshipType
from src.ingestion.validators.block_validator import BlockValidator, ValidationError
from src.ingestion.validators.relationship_validator import RelationshipValidator


def main():
    """Test schema validation."""
    
    print("=" * 70)
    print("SCHEMA VALIDATION TEST")
    print("=" * 70)
    
    # === Test 1: Valid block ===
    print("\n[1] Testing VALID block...")
    
    valid_block = ContentBlock(
        block_id="qatar_p005_text_abc12345_001",
        document_id="qatar_test_doc",
        modality=Modality.TEXT,
        content="This is a valid paragraph with some content.",
        page_number=5,
        section_hierarchy=["Staff Report", "Context"],
        bounding_box=BoundingBox(x0=0.1, y0=0.2, x1=0.9, y1=0.3),
        text_metadata=TextMetadata(
            paragraph_number=1,
            text_type=TextType.BODY,
            has_cross_references=False
        )
    )
    
    validator = BlockValidator(strict=False)
    result = validator.validate(valid_block)
    
    print(f"    {result}")
    print(f"    Errors: {len(result.errors)}")
    print(f"    Warnings: {len(result.warnings)}")
    
    # === Test 2: Invalid block (multiple errors) ===
    print("\n[2] Testing INVALID block (multiple errors)...")
    
    invalid_block = ContentBlock(
        block_id="",                        # ERROR: empty
        document_id="qatar_test_doc",
        modality=Modality.TEXT,
        content="",                         # ERROR: empty for TEXT
        page_number=0,                      # ERROR: must be >= 1
        section_hierarchy=[],               # ERROR: must not be empty
    )
    
    result = validator.validate(invalid_block)
    
    print(f"    {result}")
    print(f"    Errors ({len(result.errors)}):")
    for err in result.errors:
        print(f"      - {err.field}: {err.message}")
    print(f"    Warnings ({len(result.warnings)}):")
    for warn in result.warnings:
        print(f"      - {warn.field}: {warn.message}")
    
    # === Test 3: Fail-fast mode ===
    print("\n[3] Testing fail-fast mode (strict=True)...")
    
    strict_validator = BlockValidator(strict=True)
    
    try:
        strict_validator.validate(invalid_block)
        print("    ERROR: Should have raised ValidationError")
    except ValidationError as e:
        print(f"    ✓ Caught ValidationError: {e.result.errors[0].field}")
    
    # === Test 4: Batch validation ===
    print("\n[4] Testing batch validation...")
    
    blocks = [valid_block, invalid_block]
    
    results = validator.validate_batch(blocks, fail_fast=False)
    
    valid_count = sum(1 for r in results if r.is_valid)
    invalid_count = len(results) - valid_count
    
    print(f"    Validated {len(results)} blocks")
    print(f"    Valid: {valid_count}")
    print(f"    Invalid: {invalid_count}")
    
    # === Test 5: Relationship validation ===
    print("\n[5] Testing relationship validation...")
    
    known_ids = {
        "qatar_p005_text_abc12345_001",
        "qatar_p039_table_xyz99999_000"
    }
    
    rel_validator = RelationshipValidator(known_ids, strict=False)
    
    # Valid relationship
    valid_rel = Relationship(
        relationship_id="xref_001",
        source_block_id="qatar_p005_text_abc12345_001",
        target_block_id="qatar_p039_table_xyz99999_000",
        relationship_type=RelationshipType.CROSS_REFERENCE,
        reference_text="Table 1"
    )
    
    result = rel_validator.validate(valid_rel)
    print(f"    Valid relationship: {result}")
    
    # Invalid relationship (target not found)
    invalid_rel = Relationship(
        relationship_id="xref_002",
        source_block_id="qatar_p005_text_abc12345_001",
        target_block_id="qatar_p999_missing_000",  # Not in known_ids
        relationship_type=RelationshipType.CROSS_REFERENCE,
        reference_text="Table 99"
    )
    
    result = rel_validator.validate(invalid_rel)
    print(f"    Invalid relationship: {result}")
    
    # Self-reference (invalid)
    self_ref = Relationship(
        relationship_id="xref_003",
        source_block_id="qatar_p005_text_abc12345_001",
        target_block_id="qatar_p005_text_abc12345_001",  # Same as source
        relationship_type=RelationshipType.CROSS_REFERENCE,
        reference_text="itself"
    )
    
    result = rel_validator.validate(self_ref)
    print(f"    Self-reference: {result}")
    
    print("\n" + "=" * 70)
    print("VALIDATION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
