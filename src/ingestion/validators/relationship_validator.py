"""
Relationship Validator

Validates Relationship records against schema requirements.

Validation:
1. Required fields (relationship_id, source_block_id, target_block_id, type)
2. Block existence (both endpoints must exist in document)
3. Self-reference prevention (source != target)
4. Relationship type validity
"""

from dataclasses import dataclass, field
from typing import List, Set, Optional
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_path))

from src.schemas.relationship import Relationship, RelationshipType
from src.ingestion.validators.block_validator import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    ValidationError,
)


class RelationshipValidator:
    """
    Validates Relationship records.
    
    Usage:
        validator = RelationshipValidator(known_block_ids)
        result = validator.validate(relationship)
    """
    
    def __init__(self, known_block_ids: Set[str], strict: bool = True):
        """
        Initialize validator.
        
        Args:
            known_block_ids: Set of all valid block IDs in the document
            strict: If True, raises ValidationError on invalid relationships
        """
        self.known_block_ids = known_block_ids
        self.strict = strict
    
    def validate(self, relationship: Relationship) -> ValidationResult:
        """
        Validate a single Relationship.
        
        Args:
            relationship: Relationship to validate
            
        Returns:
            ValidationResult with all issues
            
        Raises:
            ValidationError: If strict mode and relationship is invalid
        """
        issues = []
        
        # === Required Fields ===
        if not relationship.relationship_id:
            issues.append(ValidationIssue(
                field="relationship_id",
                message="Required field is empty",
                severity=ValidationSeverity.ERROR
            ))
        
        if not relationship.source_block_id:
            issues.append(ValidationIssue(
                field="source_block_id",
                message="Required field is empty",
                severity=ValidationSeverity.ERROR
            ))
        
        if not relationship.target_block_id:
            issues.append(ValidationIssue(
                field="target_block_id",
                message="Required field is empty",
                severity=ValidationSeverity.ERROR
            ))
        
        if not relationship.relationship_type:
            issues.append(ValidationIssue(
                field="relationship_type",
                message="Required field is empty",
                severity=ValidationSeverity.ERROR
            ))
        elif not isinstance(relationship.relationship_type, RelationshipType):
            issues.append(ValidationIssue(
                field="relationship_type",
                message=f"Invalid type: {type(relationship.relationship_type)}",
                severity=ValidationSeverity.ERROR,
                value=str(relationship.relationship_type)
            ))
        
        # === Block Existence ===
        if relationship.source_block_id and relationship.source_block_id not in self.known_block_ids:
            issues.append(ValidationIssue(
                field="source_block_id",
                message=f"Block not found: {relationship.source_block_id}",
                severity=ValidationSeverity.ERROR,
                value=relationship.source_block_id
            ))
        
        if relationship.target_block_id and relationship.target_block_id not in self.known_block_ids:
            issues.append(ValidationIssue(
                field="target_block_id",
                message=f"Block not found: {relationship.target_block_id}",
                severity=ValidationSeverity.ERROR,
                value=relationship.target_block_id
            ))
        
        # === Self-Reference ===
        if relationship.source_block_id and relationship.source_block_id == relationship.target_block_id:
            issues.append(ValidationIssue(
                field="source_block_id/target_block_id",
                message="Self-reference not allowed",
                severity=ValidationSeverity.ERROR,
                value=relationship.source_block_id
            ))
        
        # Build result
        has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
        result = ValidationResult(
            block_id=relationship.relationship_id or "<no_id>",
            is_valid=not has_errors,
            issues=issues
        )
        
        if self.strict and not result.is_valid:
            raise ValidationError(result)
        
        return result
    
    def validate_batch(
        self,
        relationships: List[Relationship],
        fail_fast: bool = True
    ) -> List[ValidationResult]:
        """
        Validate multiple Relationships.
        
        Args:
            relationships: List of Relationships
            fail_fast: If True, stop on first invalid relationship
            
        Returns:
            List of ValidationResults
        """
        results = []
        
        for rel in relationships:
            try:
                result = self.validate(rel)
                results.append(result)
            except ValidationError as e:
                if fail_fast:
                    raise
                results.append(e.result)
        
        return results


EXAMPLE_OUTPUT = """
Example relationship validation:

# Valid relationship
valid_rel = Relationship(
    relationship_id='xref_001',
    source_block_id='qatar_p005_text_xxx_001',
    target_block_id='qatar_p039_table_yyy_000',
    relationship_type=RelationshipType.CROSS_REFERENCE,
    reference_text='Table 1'
)

known_ids = {'qatar_p005_text_xxx_001', 'qatar_p039_table_yyy_000'}
validator = RelationshipValidator(known_ids)
result = validator.validate(valid_rel)
# ✓ Block xref_001: VALID


# Invalid relationship (target not found)
invalid_rel = Relationship(
    relationship_id='xref_002',
    source_block_id='qatar_p005_text_xxx_001',
    target_block_id='qatar_p099_table_missing_000',  # Not in known_ids
    relationship_type=RelationshipType.CROSS_REFERENCE,
    reference_text='Table 99'
)

result = validator.validate(invalid_rel)
# ✗ Block xref_002: INVALID - target_block_id: Block not found
"""

if __name__ == "__main__":
    print(EXAMPLE_OUTPUT)
