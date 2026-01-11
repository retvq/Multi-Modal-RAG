"""
Block Validator

Validates ContentBlocks against schema requirements.

Validation levels:
1. Required fields (block_id, document_id, modality, content, page_number)
2. Field value constraints (page_number > 0, non-empty content for text)
3. Modality-specific requirements (TEXT requires text_metadata, etc.)
4. Provenance completeness (section_hierarchy non-empty)

Fail-fast behavior:
- Raises ValidationError on first critical error
- Collects warnings for non-critical issues
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(src_path))

from src.schemas.content_block import ContentBlock, Modality


class ValidationSeverity(Enum):
    """Severity level of validation issues."""
    ERROR = "ERROR"         # Block is invalid, must be rejected
    WARNING = "WARNING"     # Block is usable but has issues
    INFO = "INFO"           # Informational, not a problem


@dataclass
class ValidationIssue:
    """A single validation issue."""
    field: str
    message: str
    severity: ValidationSeverity
    value: Optional[str] = None  # The invalid value (for debugging)


@dataclass
class ValidationResult:
    """Result of validating a ContentBlock."""
    block_id: str
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    
    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.ERROR]
    
    @property  
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == ValidationSeverity.WARNING]
    
    def __str__(self) -> str:
        if self.is_valid:
            return f"✓ Block {self.block_id}: VALID"
        else:
            errors = ", ".join(f"{e.field}: {e.message}" for e in self.errors)
            return f"✗ Block {self.block_id}: INVALID - {errors}"


class ValidationError(Exception):
    """
    Raised when a ContentBlock fails validation.
    
    Used for fail-fast behavior in strict mode.
    """
    def __init__(self, result: ValidationResult):
        self.result = result
        super().__init__(str(result))


class BlockValidator:
    """
    Validates ContentBlocks against schema requirements.
    
    Usage:
        validator = BlockValidator(strict=True)
        
        # Single block
        result = validator.validate(block)
        
        # Batch with fail-fast
        valid_blocks = validator.validate_batch(blocks)  # Raises on invalid
        
        # Batch without fail-fast
        results = validator.validate_batch(blocks, fail_fast=False)
    """
    
    def __init__(self, strict: bool = True):
        """
        Initialize validator.
        
        Args:
            strict: If True, validate() raises ValidationError on invalid blocks
        """
        self.strict = strict
    
    def validate(self, block: ContentBlock) -> ValidationResult:
        """
        Validate a single ContentBlock.
        
        Args:
            block: ContentBlock to validate
            
        Returns:
            ValidationResult with all issues
            
        Raises:
            ValidationError: If strict mode and block is invalid
        """
        issues = []
        
        # === Required Field Validation ===
        issues.extend(self._validate_required_fields(block))
        
        # === Field Constraint Validation ===
        issues.extend(self._validate_field_constraints(block))
        
        # === Provenance Validation ===
        issues.extend(self._validate_provenance(block))
        
        # === Modality-Specific Validation ===
        issues.extend(self._validate_modality_specific(block))
        
        # === Citation Validation ===
        issues.extend(self._validate_citations(block))
        
        # Build result
        has_errors = any(i.severity == ValidationSeverity.ERROR for i in issues)
        result = ValidationResult(
            block_id=block.block_id or "<no_id>",
            is_valid=not has_errors,
            issues=issues
        )
        
        # Fail fast in strict mode
        if self.strict and not result.is_valid:
            raise ValidationError(result)
        
        return result
    
    def validate_batch(
        self,
        blocks: List[ContentBlock],
        fail_fast: bool = True
    ) -> List[ValidationResult]:
        """
        Validate multiple ContentBlocks.
        
        Args:
            blocks: List of ContentBlocks
            fail_fast: If True, stop on first invalid block
            
        Returns:
            List of ValidationResults
            
        Raises:
            ValidationError: If fail_fast and any block is invalid
        """
        results = []
        
        for block in blocks:
            try:
                result = self.validate(block)
                results.append(result)
            except ValidationError as e:
                if fail_fast:
                    raise
                results.append(e.result)
        
        return results
    
    def _validate_required_fields(self, block: ContentBlock) -> List[ValidationIssue]:
        """Validate that all required fields are present."""
        issues = []
        
        # block_id
        if not block.block_id:
            issues.append(ValidationIssue(
                field="block_id",
                message="Required field is empty",
                severity=ValidationSeverity.ERROR
            ))
        
        # document_id
        if not block.document_id:
            issues.append(ValidationIssue(
                field="document_id",
                message="Required field is empty",
                severity=ValidationSeverity.ERROR
            ))
        
        # modality
        if not block.modality:
            issues.append(ValidationIssue(
                field="modality",
                message="Required field is empty",
                severity=ValidationSeverity.ERROR
            ))
        elif not isinstance(block.modality, Modality):
            issues.append(ValidationIssue(
                field="modality",
                message=f"Invalid modality type: {type(block.modality)}",
                severity=ValidationSeverity.ERROR,
                value=str(block.modality)
            ))
        
        # page_number
        if block.page_number is None:
            issues.append(ValidationIssue(
                field="page_number",
                message="Required field is missing",
                severity=ValidationSeverity.ERROR
            ))
        
        return issues
    
    def _validate_field_constraints(self, block: ContentBlock) -> List[ValidationIssue]:
        """Validate field value constraints."""
        issues = []
        
        # page_number must be positive
        if block.page_number is not None and block.page_number < 1:
            issues.append(ValidationIssue(
                field="page_number",
                message=f"Must be >= 1, got {block.page_number}",
                severity=ValidationSeverity.ERROR,
                value=str(block.page_number)
            ))
        
        # content should not be empty for TEXT modality
        if block.modality == Modality.TEXT and not block.content:
            issues.append(ValidationIssue(
                field="content",
                message="TEXT blocks must have non-empty content",
                severity=ValidationSeverity.ERROR
            ))
        
        # extraction_confidence should be 0-1
        if block.extraction_confidence < 0 or block.extraction_confidence > 1:
            issues.append(ValidationIssue(
                field="extraction_confidence",
                message=f"Must be between 0 and 1, got {block.extraction_confidence}",
                severity=ValidationSeverity.WARNING,
                value=str(block.extraction_confidence)
            ))
        
        # content_length should match actual length
        if block.content and block.content_length != len(block.content):
            issues.append(ValidationIssue(
                field="content_length",
                message=f"Mismatch: stored={block.content_length}, actual={len(block.content)}",
                severity=ValidationSeverity.WARNING,
                value=str(block.content_length)
            ))
        
        return issues
    
    def _validate_provenance(self, block: ContentBlock) -> List[ValidationIssue]:
        """Validate provenance fields."""
        issues = []
        
        # section_hierarchy must not be empty
        if not block.section_hierarchy:
            issues.append(ValidationIssue(
                field="section_hierarchy",
                message="Must not be empty - every block needs section context",
                severity=ValidationSeverity.ERROR
            ))
        
        # bounding_box validation if present
        if block.bounding_box:
            bbox = block.bounding_box
            if not (0 <= bbox.x0 <= bbox.x1 <= 1):
                issues.append(ValidationIssue(
                    field="bounding_box.x",
                    message=f"Invalid x coordinates: x0={bbox.x0}, x1={bbox.x1}",
                    severity=ValidationSeverity.WARNING
                ))
            if not (0 <= bbox.y0 <= bbox.y1 <= 1):
                issues.append(ValidationIssue(
                    field="bounding_box.y",
                    message=f"Invalid y coordinates: y0={bbox.y0}, y1={bbox.y1}",
                    severity=ValidationSeverity.WARNING
                ))
        
        return issues
    
    def _validate_modality_specific(self, block: ContentBlock) -> List[ValidationIssue]:
        """Validate modality-specific requirements."""
        issues = []
        
        if block.modality == Modality.TEXT:
            # TEXT should have text_metadata
            if not block.text_metadata:
                issues.append(ValidationIssue(
                    field="text_metadata",
                    message="TEXT blocks should have text_metadata",
                    severity=ValidationSeverity.WARNING
                ))
        
        elif block.modality == Modality.TABLE:
            # TABLE should have _table_metadata
            if not hasattr(block, '_table_metadata') or not block._table_metadata:
                issues.append(ValidationIssue(
                    field="_table_metadata",
                    message="TABLE blocks should have table metadata attached",
                    severity=ValidationSeverity.WARNING
                ))
        
        elif block.modality == Modality.FIGURE:
            # FIGURE should have _figure_metadata
            if not hasattr(block, '_figure_metadata') or not block._figure_metadata:
                issues.append(ValidationIssue(
                    field="_figure_metadata",
                    message="FIGURE blocks should have figure metadata attached",
                    severity=ValidationSeverity.WARNING
                ))
            else:
                # FIGURE should have image_path
                fm = block._figure_metadata
                if not fm.image_path:
                    issues.append(ValidationIssue(
                        field="_figure_metadata.image_path",
                        message="FIGURE blocks should have image_path set",
                        severity=ValidationSeverity.WARNING
                    ))
        
        elif block.modality == Modality.FOOTNOTE:
            # FOOTNOTE should have marker
            if not hasattr(block, '_footnote_marker') or not block._footnote_marker:
                issues.append(ValidationIssue(
                    field="_footnote_marker",
                    message="FOOTNOTE blocks should have marker attached",
                    severity=ValidationSeverity.WARNING
                ))
        
        return issues
    
    def _validate_citations(self, block: ContentBlock) -> List[ValidationIssue]:
        """Validate citation fields."""
        issues = []
        
        # citation_short should not be empty
        if not block.citation_short:
            issues.append(ValidationIssue(
                field="citation_short",
                message="Citation fields should be populated",
                severity=ValidationSeverity.WARNING
            ))
        
        return issues


# Example of failed validation
EXAMPLE_INVALID_BLOCK = """
Example of failed validation:

# Create an invalid block
invalid_block = ContentBlock(
    block_id="",                    # ERROR: empty
    document_id="qatar_test",
    modality=Modality.TEXT,
    content="",                     # ERROR: empty for TEXT
    page_number=0,                  # ERROR: must be >= 1
    section_hierarchy=[],           # ERROR: must not be empty
)

# Validate
validator = BlockValidator(strict=False)
result = validator.validate(invalid_block)

# Result:
ValidationResult(
    block_id='<no_id>',
    is_valid=False,
    issues=[
        ValidationIssue(field='block_id', message='Required field is empty', severity=ERROR),
        ValidationIssue(field='page_number', message='Must be >= 1, got 0', severity=ERROR),
        ValidationIssue(field='content', message='TEXT blocks must have non-empty content', severity=ERROR),
        ValidationIssue(field='section_hierarchy', message='Must not be empty', severity=ERROR),
        ValidationIssue(field='text_metadata', message='TEXT blocks should have text_metadata', severity=WARNING),
        ValidationIssue(field='citation_short', message='Citation fields should be populated', severity=WARNING),
    ]
)

# String representation:
✗ Block <no_id>: INVALID - block_id: Required field is empty, page_number: Must be >= 1, got 0, content: TEXT blocks must have non-empty content, section_hierarchy: Must not be empty
"""

if __name__ == "__main__":
    print(EXAMPLE_INVALID_BLOCK)
