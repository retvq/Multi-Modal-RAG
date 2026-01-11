"""
Ingestion Pipeline Orchestrator

Main entry point for document ingestion. Coordinates all modality-specific
extractors, normalizers, linkers, and validators to produce a unified output.

Execution Flow:
1. Load document (validate PDF, get page count)
2. For each page:
   a. Extract text blocks
   b. Extract footnotes
3. Extract tables (all pages with tables)
4. Extract figures (all pages with figures)
5. Normalize all raw blocks to ContentBlocks
6. Run linkers (cross-references, footnotes)
7. Validate all blocks and relationships
8. Write output artifacts (blocks, relationships, manifest)

Output Structure:
- blocks.json: List of all ContentBlocks
- relationships.json: List of all Relationships
- manifest.json: Ingestion summary and metrics
- figures/: Directory of extracted figure images
"""

import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import sys

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

from src.ingestion.loaders.pdf_loader import PDFLoader, LoadedDocument
from src.ingestion.extractors.text_extractor import TextExtractor
from src.ingestion.extractors.table_extractor import TableExtractor
from src.ingestion.extractors.figure_extractor import FigureExtractor
from src.ingestion.extractors.footnote_extractor import FootnoteExtractor, normalize_footnote_to_block
from src.ingestion.normalizers.text_normalizer import TextNormalizer
from src.ingestion.normalizers.table_normalizer import TableNormalizer
from src.ingestion.normalizers.figure_normalizer import FigureNormalizer
from src.ingestion.linkers.cross_reference_linker import CrossReferenceLinker
from src.ingestion.linkers.footnote_linker import FootnoteLinker
from src.ingestion.validators.block_validator import BlockValidator
from src.ingestion.validators.relationship_validator import RelationshipValidator
from src.schemas.content_block import ContentBlock, Modality
from src.schemas.relationship import Relationship
from src.schemas.manifest import IngestionManifest, IngestionStatus


class IngestionPipeline:
    """
    Main orchestrator for document ingestion.
    
    Coordinates all extraction, normalization, linking, and validation
    to produce a complete ingestion output.
    
    Usage:
        pipeline = IngestionPipeline(
            document_path="path/to/document.pdf",
            output_dir="path/to/output"
        )
        manifest = pipeline.run()
    """
    
    def __init__(
        self,
        document_path: str,
        output_dir: str,
        extract_figures: bool = True,
        run_ocr: bool = False,
        validate_strict: bool = False,
        page_range: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            document_path: Path to PDF document
            output_dir: Directory for output artifacts
            extract_figures: Whether to extract figure images
            run_ocr: Whether to run OCR on figures
            validate_strict: Whether to fail on validation errors
            page_range: Optional (start, end) pages to process (1-indexed, inclusive)
        """
        self.document_path = Path(document_path)
        self.output_dir = Path(output_dir)
        self.extract_figures = extract_figures
        self.run_ocr = run_ocr
        self.validate_strict = validate_strict
        self.page_range = page_range
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "figures").mkdir(exist_ok=True)
        
        # Collected outputs
        self.blocks: List[ContentBlock] = []
        self.relationships: List[Relationship] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []
        
        # Processing state
        self.document: Optional[LoadedDocument] = None
        self.document_id: str = ""
    
    def run(self) -> IngestionManifest:
        """
        Execute the full ingestion pipeline.
        
        Returns:
            IngestionManifest with processing results
        """
        start_time = time.time()
        
        try:
            # Step 1: Load document
            self._log("Loading document...")
            self._load_document()
            
            # Step 2: Determine page range
            start_page, end_page = self._get_page_range()
            self._log(f"Processing pages {start_page}-{end_page}")
            
            # Step 3: Extract and normalize text
            self._log("Extracting text...")
            self._extract_text(start_page, end_page)
            
            # Step 4: Extract and normalize tables
            self._log("Extracting tables...")
            self._extract_tables(start_page, end_page)
            
            # Step 5: Extract and normalize figures
            if self.extract_figures:
                self._log("Extracting figures...")
                self._extract_figures(start_page, end_page)
            
            # Step 6: Extract and normalize footnotes
            self._log("Extracting footnotes...")
            self._extract_footnotes(start_page, end_page)
            
            # Step 7: Run linkers
            self._log("Linking cross-references...")
            self._run_linkers()
            
            # Step 8: Validate
            self._log("Validating outputs...")
            self._validate()
            
            # Step 9: Write outputs
            self._log("Writing outputs...")
            self._write_outputs()
            
            # Build manifest
            duration = time.time() - start_time
            manifest = self._build_manifest(duration, IngestionStatus.SUCCESS)
            
            # Write manifest
            self._write_manifest(manifest)
            
            self._log(f"Ingestion complete in {duration:.2f}s")
            
            return manifest
            
        except Exception as e:
            duration = time.time() - start_time
            self.errors.append(str(e))
            manifest = self._build_manifest(duration, IngestionStatus.FAILED)
            self._write_manifest(manifest)
            raise
        
        finally:
            if self.document:
                self.document.close()
    
    def _log(self, message: str):
        """Log progress message."""
        print(f"[Pipeline] {message}")
    
    def _load_document(self):
        """Load and validate the document."""
        self.document = PDFLoader.load(str(self.document_path))
        self.document_id = self.document.document_id
    
    def _get_page_range(self) -> Tuple[int, int]:
        """Determine which pages to process."""
        if self.page_range:
            return self.page_range
        return (1, self.document.total_pages)
    
    def _extract_text(self, start_page: int, end_page: int):
        """Extract text from all pages."""
        extractor = TextExtractor()
        normalizer = TextNormalizer(self.document_id)
        
        for page_num in range(start_page, end_page + 1):
            page = self.document.get_page(page_num)
            if not page:
                continue
            
            try:
                raw_blocks = extractor.extract(page)
                
                # Filter headers/footers
                content_blocks = [b for b in raw_blocks if not b.is_header_footer]
                
                # Normalize
                section_hierarchy = self._infer_section(page_num)
                normalized = normalizer.normalize_batch(
                    raw_blocks=content_blocks,
                    section_hierarchy=section_hierarchy
                )
                
                self.blocks.extend(normalized)
                
            except Exception as e:
                self.warnings.append(f"Text extraction failed on page {page_num}: {e}")
    
    def _extract_tables(self, start_page: int, end_page: int):
        """Extract tables from all pages."""
        normalizer = TableNormalizer(self.document_id)
        
        with TableExtractor(str(self.document_path)) as extractor:
            for page_num in range(start_page, end_page + 1):
                try:
                    raw_tables = extractor.extract_from_page(page_num)
                    
                    for raw in raw_tables:
                        section_hierarchy = self._infer_section(page_num, "Tables")
                        block = normalizer.normalize(raw, section_hierarchy)
                        self.blocks.append(block)
                        
                except Exception as e:
                    self.warnings.append(f"Table extraction failed on page {page_num}: {e}")
    
    def _extract_figures(self, start_page: int, end_page: int):
        """Extract figures from all pages."""
        extractor = FigureExtractor(
            output_dir=str(self.output_dir / "figures"),
            run_ocr=self.run_ocr
        )
        normalizer = FigureNormalizer(self.document_id)
        
        for page_num in range(start_page, end_page + 1):
            page = self.document.get_page(page_num)
            if not page:
                continue
            
            try:
                raw_figures = extractor.extract(page, self.document_id)
                
                for raw in raw_figures:
                    section_hierarchy = self._infer_section(page_num, "Figures")
                    block = normalizer.normalize(raw, section_hierarchy)
                    self.blocks.append(block)
                    
            except Exception as e:
                self.warnings.append(f"Figure extraction failed on page {page_num}: {e}")
    
    def _extract_footnotes(self, start_page: int, end_page: int):
        """Extract footnotes from all pages."""
        extractor = FootnoteExtractor()
        
        for page_num in range(start_page, end_page + 1):
            page = self.document.get_page(page_num)
            if not page:
                continue
            
            try:
                footnotes, markers = extractor.extract(page)
                
                for fn in footnotes:
                    block = normalize_footnote_to_block(
                        fn, 
                        self.document_id,
                        ["Document", "Footnotes"]
                    )
                    self.blocks.append(block)
                    
            except Exception as e:
                self.warnings.append(f"Footnote extraction failed on page {page_num}: {e}")
    
    def _run_linkers(self):
        """Run cross-reference and footnote linkers."""
        # Cross-reference linker
        xref_linker = CrossReferenceLinker()
        xref_rels, unresolved = xref_linker.link(self.blocks)
        self.relationships.extend(xref_rels)
        
        for ur in unresolved:
            self.warnings.append(f"Unresolved cross-reference: '{ur.reference_text}' on page {ur.page_number}")
        
        # Footnote linker
        fn_linker = FootnoteLinker()
        fn_rels, orphan_fns, orphan_markers = fn_linker.link(self.blocks)
        self.relationships.extend(fn_rels)
        
        for ofn in orphan_fns:
            self.warnings.append(f"Orphan footnote [{ofn.marker}] on page {ofn.page_number}")
    
    def _validate(self):
        """Validate all blocks and relationships."""
        block_validator = BlockValidator(strict=self.validate_strict)
        
        for block in self.blocks:
            try:
                result = block_validator.validate(block)
                if not result.is_valid:
                    for error in result.errors:
                        self.errors.append(f"Block {block.block_id}: {error.field} - {error.message}")
            except Exception as e:
                self.errors.append(f"Validation failed for block: {e}")
        
        # Validate relationships
        known_ids = {b.block_id for b in self.blocks}
        rel_validator = RelationshipValidator(known_ids, strict=False)
        
        for rel in self.relationships:
            result = rel_validator.validate(rel)
            if not result.is_valid:
                for error in result.errors:
                    self.warnings.append(f"Relationship {rel.relationship_id}: {error.message}")
    
    def _infer_section(self, page_num: int, subsection: Optional[str] = None) -> List[str]:
        """
        Infer section hierarchy from page number.
        
        This is a simplified version. A full implementation would
        detect section headings and build proper hierarchy.
        """
        hierarchy = ["Document"]
        
        # IMF document structure heuristics
        if page_num <= 3:
            hierarchy.append("Front Matter")
        elif page_num <= 6:
            hierarchy.append("Press Release")
        elif page_num <= 35:
            hierarchy.append("Staff Report")
        elif page_num <= 50:
            hierarchy.append("Statistical Appendix")
        else:
            hierarchy.append("Annexes")
        
        if subsection:
            hierarchy.append(subsection)
        
        return hierarchy
    
    def _write_outputs(self):
        """Write blocks and relationships to JSON files."""
        # Serialize blocks
        blocks_data = []
        for block in self.blocks:
            block_dict = {
                "block_id": block.block_id,
                "document_id": block.document_id,
                "modality": block.modality.value,
                "content": block.content,
                "page_number": block.page_number,
                "section_hierarchy": block.section_hierarchy,
                "citation_short": block.citation_short,
                "citation_full": block.citation_full,
                "content_length": block.content_length,
                "extraction_confidence": block.extraction_confidence,
                "extraction_warnings": block.extraction_warnings,
            }
            
            if block.bounding_box:
                block_dict["bounding_box"] = {
                    "x0": block.bounding_box.x0,
                    "y0": block.bounding_box.y0,
                    "x1": block.bounding_box.x1,
                    "y1": block.bounding_box.y1,
                }
            
            blocks_data.append(block_dict)
        
        with open(self.output_dir / "blocks.json", "w", encoding="utf-8") as f:
            json.dump(blocks_data, f, indent=2, ensure_ascii=False)
        
        # Serialize relationships
        rels_data = []
        for rel in self.relationships:
            rels_data.append({
                "relationship_id": rel.relationship_id,
                "source_block_id": rel.source_block_id,
                "target_block_id": rel.target_block_id,
                "relationship_type": rel.relationship_type.value,
                "reference_text": rel.reference_text,
            })
        
        with open(self.output_dir / "relationships.json", "w", encoding="utf-8") as f:
            json.dump(rels_data, f, indent=2, ensure_ascii=False)
    
    def _build_manifest(self, duration: float, status: IngestionStatus) -> IngestionManifest:
        """Build ingestion manifest with statistics."""
        # Count by modality
        text_count = sum(1 for b in self.blocks if b.modality == Modality.TEXT)
        table_count = sum(1 for b in self.blocks if b.modality == Modality.TABLE)
        figure_count = sum(1 for b in self.blocks if b.modality == Modality.FIGURE)
        footnote_count = sum(1 for b in self.blocks if b.modality == Modality.FOOTNOTE)
        
        # Calculate confidence
        confidences = [b.extraction_confidence for b in self.blocks]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 1.0
        low_confidence_count = sum(1 for c in confidences if c < 0.8)
        
        return IngestionManifest(
            document_id=self.document_id,
            source_path=str(self.document_path),
            status=status,
            processing_duration_seconds=duration,
            total_pages=self.document.total_pages if self.document else 0,
            text_block_count=text_count,
            table_block_count=table_count,
            figure_block_count=figure_count,
            footnote_block_count=footnote_count,
            relationship_count=len(self.relationships),
            average_extraction_confidence=avg_confidence,
            low_confidence_block_count=low_confidence_count,
            warnings=self.warnings,
            errors=self.errors,
        )
    
    def _write_manifest(self, manifest: IngestionManifest):
        """Write manifest to JSON file."""
        manifest_dict = {
            "document_id": manifest.document_id,
            "source_path": manifest.source_path,
            "status": manifest.status.value,
            "ingestion_timestamp": manifest.ingestion_timestamp,
            "processing_duration_seconds": manifest.processing_duration_seconds,
            "total_pages": manifest.total_pages,
            "block_counts": {
                "text": manifest.text_block_count,
                "table": manifest.table_block_count,
                "figure": manifest.figure_block_count,
                "footnote": manifest.footnote_block_count,
                "total": manifest.total_block_count,
            },
            "relationship_count": manifest.relationship_count,
            "quality_metrics": {
                "average_extraction_confidence": manifest.average_extraction_confidence,
                "low_confidence_block_count": manifest.low_confidence_block_count,
            },
            "warnings_count": len(manifest.warnings),
            "errors_count": len(manifest.errors),
            "warnings": manifest.warnings[:20],  # First 20 warnings
            "errors": manifest.errors,
        }
        
        with open(self.output_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest_dict, f, indent=2, ensure_ascii=False)


# Entry point for command-line usage
def main():
    """Run ingestion on Qatar document."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Document Ingestion Pipeline")
    parser.add_argument("document", help="Path to PDF document")
    parser.add_argument("--output", "-o", default="./outputs/ingested", help="Output directory")
    parser.add_argument("--pages", "-p", type=str, help="Page range (e.g., '1-10')")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure extraction")
    parser.add_argument("--ocr", action="store_true", help="Run OCR on figures")
    parser.add_argument("--strict", action="store_true", help="Fail on validation errors")
    
    args = parser.parse_args()
    
    # Parse page range
    page_range = None
    if args.pages:
        parts = args.pages.split("-")
        page_range = (int(parts[0]), int(parts[1]))
    
    # Run pipeline
    pipeline = IngestionPipeline(
        document_path=args.document,
        output_dir=args.output,
        extract_figures=not args.no_figures,
        run_ocr=args.ocr,
        validate_strict=args.strict,
        page_range=page_range
    )
    
    manifest = pipeline.run()
    
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Status: {manifest.status.value}")
    print(f"Blocks: {manifest.total_block_count}")
    print(f"  - Text: {manifest.text_block_count}")
    print(f"  - Table: {manifest.table_block_count}")
    print(f"  - Figure: {manifest.figure_block_count}")
    print(f"  - Footnote: {manifest.footnote_block_count}")
    print(f"Relationships: {manifest.relationship_count}")
    print(f"Warnings: {len(manifest.warnings)}")
    print(f"Duration: {manifest.processing_duration_seconds:.2f}s")


if __name__ == "__main__":
    main()
