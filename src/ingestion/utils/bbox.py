"""
Bounding Box Utilities

Operations on normalized bounding boxes:
- Coordinate normalization
- Containment checks
- Intersection detection
"""

from typing import Tuple, Optional
import sys
sys.path.insert(0, str(__file__).rsplit("src", 1)[0] + "src")

from schemas.content_block import BoundingBox


def normalize_bbox(
    x0: float, y0: float, x1: float, y1: float,
    page_width: float, page_height: float,
    origin_bottom_left: bool = True
) -> BoundingBox:
    """
    Normalize PDF coordinates to 0-1 range with top-left origin.
    
    PDF coordinate system has origin at bottom-left.
    We normalize to top-left origin for consistency with
    typical image coordinate systems.
    
    Args:
        x0, y0: Bottom-left corner in PDF coords
        x1, y1: Top-right corner in PDF coords
        page_width: Page width in PDF units
        page_height: Page height in PDF units
        origin_bottom_left: If True, flip y-axis
        
    Returns:
        Normalized BoundingBox
    """
    # Normalize to 0-1 range
    norm_x0 = x0 / page_width
    norm_x1 = x1 / page_width
    
    if origin_bottom_left:
        # Flip y-axis: PDF has origin at bottom-left
        norm_y0 = 1.0 - (y1 / page_height)
        norm_y1 = 1.0 - (y0 / page_height)
    else:
        norm_y0 = y0 / page_height
        norm_y1 = y1 / page_height
    
    # Clamp to valid range
    norm_x0 = max(0.0, min(1.0, norm_x0))
    norm_x1 = max(0.0, min(1.0, norm_x1))
    norm_y0 = max(0.0, min(1.0, norm_y0))
    norm_y1 = max(0.0, min(1.0, norm_y1))
    
    return BoundingBox(
        x0=norm_x0,
        y0=norm_y0,
        x1=norm_x1,
        y1=norm_y1
    )


def bbox_contains(outer: BoundingBox, inner: BoundingBox, tolerance: float = 0.01) -> bool:
    """
    Check if outer bounding box contains inner bounding box.
    
    Used for:
    - Determining if text is within a table region
    - Determining if element is within a box/annex region
    
    Args:
        outer: Potential container bounding box
        inner: Potential contained bounding box
        tolerance: Margin for containment check (handles extraction variance)
        
    Returns:
        True if inner is contained within outer
    """
    return (
        outer.x0 - tolerance <= inner.x0 and
        outer.y0 - tolerance <= inner.y0 and
        outer.x1 + tolerance >= inner.x1 and
        outer.y1 + tolerance >= inner.y1
    )


def bbox_intersects(a: BoundingBox, b: BoundingBox) -> bool:
    """
    Check if two bounding boxes overlap.
    
    Args:
        a: First bounding box
        b: Second bounding box
        
    Returns:
        True if boxes overlap
    """
    # No overlap if one is completely to the left/right/above/below
    if a.x1 < b.x0 or b.x1 < a.x0:
        return False
    if a.y1 < b.y0 or b.y1 < a.y0:
        return False
    return True


def bbox_area(bbox: BoundingBox) -> float:
    """
    Calculate area of bounding box.
    
    Args:
        bbox: Bounding box
        
    Returns:
        Area (in normalized units, 0-1 range)
    """
    return (bbox.x1 - bbox.x0) * (bbox.y1 - bbox.y0)


def merge_bboxes(bboxes: list[BoundingBox]) -> Optional[BoundingBox]:
    """
    Create bounding box that contains all input boxes.
    
    Used for computing paragraph bounds from character/word bounds.
    
    Args:
        bboxes: List of bounding boxes to merge
        
    Returns:
        Merged bounding box, or None if input is empty
    """
    if not bboxes:
        return None
    
    return BoundingBox(
        x0=min(b.x0 for b in bboxes),
        y0=min(b.y0 for b in bboxes),
        x1=max(b.x1 for b in bboxes),
        y1=max(b.y1 for b in bboxes)
    )
