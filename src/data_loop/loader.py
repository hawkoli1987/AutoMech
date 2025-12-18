"""
Simple Text Description Loader

This module has been simplified to focus on freeform CAD generation.
Category-specific logic has been removed.

For freeform generation, we only need text descriptions as input.
"""

from pathlib import Path
from typing import Optional

from src.schemas import LLM4CADSample
from src.config import get_config


# =============================================================================
# Simple Sample Creation
# =============================================================================

def create_sample_from_text(
    text_desc: str,
    sample_id: Optional[str] = None,
) -> LLM4CADSample:
    """
    Create a sample from a text description for freeform CAD generation.
    
    Args:
        text_desc: Natural language description of the part
        sample_id: Optional sample identifier
    
    Returns:
        LLM4CADSample with text description only
    """
    if sample_id is None:
        import uuid
        sample_id = f"freeform_{uuid.uuid4().hex[:8]}"
    
    return LLM4CADSample(
        sample_id=sample_id,
        text_desc=text_desc,
        gt_param_spec=None,
        stl_path=None,
    )


# =============================================================================
# Legacy Note
# =============================================================================
# The previous LLM4CADLoader class with category-specific logic has been removed.
# For freeform CAD generation, use create_sample_from_text() to create samples
# from text descriptions directly.
#
# If you need to load the original LLM4CAD dataset for evaluation purposes,
# you can implement a simple file loader that reads text descriptions from
# your dataset structure.
# =============================================================================
