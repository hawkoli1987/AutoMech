"""
Pydantic schemas for the Agentic CAD Framework.

Defines data models for GraphState, parametric specifications,
CAD results, and judge outputs.
"""

from typing import Optional, Literal, Union
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# Note: Fixed category schemas (FlangeSpec, GearSpec, etc.) have been removed.
# This codebase now focuses on freeform CAD generation using CadQueryCodeDesign.
# =============================================================================


# =============================================================================
# Freeform Code Generation (Proposal 2)
# =============================================================================

class CadQueryCodeDesign(BaseModel):
    """
    Freeform CAD design using generated CadQuery Python code.
    
    This allows LLM to generate arbitrary shapes beyond fixed templates.
    """
    description: str = Field(..., description="Human-readable description of the design intent")
    code: str = Field(..., description="Valid Python code using CadQuery API")
    entry_point: str = Field(default="result", description="Variable name containing the final CadQuery Workplane")
    required_imports: list[str] = Field(
        default_factory=lambda: ["cadquery as cq"],
        description="Python imports required by the code"
    )
    comments: Optional[str] = Field(None, description="Additional design notes or rationale")


# =============================================================================
# Judge Results
# =============================================================================

class ParamJudgeResult(BaseModel):
    """Result from JudgeParamSpec node."""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall accuracy score 0-1")
    parameter_scores: dict[str, float] = Field(default_factory=dict, description="Per-parameter scores")
    feedback: str = Field("", description="Specific issues or feedback")


class VLMJudgeResult(BaseModel):
    """Result from JudgeCAD_VLM node."""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall visual quality score 0-1")
    geometric_score: float = Field(..., ge=0.0, le=1.0, description="Geometric correctness score")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Feature completeness score")
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Visual quality score")
    feedback: str = Field("", description="Visual issues observed")


# =============================================================================
# CAD Generation Results
# =============================================================================

class CADResult(BaseModel):
    """Result from GenerateCAD node."""
    success: bool = Field(..., description="Whether CAD generation succeeded")
    cad_file: Optional[str] = Field(None, description="Path to generated CAD file (STEP/STL)")
    render_image: Optional[str] = Field(None, description="Path to rendered PNG image")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    volume: Optional[float] = Field(None, description="Volume of generated part in mm³")


# =============================================================================
# GraphState for LangGraph
# =============================================================================

class GraphStateMetadata(BaseModel):
    """Metadata stored in GraphState."""
    sample_id: str = Field(..., description="Unique sample identifier")
    stl_path: Optional[str] = Field(None, description="Path to ground truth STL")
    run_id: Optional[str] = Field(None, description="Unique run identifier")


class GraphState(BaseModel):
    """
    State object passed through the LangGraph design agent.
    
    This is the central data structure that flows through all nodes
    in the iterative design loop.
    
    Note: For freeform CAD generation, pred_param_spec is not used.
    Instead, the agent generates CadQuery code directly.
    """
    # Input fields
    text_desc: str = Field(..., description="Natural language description of the part")
    gt_param_spec: Optional[Union[dict, list]] = Field(None, description="Ground truth parametric specification (optional, for legacy compatibility)")
    
    # Generated fields (legacy - may not be used in freeform mode)
    pred_param_spec: Optional[dict] = Field(None, description="Predicted parametric specification (legacy)")
    
    # Param judge results (legacy - may not be used in freeform mode)
    param_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Parameter accuracy score (legacy)")
    param_feedback: Optional[str] = Field(None, description="Parameter judge feedback (legacy)")
    
    # CAD generation results
    cad_file: Optional[str] = Field(None, description="Path to generated CAD file")
    render_image: Optional[str] = Field(None, description="Path to rendered image")
    
    # VLM judge results
    vlm_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Visual quality score")
    vlm_feedback: Optional[str] = Field(None, description="VLM judge feedback")
    
    # Control flow
    iteration: int = Field(0, ge=0, description="Current iteration count")
    done: bool = Field(False, description="Whether agent loop should terminate")
    
    # Metadata
    metadata: GraphStateMetadata = Field(..., description="Sample and run metadata")
    
    # History (for iterative refinement)
    iteration_history: list[dict] = Field(default_factory=list, description="History of previous iterations")


# =============================================================================
# Data Sample (from dataset loader)
# =============================================================================

class LLM4CADSample(BaseModel):
    """
    A single sample from the LLM4CAD dataset.
    
    Note: This is kept for backward compatibility with existing data loaders,
    but category-specific logic has been removed for freeform generation.
    """
    sample_id: str = Field(..., description="Unique identifier")
    text_desc: str = Field(..., description="Human language description")
    gt_param_spec: Optional[Union[dict, list]] = Field(None, description="Ground truth parameters (optional, for legacy compatibility)")
    stl_path: Optional[str] = Field(None, description="Path to STL mesh file (optional)")
    
    def to_graph_state(self, run_id: Optional[str] = None) -> GraphState:
        """Convert sample to initial GraphState for agent."""
        return GraphState(
            text_desc=self.text_desc,
            gt_param_spec=self.gt_param_spec,
            pred_param_spec=None,
            param_score=None,
            param_feedback=None,
            cad_file=None,
            render_image=None,
            vlm_score=None,
            vlm_feedback=None,
            iteration=0,
            done=False,
            metadata=GraphStateMetadata(
                sample_id=self.sample_id,
                stl_path=self.stl_path,
                run_id=run_id
            ),
            iteration_history=[]
        )

