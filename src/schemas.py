"""
Pydantic schemas for the Agentic CAD Framework.

Defines data models for GraphState, parametric specifications,
CAD results, and judge outputs.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field
from enum import Enum


# =============================================================================
# Enums
# =============================================================================

class CADCategory(str, Enum):
    """LLM4CAD dataset categories."""
    FLANGE = "Flange"
    NUT = "Nut"
    SHAFT = "Shaft"
    GEAR = "Gear"
    SPRING = "Spring"


# =============================================================================
# Parametric Specifications (per category)
# =============================================================================

class FlangeSpec(BaseModel):
    """Parametric specification for Flange category."""
    base_diameter: float = Field(..., description="Base diameter in mm")
    base_height: float = Field(..., description="Base height/thickness in mm")
    outer_diameter: float = Field(..., description="Raised face outer diameter in mm")
    inner_diameter: float = Field(..., description="Bore diameter in mm")
    flange_height: float = Field(..., description="Total flange height in mm")


class GearSpec(BaseModel):
    """Parametric specification for Gear category."""
    module: float = Field(..., description="Gear module (tooth size parameter)")
    teeth_number: int = Field(..., description="Number of teeth")
    width: float = Field(..., description="Gear width/thickness in mm")
    bore_d: float = Field(..., description="Bore diameter in mm")


class NutSpec(BaseModel):
    """Parametric specification for Nut category."""
    nut_size: float = Field(..., description="Nut size (across flats) in mm")
    nut_height: float = Field(..., description="Nut height in mm")
    inner_diameter: float = Field(..., description="Thread inner diameter in mm")


class ShaftSpec(BaseModel):
    """Parametric specification for Shaft category."""
    diameter: float = Field(..., description="Shaft diameter in mm")
    length: float = Field(..., description="Shaft length in mm")
    # Additional fields may vary - keep flexible
    step_diameter: Optional[float] = Field(None, description="Step diameter if stepped shaft")
    step_length: Optional[float] = Field(None, description="Step length if stepped shaft")


class SpringSpec(BaseModel):
    """Parametric specification for Spring category."""
    wire_diameter: float = Field(..., description="Wire diameter in mm")
    coil_diameter: float = Field(..., description="Mean coil diameter in mm")
    num_coils: int = Field(..., description="Number of active coils")
    free_length: float = Field(..., description="Free length in mm")


# Union type for any parameter spec
ParamSpec = FlangeSpec | GearSpec | NutSpec | ShaftSpec | SpringSpec


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
    category: CADCategory = Field(..., description="Part category")
    stl_path: Optional[str] = Field(None, description="Path to ground truth STL")
    run_id: Optional[str] = Field(None, description="Unique run identifier")


class GraphState(BaseModel):
    """
    State object passed through the LangGraph design agent.
    
    This is the central data structure that flows through all nodes
    in the iterative design loop.
    """
    # Input fields
    text_desc: str = Field(..., description="Natural language description of the part")
    gt_param_spec: dict | list = Field(..., description="Ground truth parametric specification (dict or list for Shaft)")
    
    # Generated fields
    pred_param_spec: Optional[dict] = Field(None, description="Predicted parametric specification")
    
    # Param judge results
    param_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Parameter accuracy score")
    param_feedback: Optional[str] = Field(None, description="Parameter judge feedback")
    
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
    """A single sample from the LLM4CAD dataset."""
    sample_id: str = Field(..., description="Unique identifier (e.g., 'flange_00001')")
    category: CADCategory = Field(..., description="Part category")
    text_desc: str = Field(..., description="Human language description")
    gt_param_spec: dict | list = Field(..., description="Ground truth parameters from JSON (dict or list for Shaft)")
    stl_path: str = Field(..., description="Path to STL mesh file")
    
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
                category=self.category,
                stl_path=self.stl_path,
                run_id=run_id
            ),
            iteration_history=[]
        )

