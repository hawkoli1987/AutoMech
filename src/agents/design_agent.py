"""
LangGraph Design Agent

Implements the iterative design loop for parametric CAD generation.
Uses LLM for parameter generation and judging, VLM for visual evaluation.

Flow:
    GenerateParamSpec → JudgeParamSpec → GenerateCAD → JudgeCAD_VLM → DecideNextStep
                ↑                                                            ↓
                └────────────────── if not done ─────────────────────────────┘
"""

import json
from typing import Annotated, Any, TypedDict, Optional, Union
from operator import add

from langgraph.graph import StateGraph, END

from src.config import get_config
from src.schemas import (
    GraphState,
    GraphStateMetadata,
    ParamJudgeResult,
    VLMJudgeResult,
    CADResult,
)
from src.utils.llm_client import LLMClient, VLMClient, get_llm_client, get_vlm_client


# =============================================================================
# TypedDict State for LangGraph
# =============================================================================

class AgentState(TypedDict):
    """
    State dict passed through LangGraph nodes.
    
    Uses TypedDict for LangGraph compatibility while mirroring GraphState fields.
    """
    # Input fields
    text_desc: str
    gt_param_spec: Union[dict, list, None]
    
    # Generated fields
    pred_param_spec: Optional[dict]
    
    # Param judge results
    param_score: Optional[float]
    param_feedback: Optional[str]
    
    # CAD generation results
    cad_file: Optional[str]
    render_image: Optional[str]
    
    # VLM judge results
    vlm_score: Optional[float]
    vlm_feedback: Optional[str]
    
    # Control flow
    iteration: int
    done: bool
    
    # Metadata
    sample_id: str
    stl_path: Optional[str]
    run_id: Optional[str]
    
    # History
    iteration_history: list[dict]


def graph_state_to_agent_state(gs: GraphState) -> AgentState:
    """Convert Pydantic GraphState to LangGraph AgentState."""
    return AgentState(
        text_desc=gs.text_desc,
        gt_param_spec=gs.gt_param_spec,
        pred_param_spec=gs.pred_param_spec,
        param_score=gs.param_score,
        param_feedback=gs.param_feedback,
        cad_file=gs.cad_file,
        render_image=gs.render_image,
        vlm_score=gs.vlm_score,
        vlm_feedback=gs.vlm_feedback,
        iteration=gs.iteration,
        done=gs.done,
        sample_id=gs.metadata.sample_id,
        stl_path=gs.metadata.stl_path,
        run_id=gs.metadata.run_id,
        iteration_history=gs.iteration_history,
    )


def agent_state_to_graph_state(state: AgentState) -> GraphState:
    """Convert LangGraph AgentState back to Pydantic GraphState."""
    return GraphState(
        text_desc=state["text_desc"],
        gt_param_spec=state["gt_param_spec"],
        pred_param_spec=state["pred_param_spec"],
        param_score=state["param_score"],
        param_feedback=state["param_feedback"],
        cad_file=state["cad_file"],
        render_image=state["render_image"],
        vlm_score=state["vlm_score"],
        vlm_feedback=state["vlm_feedback"],
        iteration=state["iteration"],
        done=state["done"],
        metadata=GraphStateMetadata(
            sample_id=state["sample_id"],
            stl_path=state["stl_path"],
            run_id=state["run_id"],
        ),
        iteration_history=state["iteration_history"],
    )


# =============================================================================
# Freeform Generation Only - No Fixed Templates
# =============================================================================
# All CAD generation now uses LLM-generated CadQuery code (see CODEGEN_SYSTEM below)
# This allows arbitrary mechanical parts beyond fixed categories

JUDGE_VLM_SYSTEM = """You are an expert CAD visual quality judge specializing in mechanical parts evaluation.

Evaluate rendered CAD models for geometric accuracy and visual quality. Be precise and critical.

Output valid JSON with this exact structure:
{{
    "overall_score": <float 0.0-1.0>,
    "geometric_score": <float 0.0-1.0>,
    "completeness_score": <float 0.0-1.0>,
    "quality_score": <float 0.0-1.0>,
    "issues": ["<list of specific issues found>"],
    "feedback": "<actionable improvement suggestions>"
}}

Scoring criteria:
- geometric_score (0.0-1.0): Does the visible geometry match the specified parameters?
  * 1.0 = Perfect match to dimensions
  * 0.7-0.9 = Minor proportional differences
  * 0.4-0.6 = Noticeable dimensional errors
  * 0.0-0.3 = Major geometric issues

- completeness_score (0.0-1.0): Are all expected features present?
  * 1.0 = All features present (holes, chamfers, fillets, etc.)
  * 0.5 = Some features missing
  * 0.0 = Major features missing

- quality_score (0.0-1.0): Is the render/model well-formed?
  * 1.0 = Clean geometry, proper rendering
  * 0.5 = Minor artifacts or rendering issues
  * 0.0 = Major defects, unmanufacturable geometry

overall_score = (geometric_score + completeness_score + quality_score) / 3"""

JUDGE_VLM_USER = """Evaluate this rendered CAD model:

Original Text Description:
"{text_desc}"

Carefully examine the image and score:
1. Does the geometry match the description?
2. Are all features (holes, threads, chamfers, etc.) present as described?
3. Is the model well-formed and manufacturable?

Output only valid JSON matching the required format."""

JUDGE_VLM_COMPARISON_USER = """Compare these two CAD renders:

LEFT IMAGE: Ground truth reference
RIGHT IMAGE: Generated model to evaluate

Target Category: {category}
Target Parameters: {pred_spec}
Description: "{text_desc}"

Score how well the generated model (right) matches the reference (left):
1. geometric_score: Do dimensions and proportions match?
2. completeness_score: Are all features present as in reference?
3. quality_score: Is the generated model well-formed?

Output only valid JSON matching the required format."""


# =============================================================================
# Freeform Code Generation Prompts (Proposal 2)
# =============================================================================

CODEGEN_SYSTEM = """You are an expert CadQuery programmer specializing in parametric 3D CAD modeling.

Generate Python code using the CadQuery API to create mechanical parts from text descriptions.

CODE TEMPLATE:
```python
# NOTE: Do NOT include any import statements!
# The 'cq' module is pre-injected and already available.

# 1. Start with a workplane
result = cq.Workplane("XY")

# 2. Build base geometry (use .circle().extrude() for cylinders, NOT .cylinder())
result = result.circle(radius).extrude(height)
# OR for rectangular parts
result = result.box(length, width, height)

# 3. Add features (holes, cutouts, etc.)
result = result.faces(">Z").workplane().circle(hole_radius).cutThruAll()

# 4. Boolean operations - AVOID separate objects creating Compounds
# PREFER: Use .cut(), .cutThruAll(), .cutBlind() directly on faces
result = result.faces(">Z").workplane().rect(x, y).cutBlind(-depth)
# AVOID: other_part = cq.Workplane(...); result.union(other_part)

# 5. Finishing touches
result = result.edges("|Z").fillet(radius)
```

CRITICAL RULES:
1. **DO NOT include any 'import' statements** - cq is already available
2. **Always assign the final shape to variable 'result'**
3. **Build a SINGLE SOLID, not a Compound**:
   - AVOID creating separate objects with .union() - this makes Compound geometry
   - PREFER building features directly: .cut(), .cutThruAll(), .cutBlind()
   - Use .circle().extrude() for cylinders, NOT .cylinder()
4. **All dimensions in millimeters**
5. Comment each logical step clearly
6. Use descriptive intermediate variables when helpful
7. No external file I/O operations
8. No infinite loops or recursion
9. Infer missing dimensions using standard engineering practices

COMMON PATTERNS:
- Cylinders: cq.Workplane("XY").circle(radius).extrude(height)
- Holes: .faces(">Z").workplane().pushPoints([...]).circle(r).cutThruAll()
- Chamfers: .edges().chamfer(distance)
- Fillets: .edges().fillet(radius)
- Arrays: .rarray(xSpacing, ySpacing, xCount, yCount)
- Selection: .faces(">Z") (top face), .edges("|Z") (vertical edges)

OUTPUT FORMAT:
{{
    "description": "Brief summary of the design",
    "code": "# cq is already available\\nresult = cq.Workplane('XY')...",
    "entry_point": "result",
    "required_imports": [],
    "comments": "Any design notes or assumptions"
}}

**IMPORTANT**: Do NOT include "import cadquery as cq" or any import statements in the code field!

Output only valid JSON matching this schema."""

CODEGEN_USER = """Generate CadQuery code to create this mechanical part:

Description:
{text_desc}

{feedback_section}

Think step-by-step:
1. What base shape do I need? (box, cylinder, sphere)
2. What dimensions should I use? (infer from description)
3. What features do I need to add? (holes, chamfers, fillets)
4. What boolean operations? (cut, union, intersect)
5. How do I select the right faces/edges?

Output only valid JSON with the 'description', 'code', 'entry_point', 'required_imports', and 'comments' fields."""


# =============================================================================
# Node Functions
# =============================================================================

def generate_freeform_code(state: AgentState) -> dict:
    """
    GenerateFreeformCode Node: Generate CadQuery code for freeform CAD design.
    
    This node is used instead of generate_param_spec when in freeform mode.
    Uses LLM to generate complete CadQuery Python code.
    """
    from src.schemas import CadQueryCodeDesign
    
    config = get_config()
    client = get_llm_client()
    
    text_desc = state["text_desc"]
    iteration = state["iteration"]
    
    # Add feedback from previous iteration if available
    feedback_section = ""
    if iteration > 0:
        feedback_parts = []
        
        # Add VLM visual feedback (most important for code generation)
        if state.get("vlm_feedback"):
            feedback_parts.append(f"Visual Quality Feedback:\n{state['vlm_feedback']}")
        
        # Add previous code for reference
        if state.get("pred_param_spec", {}).get("code"):
            prev_code = state["pred_param_spec"]["code"]
            feedback_parts.append(f"Previous Code (to improve upon):\n```python\n{prev_code}\n```")
        
        if feedback_parts:
            feedback_section = f"""
=== ITERATION {iteration} - CODE IMPROVEMENT REQUIRED ===
{chr(10).join(feedback_parts)}

IMPORTANT: Address ALL feedback above. Fix dimensional errors, add missing features, and improve code quality.
"""
    
    # Build prompt
    prompt = CODEGEN_USER.format(
        text_desc=text_desc,
        feedback_section=feedback_section,
    )
    
    try:
        # Use JSON generation for structured output
        code_design_dict = client.generate_json(
            prompt=prompt,
            system_prompt=CODEGEN_SYSTEM,
            temperature=0.3,
            max_tokens=2048,  # More tokens needed for code
        )
        
        # Validate required fields
        if not isinstance(code_design_dict, dict):
            raise ValueError(f"Expected dict, got {type(code_design_dict)}")
        
        if "code" not in code_design_dict:
            raise ValueError("Missing required field 'code'")
        
        # Set defaults for optional fields
        code_design_dict.setdefault("description", text_desc[:100])
        code_design_dict.setdefault("entry_point", "result")
        code_design_dict.setdefault("required_imports", ["cadquery as cq"])
        code_design_dict.setdefault("comments", "")
        
        # Store in pred_param_spec as a dict (will be converted to CadQueryCodeDesign later)
        return {"pred_param_spec": code_design_dict}
        
    except Exception as e:
        # On failure, return minimal fallback
        print(f"[GenerateFreeformCode] Failed to generate: {e}")
        fallback_code = {
            "description": text_desc[:100],
            "code": f"import cadquery as cq\n\n# Failed to generate code: {e}\nresult = cq.Workplane('XY').box(10, 10, 10)",
            "entry_point": "result",
            "required_imports": ["cadquery as cq"],
            "comments": f"Error during generation: {e}"
        }
        return {"pred_param_spec": fallback_code}


def generate_cad(state: AgentState) -> dict:
    """
    GenerateCAD Node: Generate CAD model from LLM-generated CadQuery code.
    
    Executes the freeform code to create arbitrary geometric shapes.
    Falls back to stub paths if CadQuery is not available.
    """
    from src.cad.generators import generate_from_code, is_cadquery_available
    from src.schemas import CadQueryCodeDesign
    
    sample_id = state["sample_id"]
    run_id = state["run_id"] or "default"
    iteration = state["iteration"]
    pred_param_spec = state.get("pred_param_spec")
    
    config = get_config()
    output_dir = f"{config.storage.artifacts_dir}/{run_id}"
    
    # Check if we have generated code
    if not pred_param_spec or not isinstance(pred_param_spec, dict) or "code" not in pred_param_spec:
        return {
            "cad_file": None,
            "render_image": None,
        }
    
    # Check if CadQuery is available
    if not is_cadquery_available():
        # Return stub paths when CadQuery not available
        cad_file = f"{output_dir}/cad/{sample_id}_freeform_iter{iteration}.step"
        render_image = f"{output_dir}/renders/{sample_id}_freeform_iter{iteration}.png"
        return {
            "cad_file": cad_file,
            "render_image": render_image,
        }
    
    # Freeform code generation mode (only mode now)
    try:
        code_design = CadQueryCodeDesign(**pred_param_spec)
        filename_prefix = f"{sample_id}_freeform_iter{iteration}"
        
        result = generate_from_code(
            code_design=code_design,
            output_dir=output_dir,
            filename_prefix=filename_prefix,
        )
        
        if not result.success:
            print(f"[GenerateCAD] Failed: {result.error_message}")
            return {
                "cad_file": None,
                "render_image": None,
            }
        
        return {
            "cad_file": result.cad_file,
            "render_image": result.render_image,
        }
        
    except Exception as e:
        print(f"[GenerateCAD] Freeform mode failed: {e}")
        return {
            "cad_file": None,
            "render_image": None,
        }


def judge_cad_vlm(state: AgentState) -> dict:
    """
    JudgeCAD_VLM Node: Evaluate rendered CAD model visually.
    
    Uses VLM (Qwen3-VL) to assess geometric and visual quality.
    Provides detailed scoring breakdown and actionable feedback.
    """
    config = get_config()
    
    # Check if VLM judging is enabled
    if not config.agent.enable_vlm_judge:
        return {
            "vlm_score": 1.0,
            "vlm_feedback": "VLM judging disabled.",
        }
    
    render_image = state.get("render_image")
    
    # Check if render exists
    from pathlib import Path
    if not render_image or not Path(render_image).exists():
        return {
            "vlm_score": 0.5,
            "vlm_feedback": "No render image available for VLM evaluation.",
        }
    
    try:
        vlm_client = get_vlm_client()
        
        # Format predicted parameters
        pred_spec = state.get("pred_param_spec", {})
        if isinstance(pred_spec, dict):
            pred_spec_str = json.dumps(pred_spec, indent=2)
        else:
            pred_spec_str = str(pred_spec)
        
        # Build prompt
        prompt = JUDGE_VLM_USER.format(
            text_desc=state["text_desc"],
        )
        
        # Call VLM for evaluation
        result = vlm_client.generate_json_with_image(
            prompt=prompt,
            image_path=render_image,
            system_prompt=JUDGE_VLM_SYSTEM,
            temperature=0.2,
            max_tokens=600,
        )
        
        # Extract scores with defaults
        geometric_score = float(result.get("geometric_score", 0.5))
        completeness_score = float(result.get("completeness_score", 0.5))
        quality_score = float(result.get("quality_score", 0.5))
        
        # Calculate overall score (average of components)
        if "overall_score" in result:
            overall_score = float(result["overall_score"])
        else:
            overall_score = (geometric_score + completeness_score + quality_score) / 3
        
        # Clamp all scores to [0, 1]
        overall_score = max(0.0, min(1.0, overall_score))
        geometric_score = max(0.0, min(1.0, geometric_score))
        completeness_score = max(0.0, min(1.0, completeness_score))
        quality_score = max(0.0, min(1.0, quality_score))
        
        # Extract feedback and issues
        feedback = result.get("feedback", "")
        issues = result.get("issues", [])
        if issues and isinstance(issues, list):
            issues_str = "; ".join(issues)
            feedback = f"{feedback} Issues: {issues_str}" if feedback else issues_str
        
    except Exception as e:
        # VLM evaluation failed - use neutral scores
        overall_score = 0.5
        geometric_score = 0.5
        completeness_score = 0.5
        quality_score = 0.5
        feedback = f"VLM evaluation failed: {str(e)[:200]}"
    
    return {
        "vlm_score": overall_score,
        "vlm_feedback": feedback,
        # Store detailed scores in history for analysis
        "_vlm_details": {
            "geometric_score": geometric_score,
            "completeness_score": completeness_score,
            "quality_score": quality_score,
        },
    }


def decide_next_step(state: AgentState) -> dict:
    """
    DecideNextStep Node: Determine whether to continue iterating or terminate.
    
    Termination conditions (any triggers termination):
    1. param_score >= threshold (high-confidence match)
    2. param_score >= 0.9 (near-perfect match, VLM optional)  
    3. iteration >= max_iterations
    4. Both param_score AND vlm_score pass their thresholds
    """
    config = get_config()
    
    iteration = state["iteration"]
    param_score = state.get("param_score", 0.0) or 0.0
    vlm_score = state.get("vlm_score", 0.0) or 0.0
    
    param_threshold = config.agent.param_score_threshold
    vlm_threshold = config.agent.vlm_score_threshold
    max_iterations = config.agent.max_iterations
    
    # Check termination conditions
    both_scores_pass = param_score >= param_threshold and vlm_score >= vlm_threshold
    param_excellent = param_score >= 0.95  # Near-perfect param match, skip VLM requirement
    param_good_vlm_ok = param_score >= param_threshold and vlm_score >= 0.3  # Param passes + minimal VLM
    max_iter_reached = iteration >= max_iterations
    
    done = both_scores_pass or param_excellent or param_good_vlm_ok or max_iter_reached
    
    # Build termination reason
    if param_excellent:
        termination_reason = "param_excellent"
    elif both_scores_pass:
        termination_reason = "scores_pass"
    elif param_good_vlm_ok:
        termination_reason = "param_good"
    elif max_iter_reached:
        termination_reason = "max_iterations"
    else:
        termination_reason = None
    
    # Save current iteration to history with detailed scores
    history_entry = {
        "iteration": iteration,
        "param_score": param_score,
        "vlm_score": vlm_score,
        "pred_param_spec": state.get("pred_param_spec"),
        "param_feedback": state.get("param_feedback"),
        "vlm_feedback": state.get("vlm_feedback"),
        "cad_file": state.get("cad_file"),
        "render_image": state.get("render_image"),
    }
    
    # Add VLM detail scores if available
    vlm_details = state.get("_vlm_details")
    if vlm_details:
        history_entry["vlm_details"] = vlm_details
    
    new_history = state.get("iteration_history", []).copy()
    new_history.append(history_entry)
    
    # Log iteration summary
    status = "PASS" if done else "CONTINUE"
    print(f"[DecideNextStep] Iter {iteration}: param={param_score:.3f}, vlm={vlm_score:.3f} → {status}")
    
    return {
        "done": done,
        "iteration": iteration + 1,
        "iteration_history": new_history,
        "_termination_reason": termination_reason,
    }


# =============================================================================
# Graph Construction
# =============================================================================

def should_continue(state: AgentState) -> str:
    """Conditional edge: continue iterating or end."""
    if state.get("done", False):
        return "end"
    return "generate_param_spec"


def build_design_agent() -> StateGraph:
    """
    Build the LangGraph design agent.
    
    Returns:
        Compiled StateGraph ready for invocation.
    """
    # Create graph
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("generate_param_spec", generate_param_spec)
    graph.add_node("judge_param_spec", judge_param_spec)
    graph.add_node("generate_cad", generate_cad)
    graph.add_node("judge_cad_vlm", judge_cad_vlm)
    graph.add_node("decide_next_step", decide_next_step)
    
    # Add edges (linear flow)
    graph.add_edge("generate_param_spec", "judge_param_spec")
    graph.add_edge("judge_param_spec", "generate_cad")
    graph.add_edge("generate_cad", "judge_cad_vlm")
    graph.add_edge("judge_cad_vlm", "decide_next_step")
    
    # Conditional edge for iteration
    graph.add_conditional_edges(
        "decide_next_step",
        should_continue,
        {
            "generate_param_spec": "generate_param_spec",
            "end": END,
        }
    )
    
    # Set entry point
    graph.set_entry_point("generate_param_spec")
    
    return graph.compile()


# =============================================================================
# Main Interface
# =============================================================================

class DesignAgent:
    """
    High-level interface for the design agent.
    
    Wraps LangGraph execution with Pydantic model conversion.
    """
    
    def __init__(self):
        """Initialize the design agent."""
        self.graph = build_design_agent()
    
    def run(self, initial_state: GraphState) -> GraphState:
        """
        Run the design agent on an initial state.
        
        Args:
            initial_state: Initial GraphState with text_desc and gt_param_spec.
        
        Returns:
            Final GraphState after agent execution.
        """
        # Convert to agent state
        agent_state = graph_state_to_agent_state(initial_state)
        
        # Run graph with increased recursion limit
        # 5 nodes per iteration * max_iterations + buffer
        config = get_config()
        recursion_limit = 5 * config.agent.max_iterations + 10
        
        final_state = self.graph.invoke(
            agent_state, 
            {"recursion_limit": recursion_limit}
        )
        
        # Convert back to GraphState
        return agent_state_to_graph_state(final_state)
    
    def run_from_sample(
        self,
        text_desc: str,
        sample_id: str,
        gt_param_spec: Optional[dict | list] = None,
        run_id: Optional[str] = None,
        stl_path: Optional[str] = None,
    ) -> GraphState:
        """
        Run the design agent from raw sample data.
        
        Args:
            text_desc: Natural language description.
            sample_id: Sample identifier.
            gt_param_spec: Optional ground truth parameters (for legacy compatibility).
            run_id: Optional run identifier.
            stl_path: Optional path to reference STL.
        
        Returns:
            Final GraphState after agent execution.
        """
        initial_state = GraphState(
            text_desc=text_desc,
            gt_param_spec=gt_param_spec,
            metadata=GraphStateMetadata(
                sample_id=sample_id,
                stl_path=stl_path,
                run_id=run_id,
            ),
        )
        
        return self.run(initial_state)


def get_design_agent() -> DesignAgent:
    """Get a configured DesignAgent instance."""
    return DesignAgent()

