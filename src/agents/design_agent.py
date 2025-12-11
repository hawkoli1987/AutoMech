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
from typing import Annotated, Any, TypedDict, Optional
from operator import add

from langgraph.graph import StateGraph, END

from src.config import get_config
from src.schemas import (
    GraphState,
    GraphStateMetadata,
    CADCategory,
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
    gt_param_spec: dict | list
    
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
    category: str
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
        category=gs.metadata.category.value,
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
            category=CADCategory(state["category"]),
            stl_path=state["stl_path"],
            run_id=state["run_id"],
        ),
        iteration_history=state["iteration_history"],
    )


# =============================================================================
# Parameter Schemas per Category
# =============================================================================

CATEGORY_PARAM_SCHEMAS = {
    "Flange": {
        "type": "object",
        "properties": {
            "base_diameter": {"type": "number", "description": "Base diameter in mm"},
            "base_height": {"type": "number", "description": "Base height/thickness in mm"},
            "outer_diameter": {"type": "number", "description": "Raised face outer diameter in mm"},
            "inner_diameter": {"type": "number", "description": "Bore diameter in mm"},
            "flange_height": {"type": "number", "description": "Total flange height in mm"},
        },
        "required": ["base_diameter", "base_height", "outer_diameter", "inner_diameter", "flange_height"],
    },
    "Gear": {
        "type": "object",
        "properties": {
            "module": {"type": "number", "description": "Gear module (tooth size parameter)"},
            "teeth_number": {"type": "integer", "description": "Number of teeth"},
            "width": {"type": "number", "description": "Gear width/thickness in mm"},
            "bore_d": {"type": "number", "description": "Bore diameter in mm"},
        },
        "required": ["module", "teeth_number", "width", "bore_d"],
    },
    "Nut": {
        "type": "object",
        "properties": {
            "nut_size": {"type": "number", "description": "Nut size (across flats) in mm"},
            "nut_height": {"type": "number", "description": "Nut height in mm"},
            "inner_diameter": {"type": "number", "description": "Thread inner diameter in mm"},
        },
        "required": ["nut_size", "nut_height", "inner_diameter"],
    },
    "Shaft": {
        "type": "array",
        "items": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
            "description": "[length, diameter] pair for each shaft section",
        },
        "description": "Array of [length, diameter] pairs for stepped shaft sections",
    },
    "Spring": {
        "type": "object",
        "properties": {
            "radius": {"type": "number", "description": "Coil radius in mm"},
            "pitch": {"type": "number", "description": "Pitch (spacing between coils) in mm"},
            "height": {"type": "number", "description": "Free length/height in mm"},
            "wire_radius": {"type": "number", "description": "Wire radius in mm"},
        },
        "required": ["radius", "pitch", "height", "wire_radius"],
    },
}


# =============================================================================
# Prompt Templates
# =============================================================================

GENERATE_PARAM_SYSTEM = """You are a mechanical engineer assistant that extracts parametric specifications from natural language descriptions of mechanical parts.

You must output valid JSON matching the provided schema. Do not include any explanation or markdown formatting - output only the raw JSON object."""

GENERATE_PARAM_USER = """Given the following description of a {category} mechanical part, extract the parametric specification as a JSON object.

Description:
{text_desc}

{feedback_section}

Expected JSON schema for {category}:
{schema}

Output only the JSON object, no explanation or markdown."""

JUDGE_PARAM_SYSTEM = """You are a CAD parameter judge. Compare predicted parameters against ground truth and score the accuracy.

You must output valid JSON with the following structure:
{{
    "overall_score": <float 0-1>,
    "parameter_scores": {{<param>: <score>, ...}},
    "feedback": "<specific issues or improvements needed>"
}}

Scoring criteria:
- Exact match: 1.0
- Within 5% tolerance: 0.8
- Within 10% tolerance: 0.6
- Within 20% tolerance: 0.4
- Larger deviation: 0.0"""

JUDGE_PARAM_USER = """Compare these two parameter specifications for a {category}:

Ground Truth:
{gt_spec}

Predicted:
{pred_spec}

Analyze each parameter and provide scores. Output only JSON."""

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

JUDGE_VLM_USER = """Evaluate this rendered CAD model of a {category}:

Specified Parameters:
{pred_spec}

Original Text Description:
"{text_desc}"

Carefully examine the image and score:
1. Does the geometry match the specified dimensions?
2. Are all features (holes, threads, chamfers, etc.) present?
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
# Node Functions
# =============================================================================

def generate_param_spec(state: AgentState) -> dict:
    """
    GenerateParamSpec Node: Generate parametric specification from text description.
    
    Uses LLM to extract structured parameters from natural language.
    """
    config = get_config()
    client = get_llm_client()
    
    category = state["category"]
    text_desc = state["text_desc"]
    iteration = state["iteration"]
    
    # Get schema for this category
    schema = CATEGORY_PARAM_SCHEMAS.get(category, {})
    schema_str = json.dumps(schema, indent=2)
    
    # Add feedback from previous iteration if available (both param and VLM feedback)
    feedback_section = ""
    if iteration > 0:
        feedback_parts = []
        
        # Add parameter judge feedback
        if state.get("param_feedback"):
            feedback_parts.append(f"Parameter Judge Feedback:\n{state['param_feedback']}")
        
        # Add VLM visual feedback
        if state.get("vlm_feedback"):
            feedback_parts.append(f"Visual Quality Feedback:\n{state['vlm_feedback']}")
        
        # Add previous prediction for reference
        if state.get("pred_param_spec"):
            prev_pred = json.dumps(state["pred_param_spec"], indent=2)
            feedback_parts.append(f"Previous Prediction (to improve upon):\n{prev_pred}")
        
        if feedback_parts:
            feedback_section = f"""
=== ITERATION {iteration} - IMPROVEMENT REQUIRED ===
{chr(10).join(feedback_parts)}

IMPORTANT: Carefully address ALL the feedback above and generate improved parameters.
"""
    
    # Build prompt
    prompt = GENERATE_PARAM_USER.format(
        category=category,
        text_desc=text_desc,
        feedback_section=feedback_section,
        schema=schema_str,
    )
    
    try:
        pred_spec = client.generate_json(
            prompt=prompt,
            system_prompt=GENERATE_PARAM_SYSTEM,
            temperature=0.3,
            max_tokens=512,
        )
    except Exception as e:
        # On failure, return empty dict and let judge handle it
        pred_spec = {}
        print(f"[GenerateParamSpec] Failed to generate: {e}")
    
    return {"pred_param_spec": pred_spec}


def judge_param_spec(state: AgentState) -> dict:
    """
    JudgeParamSpec Node: Compare predicted vs ground truth parameters.
    
    Outputs score and feedback for iterative improvement.
    """
    config = get_config()
    client = get_llm_client()
    
    category = state["category"]
    gt_spec = state["gt_param_spec"]
    pred_spec = state["pred_param_spec"]
    
    # If no prediction, score is 0
    if not pred_spec:
        return {
            "param_score": 0.0,
            "param_feedback": "No parameters were generated. Please try again.",
        }
    
    # Build prompt
    prompt = JUDGE_PARAM_USER.format(
        category=category,
        gt_spec=json.dumps(gt_spec, indent=2),
        pred_spec=json.dumps(pred_spec, indent=2),
    )
    
    try:
        result = client.generate_json(
            prompt=prompt,
            system_prompt=JUDGE_PARAM_SYSTEM,
            temperature=0.2,
            max_tokens=512,
        )
        
        # Extract score and feedback
        overall_score = float(result.get("overall_score", 0.0))
        feedback = result.get("feedback", "")
        
        # Clamp score to [0, 1]
        overall_score = max(0.0, min(1.0, overall_score))
        
    except Exception as e:
        # On failure, use heuristic scoring
        overall_score = _heuristic_param_score(gt_spec, pred_spec)
        feedback = f"LLM judge failed ({e}), using heuristic score."
    
    return {
        "param_score": overall_score,
        "param_feedback": feedback,
    }


def _heuristic_param_score(gt_spec: dict | list, pred_spec: dict | list) -> float:
    """Fallback heuristic scoring when LLM judge fails."""
    if isinstance(gt_spec, list) and isinstance(pred_spec, list):
        # Shaft format: array of [length, diameter] pairs
        if len(gt_spec) != len(pred_spec):
            return 0.3
        scores = []
        for gt_pair, pred_pair in zip(gt_spec, pred_spec):
            if len(gt_pair) == 2 and len(pred_pair) == 2:
                for gt_val, pred_val in zip(gt_pair, pred_pair):
                    if gt_val == 0:
                        scores.append(1.0 if pred_val == 0 else 0.0)
                    else:
                        diff = abs(gt_val - pred_val) / abs(gt_val)
                        scores.append(max(0, 1 - diff))
        return sum(scores) / len(scores) if scores else 0.0
    
    if isinstance(gt_spec, dict) and isinstance(pred_spec, dict):
        # Dict format: compare each key
        if not gt_spec:
            return 0.0
        scores = []
        for key, gt_val in gt_spec.items():
            if key not in pred_spec:
                scores.append(0.0)
                continue
            pred_val = pred_spec[key]
            if isinstance(gt_val, (int, float)) and isinstance(pred_val, (int, float)):
                if gt_val == 0:
                    scores.append(1.0 if pred_val == 0 else 0.0)
                else:
                    diff = abs(gt_val - pred_val) / abs(gt_val)
                    scores.append(max(0, 1 - diff))
            elif gt_val == pred_val:
                scores.append(1.0)
            else:
                scores.append(0.0)
        return sum(scores) / len(scores) if scores else 0.0
    
    return 0.0


def generate_cad(state: AgentState) -> dict:
    """
    GenerateCAD Node: Generate CAD model from parameters.
    
    Uses CadQuery to generate STEP file and render PNG.
    Falls back to stub paths if CadQuery is not available.
    """
    from src.cad.generators import generate_and_export, is_cadquery_available
    
    sample_id = state["sample_id"]
    category = state["category"]
    run_id = state["run_id"] or "default"
    iteration = state["iteration"]
    pred_param_spec = state.get("pred_param_spec")
    
    config = get_config()
    output_dir = f"{config.storage.artifacts_dir}/{run_id}"
    
    # Check if we have predicted parameters
    if not pred_param_spec:
        return {
            "cad_file": None,
            "render_image": None,
        }
    
    # Check if CadQuery is available
    if not is_cadquery_available():
        # Return stub paths when CadQuery not available
        cad_file = f"{output_dir}/cad/{sample_id}_iter{iteration}.step"
        render_image = f"{output_dir}/renders/{sample_id}_iter{iteration}.png"
        return {
            "cad_file": cad_file,
            "render_image": render_image,
        }
    
    # Generate CAD model
    result = generate_and_export(
        category=category,
        params=pred_param_spec,
        output_dir=output_dir,
        sample_id=sample_id,
        iteration=iteration,
        export_step=True,
        render_png=True,
    )
    
    if not result.success:
        print(f"[GenerateCAD] Failed: {result.error_message}")
    
    return {
        "cad_file": result.cad_file,
        "render_image": result.render_image,
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
            category=state["category"],
            pred_spec=pred_spec_str,
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
    
    Termination conditions:
    1. param_score >= threshold AND vlm_score >= threshold
    2. iteration >= max_iterations
    """
    config = get_config()
    
    iteration = state["iteration"]
    param_score = state.get("param_score", 0.0) or 0.0
    vlm_score = state.get("vlm_score", 0.0) or 0.0
    
    param_threshold = config.agent.param_score_threshold
    vlm_threshold = config.agent.vlm_score_threshold
    max_iterations = config.agent.max_iterations
    
    # Check termination conditions
    scores_pass = param_score >= param_threshold and vlm_score >= vlm_threshold
    max_iter_reached = iteration >= max_iterations
    
    done = scores_pass or max_iter_reached
    
    # Build termination reason
    if scores_pass:
        termination_reason = "scores_pass"
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
        
        # Run graph
        final_state = self.graph.invoke(agent_state)
        
        # Convert back to GraphState
        return agent_state_to_graph_state(final_state)
    
    def run_from_sample(
        self,
        text_desc: str,
        gt_param_spec: dict | list,
        category: CADCategory | str,
        sample_id: str,
        run_id: Optional[str] = None,
        stl_path: Optional[str] = None,
    ) -> GraphState:
        """
        Run the design agent from raw sample data.
        
        Args:
            text_desc: Natural language description.
            gt_param_spec: Ground truth parameters.
            category: Part category.
            sample_id: Sample identifier.
            run_id: Optional run identifier.
            stl_path: Optional path to reference STL.
        
        Returns:
            Final GraphState after agent execution.
        """
        if isinstance(category, str):
            category = CADCategory(category)
        
        initial_state = GraphState(
            text_desc=text_desc,
            gt_param_spec=gt_param_spec,
            metadata=GraphStateMetadata(
                sample_id=sample_id,
                category=category,
                stl_path=stl_path,
                run_id=run_id,
            ),
        )
        
        return self.run(initial_state)


def get_design_agent() -> DesignAgent:
    """Get a configured DesignAgent instance."""
    return DesignAgent()

