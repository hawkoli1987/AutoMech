#!/usr/bin/env python3
"""
Demo: Design Agent

Shows the LangGraph design agent structure and flow.
Uses actual LLM calls when servers are available.

Run from project root: python3 scripts/demo_design_agent.py
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree


def check_llm_available():
    """Check if LLM server is available."""
    import requests
    api_base = os.getenv("OPENAI_API_BASE", "http://localhost:8001")
    try:
        response = requests.get(f"{api_base}/v1/models", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


def heuristic_param_score(gt_spec: dict, pred_spec: dict) -> float:
    """Calculate heuristic score between ground truth and predicted params."""
    if not gt_spec or not pred_spec:
        return 0.0
    
    scores = []
    for key, gt_val in gt_spec.items():
        if key not in pred_spec:
            scores.append(0.0)
            continue
        
        pred_val = pred_spec[key]
        if isinstance(gt_val, (int, float)) and isinstance(pred_val, (int, float)):
            if gt_val == 0:
                score = 1.0 if pred_val == 0 else 0.0
            else:
                error = abs(gt_val - pred_val) / abs(gt_val)
                score = max(0.0, 1.0 - error)
            scores.append(score)
        elif gt_val == pred_val:
            scores.append(1.0)
        else:
            scores.append(0.0)
    
    return sum(scores) / len(scores) if scores else 0.0


def main():
    console = Console()
    console.print(Panel.fit("[bold cyan]Design Agent Demo[/bold cyan]"))
    
    from src.agents.design_agent import (
        DesignAgent,
        AgentState,
        build_design_agent,
    )
    from src.schemas import CADCategory, FlangeSpec, GearSpec, NutSpec, ShaftSpec, SpringSpec
    from src.config import get_config
    
    # Show agent architecture
    console.print("\n[bold]1. Agent Architecture[/bold]")
    
    tree = Tree("[bold]DesignAgent (LangGraph)[/bold]")
    nodes = tree.add("[cyan]Nodes[/cyan]")
    nodes.add("generate_param_spec - LLM generates parameters from text")
    nodes.add("judge_param_spec - Score predicted vs ground truth")
    nodes.add("generate_cad - CadQuery generates STEP/STL + render")
    nodes.add("judge_cad_vlm - VLM evaluates rendered image")
    nodes.add("decide_next_step - Terminate or iterate")
    
    edges = tree.add("[cyan]Flow[/cyan]")
    edges.add("generate_param_spec → judge_param_spec → generate_cad → judge_cad_vlm → decide_next_step")
    edges.add("decide_next_step → END (if done) or generate_param_spec (iterate)")
    
    console.print(tree)
    
    # Show category schemas
    console.print("\n[bold]2. Category Schemas[/bold]")
    schemas = {
        CADCategory.FLANGE: FlangeSpec,
        CADCategory.GEAR: GearSpec,
        CADCategory.NUT: NutSpec,
        CADCategory.SHAFT: ShaftSpec,
        CADCategory.SPRING: SpringSpec,
    }
    for cat, schema in schemas.items():
        fields = list(schema.model_fields.keys())
        console.print(f"   {cat.value}: {schema.__name__} → {fields}")
    
    # Demonstrate heuristic scoring
    console.print("\n[bold]3. Heuristic Param Scoring[/bold]")
    
    gt = {"outer_diameter": 100.0, "inner_diameter": 50.0, "height": 20.0}
    pred_exact = {"outer_diameter": 100.0, "inner_diameter": 50.0, "height": 20.0}
    pred_close = {"outer_diameter": 102.0, "inner_diameter": 51.0, "height": 19.5}
    pred_wrong = {"outer_diameter": 150.0, "inner_diameter": 30.0, "height": 10.0}
    
    console.print(f"   Ground Truth: {gt}")
    console.print(f"   Exact match score: {heuristic_param_score(gt, pred_exact):.3f}")
    console.print(f"   Close match score: {heuristic_param_score(gt, pred_close):.3f}")
    console.print(f"   Wrong match score: {heuristic_param_score(gt, pred_wrong):.3f}")
    
    # Demonstrate state management
    console.print("\n[bold]4. Agent State Management[/bold]")
    
    state: AgentState = {
        "text_desc": "A flange with 100mm outer diameter, 50mm inner diameter, 20mm height",
        "category": "Flange",
        "gt_param_spec": gt,
        "pred_param_spec": None,
        "param_score": None,
        "param_feedback": None,
        "cad_file": None,
        "render_image": None,
        "vlm_score": None,
        "vlm_feedback": None,
        "iteration": 0,
        "done": False,
        "sample_id": "flange_00001",
        "history": [],
    }
    
    console.print(f"   Initial state:")
    console.print(f"     text_desc: '{state['text_desc'][:50]}...'")
    console.print(f"     iteration: {state['iteration']}")
    console.print(f"     done: {state['done']}")
    
    # Check termination logic
    config = get_config()
    console.print(f"\n   Termination thresholds:")
    console.print(f"     param_threshold: {config.agent.param_score_threshold}")
    console.print(f"     vlm_threshold: {config.agent.vlm_score_threshold}")
    console.print(f"     max_iterations: {config.agent.max_iterations}")
    
    # Test with actual LLM if available
    console.print("\n[bold]5. Live LLM Parameter Generation[/bold]")
    
    if check_llm_available():
        console.print("   [green]LLM server available, using actual model[/green]")
        
        from src.utils.llm_client import get_llm_client
        import json
        
        client = get_llm_client()
        console.print(f"   Model: {client.model}")
        
        # Generate parameters from text description
        text_desc = "A flange with outer diameter 120mm, inner diameter 60mm, and height 15mm"
        
        # Use simple example-based prompt instead of schema (clearer for LLM)
        prompt = f"""Extract dimensions from this description and return as JSON.

Description: {text_desc}

Return ONLY a JSON object like this example (no explanation):
{{"outer_diameter": 100.0, "inner_diameter": 50.0, "height": 10.0}}"""

        console.print(f"\n   Input text: '{text_desc}'")
        console.print(f"   Generating parameters...")
        
        try:
            result = client.generate_json(prompt, max_tokens=200)
            console.print(f"   [green]LLM Output:[/green] {result}")
            
            # Calculate score against ground truth
            gt_for_test = {"outer_diameter": 120.0, "inner_diameter": 60.0, "height": 15.0}
            if isinstance(result, dict):
                score = heuristic_param_score(gt_for_test, result)
                console.print(f"   [green]Param Score:[/green] {score:.3f}")
        except Exception as e:
            console.print(f"   [red]Error: {e}[/red]")
    else:
        console.print("   [yellow]LLM server not available, skipping live test[/yellow]")
        console.print("   [dim]Set OPENAI_API_BASE=http://localhost:8001 and start vLLM[/dim]")
    
    # Build and compile the graph
    console.print("\n[bold]6. LangGraph Compilation[/bold]")
    
    if check_llm_available():
        # Use actual graph builder
        graph = build_design_agent()
        console.print(f"   Graph compiled with [green]actual LLM/VLM clients[/green]")
    else:
        # Mock clients
        from unittest.mock import MagicMock, patch
        with patch('src.agents.design_agent.get_llm_client') as mock_llm, \
             patch('src.agents.design_agent.get_vlm_client') as mock_vlm:
            mock_llm.return_value = MagicMock()
            mock_vlm.return_value = MagicMock()
            graph = build_design_agent()
        console.print(f"   Graph compiled with [yellow]mocked clients[/yellow]")
    
    console.print(f"   Entry point: generate_param_spec")
    console.print(f"   Nodes: generate_param_spec → judge_param_spec → generate_cad → judge_cad_vlm → decide_next_step")
    console.print(f"   Terminal: END")
    
    console.print("\n[green]✓ Design agent demo complete![/green]")


if __name__ == "__main__":
    main()
