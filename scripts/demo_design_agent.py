#!/usr/bin/env python3
"""
Demo: Design Agent

Shows the LangGraph design agent structure and flow.
Uses mocked LLM/VLM calls for demonstration.

Run from project root: python3 scripts/demo_design_agent.py
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.tree import Tree


def main():
    console = Console()
    console.print(Panel.fit("[bold cyan]Design Agent Demo[/bold cyan]"))
    
    from src.agents.design_agent import (
        DesignAgent,
        DesignAgentState,
        _get_category_schema,
        _heuristic_param_score,
    )
    from src.schemas import CADCategory, FlangeSpec
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
    for cat in CADCategory:
        schema = _get_category_schema(cat)
        console.print(f"   {cat.value}: {schema.__name__}")
    
    # Demonstrate heuristic scoring
    console.print("\n[bold]3. Heuristic Param Scoring[/bold]")
    
    gt = {"outer_diameter": 100.0, "inner_diameter": 50.0, "height": 20.0}
    pred_exact = {"outer_diameter": 100.0, "inner_diameter": 50.0, "height": 20.0}
    pred_close = {"outer_diameter": 102.0, "inner_diameter": 51.0, "height": 19.5}
    pred_wrong = {"outer_diameter": 150.0, "inner_diameter": 30.0, "height": 10.0}
    
    console.print(f"   Ground Truth: {gt}")
    console.print(f"   Exact match score: {_heuristic_param_score(gt, pred_exact):.3f}")
    console.print(f"   Close match score: {_heuristic_param_score(gt, pred_close):.3f}")
    console.print(f"   Wrong match score: {_heuristic_param_score(gt, pred_wrong):.3f}")
    
    # Demonstrate state management
    console.print("\n[bold]4. Agent State[/bold]")
    
    state = DesignAgentState(
        text_desc="A flange with 100mm outer diameter",
        category="Flange",
        gt_param_spec=gt,
        pred_param_spec=None,
        param_score=None,
        param_feedback=None,
        cad_file=None,
        render_image=None,
        vlm_score=None,
        vlm_feedback=None,
        iteration=0,
        done=False,
        metadata={"sample_id": "flange_00001"},
        history=[],
    )
    
    console.print(f"   Initial state:")
    console.print(f"     iteration: {state['iteration']}")
    console.print(f"     done: {state['done']}")
    console.print(f"     pred_param_spec: {state['pred_param_spec']}")
    
    # Simulate iteration
    state["iteration"] = 1
    state["pred_param_spec"] = pred_close
    state["param_score"] = _heuristic_param_score(gt, pred_close)
    state["vlm_score"] = 0.82
    
    console.print(f"\n   After iteration 1:")
    console.print(f"     iteration: {state['iteration']}")
    console.print(f"     param_score: {state['param_score']:.3f}")
    console.print(f"     vlm_score: {state['vlm_score']}")
    
    # Check termination
    config = get_config()
    should_terminate = (
        state["param_score"] >= config.agent.param_score_threshold and
        state["vlm_score"] >= config.agent.vlm_score_threshold
    )
    console.print(f"\n   Termination check:")
    console.print(f"     param_threshold: {config.agent.param_score_threshold}")
    console.print(f"     vlm_threshold: {config.agent.vlm_score_threshold}")
    console.print(f"     should_terminate: {should_terminate}")
    
    # Build graph (without running)
    console.print("\n[bold]5. LangGraph Compilation[/bold]")
    
    # Mock the clients to avoid actual API calls
    with patch('src.agents.design_agent.get_llm_client') as mock_llm, \
         patch('src.agents.design_agent.get_vlm_client') as mock_vlm:
        
        mock_llm.return_value = MagicMock()
        mock_vlm.return_value = MagicMock()
        
        agent = DesignAgent()
        graph = agent.build_design_agent()
        
        console.print(f"   Graph compiled: [green]✓[/green]")
        console.print(f"   Entry point: generate_param_spec")
        console.print(f"   Terminal: END")
    
    console.print("\n[green]✓ Design agent demo complete![/green]")
    console.print("\n[dim]To run with actual LLM/VLM, use the DatasetRunner.[/dim]")


if __name__ == "__main__":
    main()

