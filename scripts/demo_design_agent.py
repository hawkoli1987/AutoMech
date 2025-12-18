#!/usr/bin/env python3
"""
Demo: Design Agent - Freeform CAD Generation

Shows the LangGraph design agent structure and flow for freeform code generation.
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


def main():
    console = Console()
    console.print(Panel.fit("[bold cyan]Design Agent Demo - Freeform CAD Generation[/bold cyan]"))
    
    from src.agents.design_agent import (
        DesignAgent,
        AgentState,
        build_design_agent,
    )
    from src.schemas import CadQueryCodeDesign
    from src.config import get_config
    
    # Show agent architecture
    console.print("\n[bold]1. Agent Architecture[/bold]")
    
    tree = Tree("[bold]DesignAgent (LangGraph)[/bold]")
    nodes = tree.add("[cyan]Nodes[/cyan]")
    nodes.add("generate_code - LLM generates CadQuery Python code from text")
    nodes.add("generate_cad - Execute code safely to create STEP/STL + render")
    nodes.add("judge_cad_vlm - VLM evaluates visual quality of render")
    nodes.add("decide_next_step - Check thresholds and decide if done")
    
    flow = tree.add("[cyan]Flow[/cyan]")
    flow.add("generate_code → generate_cad → judge_cad_vlm → decide_next_step")
    flow.add("If not done: loop back to generate_code with feedback")
    
    console.print(tree)
    
    # Show freeform code design schema
    console.print("\n[bold]2. Freeform Code Design Schema[/bold]")
    console.print("  [cyan]CadQueryCodeDesign[/cyan]: LLM-generated Python code for arbitrary parts")
    console.print("    - description: Human-readable design intent")
    console.print("    - code: Valid CadQuery Python code")
    console.print("    - entry_point: Variable name containing final Workplane (usually 'result')")
    console.print("    - required_imports: Python imports (cq is pre-injected)")
    console.print("    - comments: Additional design notes")
    
    # Show example code design
    console.print("\n[bold]3. Example Code Design[/bold]")
    example_design = CadQueryCodeDesign(
        description="A mounting bracket with two holes",
        code="""# Create base bracket
result = cq.Workplane('XY').box(50, 30, 5)

# Add two mounting holes
result = result.faces('>Z').workplane().pushPoints([(-15, 0), (15, 0)]).circle(4).cutThruAll()

# Add chamfers
result = result.edges('|Z').chamfer(1)""",
        entry_point="result",
        required_imports=["cadquery as cq"],
        comments="Simple L-bracket with chamfered edges"
    )
    
    console.print(f"  Description: {example_design.description}")
    console.print(f"  Entry Point: {example_design.entry_point}")
    console.print(f"  Code Preview:")
    for line in example_design.code.split('\n')[:5]:
        console.print(f"    {line}")
    console.print("    ...")
    
    # Show agent state
    console.print("\n[bold]4. Agent State[/bold]")
    console.print("  Initial state:")
    console.print("    - text_desc: 'A mounting bracket with two holes'")
    console.print("    - iteration: 0")
    console.print("    - done: False")
    console.print("  After iteration 1:")
    console.print("    - cad_file: 'artifacts/cad/bracket_iter0.step'")
    console.print("    - render_image: 'artifacts/renders/bracket_iter0.png'")
    console.print("    - vlm_score: 0.85")
    console.print("    - vlm_feedback: 'Good geometry, holes properly placed'")
    
    # Show termination logic
    console.print("\n[bold]5. Termination Logic[/bold]")
    config = get_config()
    console.print(f"  VLM threshold: {config.agent.vlm_threshold}")
    console.print(f"  Max iterations: {config.agent.max_iterations}")
    console.print("  Agent terminates when:")
    console.print("    - VLM score >= threshold, OR")
    console.print("    - Max iterations reached")
    
    # Check if LLM is available for live demo
    console.print("\n[bold]6. LLM Availability[/bold]")
    if check_llm_available():
        console.print("  [green]✓[/green] LLM server is available")
        console.print("  [dim]You can run the full agent with: python3 scripts/demo_freeform_cad.py[/dim]")
    else:
        console.print("  [yellow]✗[/yellow] LLM server not available")
        console.print("  [dim]Start LLM server to run full agent demo[/dim]")
    
    # Show graph compilation
    console.print("\n[bold]7. Graph Compilation[/bold]")
    try:
        agent = DesignAgent()
        console.print("  [green]✓[/green] LangGraph compiled successfully")
        console.print(f"  Nodes: {len(agent.graph.nodes)} nodes")
    except Exception as e:
        console.print(f"  [red]✗[/red] Failed to compile: {e}")
    
    console.print("\n[green]✓ Design agent demo complete![/green]")
    console.print("\n[dim]The agent uses freeform code generation to create arbitrary CAD models")
    console.print("beyond fixed templates, enabling truly flexible parametric design.[/dim]\n")


if __name__ == "__main__":
    main()
